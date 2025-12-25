# fed_client.py

import time

import torch.nn as nn
from torch import optim
from tqdm import tqdm

from app.config import config
from app.config.logger import fed_logger
from app.dto.message import (
    GlobalWeightMessage,
    NetworkTestMessage,
    SplitLayerConfigMessage,
    IterationFlagMessage
)
from app.dto.received_message import ReceivedMessage
from app.entity.aggregators.base_aggregator import BaseAggregator
from app.entity.fed_base_node_interface import FedBaseNodeInterface
from app.entity.http_communicator import HTTPCommunicator
from app.entity.mobility_manager import MobilityManager
from app.entity.node_identifier import NodeIdentifier
from app.entity.node_type import NodeType
from app.model.utils import get_available_torch_device
from app.util import model_utils


# noinspection PyTypeChecker
class FedClient(FedBaseNodeInterface):
    """Federated Learning Client with adaptive split learning support"""

    def __init__(
            self,
            ip: str,
            port: int,
            model_name: str,
            dataset: str,
            train_loader,
            LR: float,
            cluster: str,
            aggregator: BaseAggregator,
            neighbors: list[NodeIdentifier] = None
    ):
        super().__init__(ip, port, NodeType.CLIENT, cluster, neighbors)

        self._edge_based = None
        self.device = get_available_torch_device()
        self.model_name = model_name
        self.dataset = dataset
        self.train_loader = train_loader
        self.split_layers = None
        self.criterion = nn.CrossEntropyLoss()
        self.mobility_manager = MobilityManager(self)
        self.aggregator = aggregator

        # Initialize complete model for local training
        self.uninet = model_utils.get_model('Unit', None, self.device, self.is_edge_based)
        self.net = self.uninet

        # Setup optimizer and scheduler
        self.optimizer = optim.SGD(self.net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, config.lr_step_size, config.lr_gamma)

    def initialize(self, learning_rate: float):
        """Initialize client-side split model after receiving split configuration"""
        self.net = model_utils.get_model('Client', self.split_layers, self.device, self.is_edge_based)
        self.optimizer = optim.SGD(self.net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

    def gather_global_weights(self, node_type: NodeType):
        """Download and load global model weights from server/edge"""
        msgs: list[ReceivedMessage] = self.gather_msgs(GlobalWeightMessage.MESSAGE_TYPE, [node_type])
        msg: GlobalWeightMessage = msgs[0].message

        # Extract client-side weights from global model
        pweights = model_utils.split_weights_client(msg.weights[0], self.net.state_dict())
        self.net.load_state_dict(pweights)

    def scatter_network_speed_to_edges(self):
        """Send test data to edge for bandwidth measurement (or use hardcoded BW)"""
        msg = NetworkTestMessage([self.net.to(self.device).state_dict()])
        self.scatter_msg(msg, [NodeType.EDGE])
        fed_logger.info("test network sent")

        # Wait for acknowledgment
        _ = self.gather_msgs(NetworkTestMessage.MESSAGE_TYPE, [NodeType.EDGE])
        fed_logger.info("test network received")

    def gather_split_config(self):
        """Receive optimal split point from edge server"""
        msgs = self.gather_msgs(SplitLayerConfigMessage.MESSAGE_TYPE, [NodeType.EDGE])
        msg: SplitLayerConfigMessage = msgs[0].message
        self.split_layers = msg.data

    def start_offloading_train(self):
        """Execute training with adaptive offloading based on split configuration"""
        self.net.to(self.device)
        self.net.train()

        # Handle both int and list split_layers format
        split_point = self.split_layers
        if isinstance(self.split_layers, list):
            split_point = self.split_layers[0]

        # Mode 1: No offloading - full local training
        if split_point == model_utils.get_unit_model_len() - 1:
            fed_logger.info("no offloading training start----------------------------")
            self.scatter_msg(IterationFlagMessage(False), [NodeType.EDGE])

            for batch_idx, (inputs, targets) in enumerate(tqdm(self.train_loader)):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

            self.scheduler.step()

        # Mode 2: Split learning with edge offloading
        elif split_point < model_utils.get_unit_model_len() - 1:
            fed_logger.info(f"offloading training start {self.split_layers}----------------------------")
            self.scatter_msg(IterationFlagMessage(True), [NodeType.EDGE])

            for batch_idx, (inputs, targets) in enumerate(tqdm(self.train_loader)):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Forward pass through client-side layers
                if self.optimizer is not None:
                    self.optimizer.zero_grad()
                outputs = self.net(inputs)

                # Send iteration flag
                self.scatter_msg(IterationFlagMessage(True), [NodeType.EDGE])

                # Send intermediate activations and labels to edge
                msg = GlobalWeightMessage([outputs.to(self.device), targets.to(self.device)])
                self.scatter_msg(msg, [NodeType.EDGE])

                # Receive gradients from edge
                fed_logger.info("receiving gradients")
                msgs: list[ReceivedMessage] = self.gather_msgs(GlobalWeightMessage.MESSAGE_TYPE, [NodeType.EDGE])
                msg: GlobalWeightMessage = msgs[0].message
                gradients = msg.weights[0].to(self.device)
                fed_logger.info("received gradients")

                # Backward pass through client-side layers
                outputs.backward(gradients)
                if self.optimizer is not None:
                    self.optimizer.step()

            self.scheduler.step()
            self.scatter_msg(IterationFlagMessage(False), [NodeType.EDGE])

    def scatter_local_weights(self):
        """Upload client model weights to edge server"""
        self.scatter_msg(GlobalWeightMessage([self.net.to(self.device).state_dict()]), [NodeType.EDGE])

    def scatter_random_local_weights(self):
        """Upload weights to server if elected as cluster leader"""
        server_neighbors = self.get_neighbors([NodeType.SERVER])
        if server_neighbors:
            server = server_neighbors[0]
            fed_logger.info("waiting for leader election to complete")

            # Wait for leader election
            while not HTTPCommunicator.get_leader_election_completed(server):
                fed_logger.info("leader election not completed yet, waiting...")
                time.sleep(1)
            fed_logger.info("leader election completed")

        # Only leader uploads to server
        is_leader = HTTPCommunicator.get_is_leader(self)
        if is_leader:
            self.scatter_msg(GlobalWeightMessage([self.net.to(self.device).state_dict()]), [NodeType.SERVER])

    def no_offloading_train(self):
        """Execute full local training without split learning"""
        self.net.to(self.device)
        self.net.train()

        for batch_idx, (inputs, targets) in enumerate(tqdm(self.train_loader)):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.net(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

    def gossip_with_neighbors(self):
        """Exchange and aggregate models with neighboring clients (D2D mode)"""
        client_neighbors = self.get_neighbors([NodeType.CLIENT])

        # Broadcast local model to neighbors
        msg = GlobalWeightMessage([self.uninet.to(self.device).state_dict()])
        self.scatter_msg(msg, [NodeType.CLIENT])

        # Collect neighbor models
        gathered_msgs = self.gather_msgs(GlobalWeightMessage.MESSAGE_TYPE, [NodeType.CLIENT])
        gathered_models = [(msg.message.weights[0], config.N / len(client_neighbors)) for msg in gathered_msgs]

        # Aggregate and update local model
        zero_model = model_utils.zero_init(self.uninet).state_dict()
        aggregated_model = self.aggregator.aggregate(zero_model, gathered_models)
        self.uninet.load_state_dict(aggregated_model)
