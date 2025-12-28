"""
FedEdgeServer - Edge Server Node Implementation

Handles:
    - Client coordination and training offload
    - Model splitting and aggregation
    - Bandwidth management (hardcoded/measured)
    - Decentralized and centralized FL modes
"""

import threading
import time
from typing import Dict, List

from torch import optim, nn

from app.config import config
from app.config.logger import fed_logger
from app.dto.bandwidth import BandWidth
from app.dto.base_model import BaseModel
from app.dto.message import (
    IterationFlagMessage,
    GlobalWeightMessage,
    SplitLayerConfigMessage,
    NetworkTestMessage
)
from app.entity.aggregators.base_aggregator import BaseAggregator
from app.entity.communicator import Communicator
from app.entity.fed_base_node_interface import FedBaseNodeInterface
from app.entity.http_communicator import HTTPCommunicator
from app.entity.node_identifier import NodeIdentifier
from app.entity.node_type import NodeType
from app.fl_method import fl_method_parser
from app.model.utils import get_available_torch_device
from app.util import model_utils, data_utils


class FedEdgeServer(FedBaseNodeInterface):
    """Edge server node for semi-decentralized federated learning"""

    def __init__(
            self,
            ip: str,
            port: int,
            model_name: str,
            dataset: str,
            offload: bool,
            aggregator: BaseAggregator,
            neighbors: List[NodeIdentifier]
    ):
        super().__init__(ip, port, NodeType.EDGE, '', neighbors)

        # Model and device setup
        self.device = get_available_torch_device()
        self.model_name = model_name
        self.dataset = dataset
        self.criterion = nn.CrossEntropyLoss()

        # Client-specific models and optimizers
        self.nets: Dict[NodeIdentifier, nn.Module] = {}
        self.optimizers: Dict[NodeIdentifier, optim.Optimizer] = {}
        self.scheduler: Dict[NodeIdentifier, optim.lr_scheduler.StepLR] = {}

        # FL configuration
        self.split_layers: Dict[NodeIdentifier, int] = {}
        self.group_labels = None
        self.offload = offload
        self.aggregator = aggregator

        # Communication
        self.central_server_communicator = Communicator()
        self.threads: Dict[NodeIdentifier, threading.Thread] = {}

        # Bandwidth tracking (inherited from base but explicitly typed)
        self.neighbor_bandwidth: Dict[NodeIdentifier, BandWidth] = {}

        # Global model and test data (if offloading enabled)
        if offload:
            model_len = model_utils.get_unit_model_len()
            self.uninet = model_utils.get_model(
                'Unit',
                [model_len - 1, model_len - 1],
                self.device,
                self.is_edge_based
            )
            self.testset = data_utils.get_testset()
            self.testloader = data_utils.get_testloader(self.testset, 0)
        else:
            self.uninet = None
            self.testset = None
            self.testloader = None

    @property
    def is_edge_based(self) -> bool:
        """Check if this edge is connected to a central server"""
        if self._edge_based is not None:
            return self._edge_based

        server_neighbors = self.get_neighbors([NodeType.SERVER])
        self._edge_based = len(server_neighbors) > 0
        return self._edge_based

    # ================================================================
    # INITIALIZATION
    # ================================================================

    def initialize(self, learning_rate: float):
        """Initialize client models and optimizers based on split configuration"""
        self.nets = {}
        self.optimizers = {}
        self.scheduler = {}

        # Initialize split layers for decentralized mode
        if not self.is_edge_based:
            self.initialize_split_layers()

        # Create client-specific models
        for neighbor in self.get_neighbors([NodeType.CLIENT]):
            split_point = self.split_layers[neighbor]

            # Ensure split_point is int (2-tier architecture)
            if isinstance(split_point, list):
                fed_logger.warning(
                    f"Converting list split_point to int for {neighbor}"
                )
                split_point = split_point[0]
                self.split_layers[neighbor] = split_point

            # Create edge-side model if offloading
            if split_point < len(self.uninet.cfg) - 1:
                self.nets[neighbor] = model_utils.get_model(
                    'Edge',
                    split_point,
                    self.device,
                    self.is_edge_based
                )

                # Initialize weights from global model
                cweights = model_utils.get_model(
                    'Client',
                    split_point,
                    self.device,
                    self.is_edge_based
                ).state_dict()

                pweights = model_utils.split_weights_server(
                    self.uninet.state_dict(),
                    cweights,
                    self.nets[neighbor].state_dict(),
                    []
                )
                self.nets[neighbor].load_state_dict(pweights)

                # Setup optimizer if model has parameters
                if len(list(self.nets[neighbor].parameters())) != 0:
                    self.optimizers[neighbor] = optim.SGD(
                        self.nets[neighbor].parameters(),
                        lr=learning_rate,
                        momentum=0.9,
                        weight_decay=5e-4
                    )
                    self.scheduler[neighbor] = optim.lr_scheduler.StepLR(
                        self.optimizers[neighbor],
                        config.lr_step_size,
                        config.lr_gamma
                    )
            else:
                # No offloading for this client
                self.nets[neighbor] = model_utils.get_model(
                    'Edge',
                    split_point,
                    self.device,
                    self.is_edge_based
                )

    def initialize_split_layers(self):
        """Initialize split configuration with default values (no offloading)"""
        model_len = model_utils.get_unit_model_len()
        default_split = model_len - 1  # All layers on client

        for neighbor in self.get_neighbors([NodeType.CLIENT]):
            self.split_layers[neighbor] = default_split

        fed_logger.info(
            f"[Edge {self.node_identifier}] Initialized split layers "
            f"with default: {default_split}"
        )

    # ================================================================
    # BANDWIDTH MANAGEMENT
    # ================================================================

    def gather_neighbors_network_bandwidth(self):
        """
        Gather bandwidth information from all neighbors.
        Uses hardcoded values if config.USE_HARDCODED_BW is True.
        """
        mode = "Hardcoded" if config.USE_HARDCODED_BW else "Measured"
        fed_logger.info(
            f"[Edge {self.node_identifier}] Gathering bandwidth ({mode} mode)"
        )

        for neighbor in self.get_neighbors([NodeType.CLIENT, NodeType.EDGE]):
            if config.USE_HARDCODED_BW:
                bandwidth = self._get_hardcoded_bandwidth(neighbor)
                fed_logger.info(
                    f"[Edge {self.node_identifier}] Hardcoded BW for {neighbor}: "
                    f"{bandwidth.to_mbps():.2f} Mbps"
                )
            else:
                bandwidth = self._measure_bandwidth(neighbor)
                fed_logger.info(
                    f"[Edge {self.node_identifier}] Measured BW for {neighbor}: "
                    f"{bandwidth.to_mbps():.2f} Mbps"
                )

            self.neighbor_bandwidth[neighbor] = bandwidth

    def _get_hardcoded_bandwidth(self, neighbor: NodeIdentifier) -> BandWidth:
        """
        Get hardcoded bandwidth for a neighbor.
        Priority: CLIENT_BANDWIDTH_MAP > Node Type Default > Fallback
        """
        # Check custom bandwidth map
        node_key = f"{neighbor.ip}_{neighbor.port}"

        if hasattr(config, 'CLIENT_BANDWIDTH_MAP') and node_key in config.CLIENT_BANDWIDTH_MAP:
            bw_value = config.CLIENT_BANDWIDTH_MAP[node_key]
            fed_logger.debug(
                f"Using custom hardcoded BW for {node_key}: {bw_value} Mbps"
            )
            return BandWidth(hardcoded_value=bw_value)

        # Determine node type for default value
        try:
            neighbor_type = HTTPCommunicator.get_node_type(neighbor)
        except Exception as e:
            fed_logger.warning(
                f"Failed to get node type for {neighbor}, "
                f"defaulting to CLIENT. Error: {e}"
            )
            neighbor_type = NodeType.CLIENT

        # Select default based on type
        if neighbor_type == NodeType.CLIENT:
            bw_value = config.HARDCODED_CLIENT_BW
        elif neighbor_type == NodeType.EDGE:
            bw_value = config.HARDCODED_EDGE_BW
        else:
            fed_logger.warning(
                f"Unknown node type {neighbor_type} for {neighbor}, "
                f"using CLIENT default"
            )
            bw_value = config.HARDCODED_CLIENT_BW

        return BandWidth(hardcoded_value=bw_value)

    def _measure_bandwidth(self, neighbor: NodeIdentifier) -> BandWidth:
        """
        Measure actual bandwidth by sending test data.
        Falls back to hardcoded value on error.
        """
        try:
            # Prepare 1KB test data
            test_data_size = 1024
            test_data = b'\0' * test_data_size

            # Measure transfer time
            start_time = time.time()

            test_msg = NetworkTestMessage([test_data])
            self.send_msg(
                self.get_exchange_name(),
                HTTPCommunicator.get_rabbitmq_url(neighbor),
                test_msg
            )

            # Wait for response
            response = self.recv_msg(
                neighbor.get_exchange_name(),
                config.current_node_mq_url,
                NetworkTestMessage.MESSAGE_TYPE
            )

            end_time = time.time()
            elapsed_time = end_time - start_time

            # Validate measurement
            if elapsed_time <= 0:
                fed_logger.warning(
                    f"Invalid elapsed time ({elapsed_time}s) for {neighbor}, "
                    f"using fallback"
                )
                return BandWidth(hardcoded_value=config.HARDCODED_CLIENT_BW)

            # Return measured bandwidth
            return BandWidth(
                transferred_bytes=test_data_size,
                time=elapsed_time
            )

        except Exception as e:
            fed_logger.error(
                f"Bandwidth measurement failed for {neighbor}: {e}. "
                f"Using fallback: {config.HARDCODED_CLIENT_BW} Mbps"
            )
            return BandWidth(hardcoded_value=config.HARDCODED_CLIENT_BW)

    def get_neighbors_bandwidth(self) -> Dict[NodeIdentifier, BandWidth]:
        """Get current bandwidth measurements for all neighbors"""
        return self.neighbor_bandwidth

    # ================================================================
    # MODEL SPLITTING
    # ================================================================

    def split(self, client_bandwidths: List[float], options: dict):
        """
        Calculate optimal split points based on client bandwidths.

        Args:
            client_bandwidths: List of bandwidth values in Mbps
            options: Configuration dict containing splitting method
        """
        splitting_method = options.get('splitting', 'optimal_split')
        split_func = fl_method_parser.fl_methods.get(splitting_method)

        if split_func is None:
            fed_logger.error(
                f"Splitting method '{splitting_method}' not found!"
            )
            raise ValueError(f"Unknown splitting method: {splitting_method}")

        # Call splitting function with node reference
        self.split_layers = split_func(
            client_bandwidths,
            self.group_labels,
            node=self
        )

        # Ensure all split points are int (2-tier architecture)
        for client, split_point in self.split_layers.items():
            if isinstance(split_point, list):
                fed_logger.warning(
                    f"Converting list split to int for {client}"
                )
                self.split_layers[client] = split_point[0]

        fed_logger.info(f"Next Round Split Points: {self.split_layers}")

    def clustering(self, options: dict):
        """
        Perform client clustering (disabled in 2-tier architecture).

        Args:
            options: Configuration dict
        """
        if options.get('clustering') and options.get('clustering') != 'none':
            fed_logger.warning(
                "Clustering not supported in 2-tier mode, skipping..."
            )
        self.group_labels = None

    # ================================================================
    # CENTRALIZED MODE (LEGACY 3-TIER)
    # ================================================================

    def gather_and_scatter_split_config(self):
        """Receive split config from server and distribute to clients"""
        received_messages = self.gather_msgs(
            SplitLayerConfigMessage.MESSAGE_TYPE,
            [NodeType.SERVER]
        )
        msg: SplitLayerConfigMessage = received_messages[0].message
        self.split_layers = msg.data
        self.scatter_split_layers([NodeType.CLIENT])

    def gather_and_scatter_global_weight(self):
        """Receive global model from server and distribute to clients"""
        received_messages = self.gather_msgs(
            GlobalWeightMessage.MESSAGE_TYPE,
            [NodeType.SERVER]
        )
        msg: GlobalWeightMessage = received_messages[0].message
        weights = msg.weights[0]

        # Update edge models and distribute to clients
        for neighbor in self.get_neighbors([NodeType.CLIENT]):
            split_point = self.split_layers[neighbor]

            # Handle 2-tier split (int, not list)
            if isinstance(split_point, list):
                split_point = split_point[0]

            cweights = model_utils.get_model(
                'Client',
                split_point,
                self.device,
                self.is_edge_based
            ).state_dict()

            pweights = model_utils.split_weights_edgeserver(
                weights,
                cweights,
                self.nets[neighbor].state_dict()
            )
            self.nets[neighbor].load_state_dict(pweights)

        self.scatter_msg(GlobalWeightMessage([weights]), [NodeType.CLIENT])

    def start_centralized_training(self):
        """Start centralized training with all clients (legacy mode)"""
        self.threads = {}
        client_neighbors = self.get_neighbors([NodeType.CLIENT])

        for neighbor in client_neighbors:
            self.threads[neighbor] = threading.Thread(
                target=self._thread_centralized_training,
                args=(neighbor,),
                name=str(neighbor)
            )
            fed_logger.info(f"{neighbor} centralized training start")
            self.threads[neighbor].start()

        fed_logger.info("Waiting for centralized training to finish")
        for neighbor in client_neighbors:
            self.threads[neighbor].join()
        fed_logger.info("Centralized training finished")

    def _thread_centralized_training(self, neighbor: NodeIdentifier):
        """Handle centralized training for a single client"""
        self._forward_propagation(neighbor)
        self._send_back_local_weight(neighbor)

    def _forward_propagation(self, neighbor: NodeIdentifier):
        """Handle forward propagation in centralized mode"""
        msg: IterationFlagMessage = self.recv_msg(
            neighbor.get_exchange_name(),
            config.current_node_mq_url,
            IterationFlagMessage.MESSAGE_TYPE
        )

        server_neighbor = self.get_neighbors([NodeType.SERVER])[0]
        edge_exchange = self.get_exchange_name(neighbor)
        flag: bool = msg.flag

        # Convert split_point to list for 3-tier compatibility
        split_point = self.split_layers[neighbor]
        if not isinstance(split_point, list):
            split_layers_list = [split_point, split_point]
        else:
            split_layers_list = split_point

        # Send initial flag to server
        if split_layers_list[1] < model_utils.get_unit_model_len() - 1:
            self.send_msg(
                edge_exchange,
                HTTPCommunicator.get_rabbitmq_url(server_neighbor),
                IterationFlagMessage(flag)
            )
        else:
            self.send_msg(
                edge_exchange,
                HTTPCommunicator.get_rabbitmq_url(server_neighbor),
                IterationFlagMessage(False)
            )

        # Training loop
        while flag:
            if split_layers_list[0] < model_utils.get_unit_model_len() - 1:
                msg: IterationFlagMessage = self.recv_msg(
                    neighbor.get_exchange_name(),
                    config.current_node_mq_url,
                    IterationFlagMessage.MESSAGE_TYPE
                )
                flag = msg.flag

                if not flag:
                    self.send_msg(
                        edge_exchange,
                        HTTPCommunicator.get_rabbitmq_url(server_neighbor),
                        IterationFlagMessage(flag)
                    )
                    break

                # Receive smashed data from client
                msg: GlobalWeightMessage = self.recv_msg(
                    neighbor.get_exchange_name(),
                    config.current_node_mq_url,
                    GlobalWeightMessage.MESSAGE_TYPE
                )
                smashed_layers = msg.weights[0]
                labels = msg.weights[1]

                inputs = smashed_layers.to(self.device)
                targets = labels.to(self.device)

                # Process based on split configuration
                if split_layers_list[0] < split_layers_list[1]:
                    if neighbor in self.optimizers:
                        self.optimizers[neighbor].zero_grad()

                    outputs = self.nets[neighbor](inputs)

                    if split_layers_list[1] < model_utils.get_unit_model_len() - 1:
                        # Forward to server
                        self.send_msg(
                            edge_exchange,
                            HTTPCommunicator.get_rabbitmq_url(server_neighbor),
                            IterationFlagMessage(flag)
                        )

                        msg_data = [outputs.to(self.device), targets.to(self.device)]
                        self.send_msg(
                            edge_exchange,
                            HTTPCommunicator.get_rabbitmq_url(server_neighbor),
                            GlobalWeightMessage(msg_data)
                        )

                        # Receive gradients from server
                        msg: GlobalWeightMessage = self.recv_msg(
                            edge_exchange,
                            config.current_node_mq_url,
                            GlobalWeightMessage.MESSAGE_TYPE
                        )
                        gradients = msg.weights[0].to(self.device)
                        outputs.backward(gradients)

                        # Send gradients to client
                        msg_data = [inputs.grad]
                        self.send_msg(
                            self.get_exchange_name(),
                            HTTPCommunicator.get_rabbitmq_url(neighbor),
                            GlobalWeightMessage(msg_data)
                        )
                    else:
                        # Train locally on edge
                        outputs = self.nets[neighbor](inputs)
                        loss = self.criterion(outputs, targets)
                        loss.backward()

                        if neighbor in self.optimizers:
                            self.optimizers[neighbor].step()

                        # Send gradients to client
                        msg_data = [inputs.grad]
                        self.send_msg(
                            self.get_exchange_name(),
                            HTTPCommunicator.get_rabbitmq_url(neighbor),
                            GlobalWeightMessage(msg_data)
                        )
                else:
                    # Forward directly to server
                    self.send_msg(
                        edge_exchange,
                        HTTPCommunicator.get_rabbitmq_url(server_neighbor),
                        IterationFlagMessage(flag)
                    )

                    msg_data = [inputs.cpu(), targets.cpu()]
                    self.send_msg(
                        edge_exchange,
                        HTTPCommunicator.get_rabbitmq_url(server_neighbor),
                        GlobalWeightMessage(msg_data)
                    )

                    # Forward server response to client
                    msg: GlobalWeightMessage = self.recv_msg(
                        neighbor.get_exchange_name(),
                        config.current_node_mq_url,
                        GlobalWeightMessage.MESSAGE_TYPE
                    )
                    self.send_msg(
                        self.get_exchange_name(),
                        HTTPCommunicator.get_rabbitmq_url(neighbor),
                        msg
                    )

        fed_logger.info(f"{neighbor} centralized training end")

    def _send_back_local_weight(self, neighbor: NodeIdentifier):
        """Send aggregated local weights to server"""
        cweights = self.recv_msg(
            neighbor.get_exchange_name(),
            config.current_node_mq_url,
            GlobalWeightMessage.MESSAGE_TYPE
        ).weights[0]

        server_neighbor = self.get_neighbors([NodeType.SERVER])[0]

        # Handle 2-tier split_point
        split_point = self.split_layers[neighbor]
        if isinstance(split_point, list):
            split_point = split_point[0]

        # Concatenate weights if offloading
        if split_point != (config.model_len - 1):
            w_local = model_utils.concat_weights(
                self.uninet.state_dict(),
                cweights,
                self.nets[neighbor].state_dict()
            )
        else:
            w_local = cweights

        msg = GlobalWeightMessage([w_local])
        self.send_msg(
            self.get_exchange_name(neighbor),
            HTTPCommunicator.get_rabbitmq_url(server_neighbor),
            msg
        )

    # ================================================================
    # DECENTRALIZED MODE (2-TIER)
    # ================================================================

    def start_decentralized_training(self):
        """Start decentralized training with all clients"""
        self.threads = {}
        client_neighbors = self.get_neighbors([NodeType.CLIENT])

        for neighbor in client_neighbors:
            self.threads[neighbor] = threading.Thread(
                target=self._thread_decentralized_training,
                args=(neighbor,),
                name=str(neighbor)
            )
            fed_logger.info(f"{neighbor} decentralized training start")
            self.threads[neighbor].start()

        fed_logger.info("Waiting for decentralized training to finish")
        for neighbor in client_neighbors:
            self.threads[neighbor].join()
        fed_logger.info("Decentralized training finished")

    def _thread_decentralized_training(self, neighbor: NodeIdentifier):
        """Handle decentralized training for a single client"""
        neighbor_rabbitmq_url = HTTPCommunicator.get_rabbitmq_url(neighbor)

        # Receive initial flag
        flag: bool = self.recv_msg(
            neighbor.get_exchange_name(),
            config.current_node_mq_url,
            IterationFlagMessage.MESSAGE_TYPE
        ).flag

        # Training loop
        while flag:
            # Receive next iteration flag
            flag = self.recv_msg(
                neighbor.get_exchange_name(),
                config.current_node_mq_url,
                IterationFlagMessage.MESSAGE_TYPE
            ).flag

            if not flag:
                break

            # Receive smashed data
            msg: GlobalWeightMessage = self.recv_msg(
                neighbor.get_exchange_name(),
                config.current_node_mq_url,
                GlobalWeightMessage.MESSAGE_TYPE
            )
            smashed_layers = msg.weights[0]
            labels = msg.weights[1]

            inputs = smashed_layers.to(self.device)
            targets = labels.to(self.device)

            # Get split point (ensure it's int)
            split_point = self.split_layers[neighbor]
            if isinstance(split_point, list):
                split_point = split_point[0]

            # Forward and backward pass
            if split_point < len(self.uninet.cfg) - 1:
                if neighbor in self.optimizers:
                    self.optimizers[neighbor].zero_grad()

            outputs = self.nets[neighbor](inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()

            if split_point < len(self.uninet.cfg) - 1:
                if neighbor in self.optimizers:
                    self.optimizers[neighbor].step()
                    self.scheduler[neighbor].step()

            # Send gradients back to client
            fed_logger.info(f"{neighbor} sending gradients")
            msg = GlobalWeightMessage([inputs.grad])
            self.send_msg(
                self.get_exchange_name(),
                neighbor_rabbitmq_url,
                msg
            )

        fed_logger.info(f"{neighbor} decentralized training end")

    # ================================================================
    # AGGREGATION
    # ================================================================

    def gather_local_weights(self) -> Dict[NodeIdentifier, BaseModel]:
        """Collect local model weights from all clients"""
        client_local_weights = {}

        for neighbor in self.get_neighbors([NodeType.CLIENT]):
            msg: GlobalWeightMessage = self.recv_msg(
                neighbor.get_exchange_name(),
                config.current_node_mq_url,
                GlobalWeightMessage.MESSAGE_TYPE
            )
            client_local_weights[neighbor] = msg.weights[0]

        return client_local_weights

    def aggregate(self, client_local_weights: Dict[NodeIdentifier, BaseModel]):
        """
        Aggregate client weights into global model.

        Args:
            client_local_weights: Dict mapping clients to their local weights
        """
        zero_model = model_utils.zero_init(self.uninet).state_dict()
        w_local_list = self._concat_neighbor_local_weights(client_local_weights)
        aggregated_model = self.aggregator.aggregate(zero_model, w_local_list)
        self.uninet.load_state_dict(aggregated_model)

    def _concat_neighbor_local_weights(
            self,
            client_local_weights: Dict[NodeIdentifier, BaseModel]
    ) -> List[tuple]:
        """
        Concatenate client and edge weights for aggregation.

        Args:
            client_local_weights: Client local weights

        Returns:
            List of (weights, sample_weight) tuples
        """
        w_local_list = []
        client_neighbors = self.get_neighbors([NodeType.CLIENT])

        for neighbor in client_neighbors:
            # Handle 2-tier split_point (int)
            split_point = self.split_layers[neighbor]
            if isinstance(split_point, list):
                split_point = split_point[0]

            # Calculate sample weight
            sample_weight = config.N / len(client_neighbors)

            # Concatenate weights if offloading
            if self.offload and split_point != (config.model_len - 1):
                w_local = model_utils.concat_weights(
                    self.uninet.state_dict(),
                    client_local_weights[neighbor],
                    self.nets[neighbor].state_dict()
                )
            else:
                w_local = client_local_weights[neighbor]

            w_local_list.append((w_local, sample_weight))

        return w_local_list

    # ================================================================
    # GOSSIP (EDGE-TO-EDGE COMMUNICATION)
    # ================================================================

    def gossip_with_neighbors(self):
        """Exchange and aggregate models with neighboring edge servers"""
        edge_neighbors = self.get_neighbors([NodeType.EDGE])

        if len(edge_neighbors) == 0:
            return

        # Send local model to neighbors
        msg = GlobalWeightMessage([self.uninet.to(self.device).state_dict()])
        self.scatter_msg(msg, [NodeType.EDGE])

        # Receive models from neighbors
        gathered_msgs = self.gather_msgs(
            GlobalWeightMessage.MESSAGE_TYPE,
            [NodeType.EDGE]
        )

        # Prepare for aggregation
        gathered_models = [
            (msg.message.weights[0], config.N / len(edge_neighbors))
            for msg in gathered_msgs
        ]

        # Aggregate
        zero_model = model_utils.zero_init(self.uninet).state_dict()
        aggregated_model = self.aggregator.aggregate(zero_model, gathered_models)
        self.uninet.load_state_dict(aggregated_model)
