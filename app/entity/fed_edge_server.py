# fed_edge_server.py

import threading
import time

from torch import optim, nn

from app.config import config
from app.config.logger import fed_logger
from app.entity.bandwidth import BandWidth  # Updated import path
from app.dto.base_model import BaseModel
from app.dto.message import IterationFlagMessage, GlobalWeightMessage, SplitLayerConfigMessage, NetworkTestMessage
from app.entity.aggregators.base_aggregator import BaseAggregator
from app.entity.communicator import Communicator
from app.entity.fed_base_node_interface import FedBaseNodeInterface
from app.entity.http_communicator import HTTPCommunicator
from app.entity.node_identifier import NodeIdentifier
from app.entity.node_type import NodeType
from app.fl_method import fl_method_parser
from app.model.utils import get_available_torch_device
from app.util import model_utils, data_utils


# noinspection PyTypeChecker
class FedEdgeServer(FedBaseNodeInterface):

    def __init__(self, ip: str, port: int, model_name, dataset, offload, aggregator: BaseAggregator,
                 neighbors: list[NodeIdentifier]):
        super().__init__(ip, port, NodeType.EDGE, '', neighbors)
        self._edge_based = None
        self.device = get_available_torch_device()
        self.model_name = model_name
        self.nets = {}
        self.group_labels = None
        self.criterion = nn.CrossEntropyLoss()
        self.split_layers = None
        self.state = None
        self.client_bandwidth = {}
        self.dataset = dataset
        self.threads = None
        self.net_threads = None
        self.central_server_communicator = Communicator()
        self.offload = offload
        self.aggregator = aggregator

        if offload:
            model_len = model_utils.get_unit_model_len()
            self.uninet = model_utils.get_model('Unit', [model_len - 1, model_len - 1], self.device, self.is_edge_based)

            self.testset = data_utils.get_testset()
            self.testloader = data_utils.get_testloader(self.testset, 0)
        self.neighbor_bandwidth: dict[NodeIdentifier, BandWidth] = {}
        self.optimizers = None
        self.split_layers = {}

    @property
    def is_edge_based(self) -> bool:
        if self._edge_based is not None:
            return self._edge_based
        server_neighbors = self.get_neighbors([NodeType.SERVER])
        self._edge_based = len(server_neighbors) > 0
        return self._edge_based

    def initialize(self, learning_rate):
        self.nets = {}
        self.optimizers = {}
        self.scheduler = {}

        # Initialize split layers for decentralized mode
        if not self.is_edge_based:
            self.initialize_split_layers()

        for neighbor in self.get_neighbors([NodeType.CLIENT]):
            split_point = self.split_layers[neighbor]

            # Handle 2-tier: split_point is int, not list
            if isinstance(split_point, list):
                split_point = split_point[0]

            if split_point < len(self.uninet.cfg) - 1:
                self.nets[neighbor] = model_utils.get_model('Edge', split_point, self.device,
                                                            self.is_edge_based)
                cweights = model_utils.get_model('Client', split_point, self.device,
                                                 self.is_edge_based).state_dict()
                pweights = model_utils.split_weights_server(self.uninet.state_dict(), cweights,
                                                            self.nets[neighbor].state_dict(), [])
                self.nets[neighbor].load_state_dict(pweights)

                if len(list(self.nets[neighbor].parameters())) != 0:
                    self.optimizers[neighbor] = optim.SGD(self.nets[neighbor].parameters(), lr=learning_rate,
                                                          momentum=0.9, weight_decay=5e-4)
                    self.scheduler[neighbor] = optim.lr_scheduler.StepLR(self.optimizers[neighbor],
                                                                         config.lr_step_size, config.lr_gamma)
            else:
                self.nets[neighbor] = model_utils.get_model('Edge', split_point, self.device,
                                                            self.is_edge_based)

    def initialize_split_layers(self):
        """Initialize split_layers with default values for decentralized mode."""
        model_len = model_utils.get_unit_model_len()
        default_split = model_len - 1  # No offloading by default

        for neighbor in self.get_neighbors([NodeType.CLIENT]):
            self.split_layers[neighbor] = default_split

        fed_logger.info(
            f"[Edge {self.node_identifier}] Initialized split layers with default value: {default_split}"
        )

    # ================================================================
    # BANDWIDTH MANAGEMENT - NEW METHODS
    # ================================================================

    def gather_neighbors_network_bandwidth(self):
        """
        Gather network bandwidth information from neighbors.
        Supports both hardcoded and measured modes based on config.USE_HARDCODED_BW.
        """
        mode_str = 'Hardcoded' if config.USE_HARDCODED_BW else 'Measured'
        fed_logger.info(
            f"[Edge {self.node_identifier}] Gathering network bandwidth (Mode: {mode_str})"
        )

        for neighbor in self.get_neighbors([NodeType.CLIENT, NodeType.EDGE]):
            if config.USE_HARDCODED_BW:
                # Use hardcoded bandwidth values
                bandwidth = self._get_hardcoded_bandwidth(neighbor)
                fed_logger.info(
                    f"[Edge {self.node_identifier}] Hardcoded BW for {neighbor}: "
                    f"{bandwidth.to_mbps():.2f} Mbps ({bandwidth.to_mbytes_per_sec():.2f} MB/s)"
                )
            else:
                # Measure actual bandwidth
                bandwidth = self._measure_bandwidth(neighbor)
                fed_logger.info(
                    f"[Edge {self.node_identifier}] Measured BW for {neighbor}: "
                    f"{bandwidth.to_mbps():.2f} Mbps ({bandwidth.to_mbytes_per_sec():.2f} MB/s)"
                )

            self.neighbor_bandwidth[neighbor] = bandwidth

    def _get_hardcoded_bandwidth(self, neighbor: NodeIdentifier) -> BandWidth:
        """
        Get hardcoded bandwidth value for a neighbor.
        Priority: Custom Map > Node Type Default > CLIENT Default
        """
        # Check custom bandwidth map first
        node_key = f"{neighbor.ip}_{neighbor.port}"
        if node_key in config.HARDCODED_BW_MAP:
            bw_value = config.HARDCODED_BW_MAP[node_key]
            fed_logger.debug(f"Using custom hardcoded BW for {node_key}: {bw_value}")
            return BandWidth(hardcoded_value=bw_value)

        # Determine node type for default value
        try:
            neighbor_type = HTTPCommunicator.get_node_type(neighbor)
        except Exception as e:
            fed_logger.warning(
                f"Failed to determine node type for {neighbor}, defaulting to CLIENT. Error: {e}"
            )
            neighbor_type = NodeType.CLIENT

        # Select default bandwidth based on node type
        if neighbor_type == NodeType.CLIENT:
            bw_value = config.HARDCODED_CLIENT_BW
        elif neighbor_type == NodeType.EDGE:
            bw_value = config.HARDCODED_EDGE_BW
        else:
            fed_logger.warning(
                f"Unknown node type {neighbor_type} for {neighbor}, using CLIENT BW default"
            )
            bw_value = config.HARDCODED_CLIENT_BW

        return BandWidth(hardcoded_value=bw_value)

    def _measure_bandwidth(self, neighbor: NodeIdentifier) -> BandWidth:
        """
        Measure actual bandwidth by sending test data to neighbor.
        Falls back to hardcoded value on error.
        """
        try:
            # Prepare test data (1 KB)
            test_data_size = 1024  # bytes
            test_data = b'\0' * test_data_size

            # Send test message and measure time
            start_time = time.time()

            test_msg = NetworkTestMessage([test_data])
            self.send_msg(
                self.get_exchange_name(),
                HTTPCommunicator.get_rabbitmq_url(neighbor),
                test_msg
            )

            # Wait for acknowledgment
            response = self.recv_msg(
                neighbor.get_exchange_name(),
                config.current_node_mq_url,
                NetworkTestMessage.MESSAGE_TYPE
            )

            end_time = time.time()
            elapsed_time = end_time - start_time

            # Validate elapsed time
            if elapsed_time <= 0:
                fed_logger.warning(
                    f"Invalid elapsed time ({elapsed_time}s) for {neighbor}, "
                    f"falling back to hardcoded BW"
                )
                return BandWidth(hardcoded_value=config.HARDCODED_CLIENT_BW)

            # Create BandWidth object with measured values
            return BandWidth(
                transferred_bytes=test_data_size,
                time=elapsed_time
            )

        except Exception as e:
            fed_logger.error(
                f"Bandwidth measurement failed for {neighbor}: {e}. "
                f"Using fallback hardcoded value: {config.HARDCODED_CLIENT_BW}"
            )
            return BandWidth(hardcoded_value=config.HARDCODED_CLIENT_BW)

    # ================================================================
    # EXISTING METHODS (UPDATED WHERE NEEDED)
    # ================================================================

    def gather_and_scatter_global_weight(self):
        received_messages = self.gather_msgs(GlobalWeightMessage.MESSAGE_TYPE, [NodeType.SERVER])
        msg: GlobalWeightMessage = received_messages[0].message
        weights = msg.weights[0]
        for neighbor in self.get_neighbors([NodeType.CLIENT]):
            # Handle 2-tier split_layers (int, not list)
            split_point = self.split_layers[neighbor]
            if isinstance(split_point, list):
                split_point = split_point[0]

            cweights = model_utils.get_model('Client', split_point, self.device,
                                             self.is_edge_based).state_dict()
            pweights = model_utils.split_weights_edgeserver(weights, cweights,
                                                            self.nets[neighbor].state_dict())
            self.nets[neighbor].load_state_dict(pweights)
        self.scatter_msg(GlobalWeightMessage([weights]), [NodeType.CLIENT])

    def clustering(self, options: dict):
        # Clustering disabled for 2-tier architecture
        if options.get('clustering') and options.get('clustering') != 'none':
            fed_logger.warning("⚠️ Clustering not supported in 2-tier mode, skipping...")
            self.group_labels = None
        else:
            self.group_labels = None

    def get_neighbors_bandwidth(self) -> dict[NodeIdentifier, BandWidth]:
        return self.neighbor_bandwidth

    def split(self, state, options: dict):
        """
        Invoke splitting method with proper kwargs.
        Pass node=self for hardcoded BW support.
        """
        splitting_method = options.get('splitting', 'optimal_split')
        split_func = fl_method_parser.fl_methods.get(splitting_method)

        if split_func is None:
            fed_logger.error(f"❌ Splitting method '{splitting_method}' not found!")
            raise ValueError(f"Unknown splitting method: {splitting_method}")

        # Call with node kwarg for hardcoded BW access
        self.split_layers = split_func(state, self.group_labels, node=self)

        # Validate: ensure all values are int (not list)
        for client, split_point in self.split_layers.items():
            if isinstance(split_point, list):
                fed_logger.warning(f"⚠️ Converting list split to int for {client}")
                self.split_layers[client] = split_point[0]

        fed_logger.info('Next Round Split Points: ' + str(self.split_layers))

    def gather_and_scatter_split_config(self):
        received_messages = self.gather_msgs(SplitLayerConfigMessage.MESSAGE_TYPE, [NodeType.SERVER])
        msg: SplitLayerConfigMessage = received_messages[0].message
        self.split_layers = msg.data
        self.scatter_split_layers([NodeType.CLIENT])

    def start_decentralized_training(self):
        self.threads = {}
        client_neighbors = self.get_neighbors([NodeType.CLIENT])
        for neighbor in client_neighbors:
            self.threads[neighbor] = threading.Thread(target=self._thread_decentralized_training,
                                                      args=(neighbor,), name=str(neighbor))
            fed_logger.info(str(neighbor) + ' offloading training start')
            self.threads[neighbor].start()

        fed_logger.info('waiting for offloading training to finish')
        for neighbor in client_neighbors:
            self.threads[neighbor].join()
        fed_logger.info('offloading training finished')

    def _thread_decentralized_training(self, neighbor: NodeIdentifier):
        neighbor_rabbitmq_url = HTTPCommunicator.get_rabbitmq_url(neighbor)
        flag: bool = self.recv_msg(neighbor.get_exchange_name(), config.current_node_mq_url,
                                   IterationFlagMessage.MESSAGE_TYPE).flag
        while flag:
            flag = self.recv_msg(neighbor.get_exchange_name(), config.current_node_mq_url,
                                 IterationFlagMessage.MESSAGE_TYPE).flag
            if not flag:
                break
            msg: GlobalWeightMessage = self.recv_msg(neighbor.get_exchange_name(), config.current_node_mq_url,
                                                     GlobalWeightMessage.MESSAGE_TYPE)
            smashed_layers = msg.weights[0]
            labels = msg.weights[1]
            inputs, targets = smashed_layers.to(self.device), labels.to(self.device)

            # Handle 2-tier split_point (int)
            split_point = self.split_layers[neighbor]
            if isinstance(split_point, list):
                split_point = split_point[0]

            if split_point < len(self.uninet.cfg) - 1:
                if neighbor in self.optimizers.keys():
                    self.optimizers[neighbor].zero_grad()
            outputs = self.nets[neighbor](inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            if split_point < len(self.uninet.cfg) - 1:
                if neighbor in self.optimizers:
                    self.optimizers[neighbor].step()
                    self.scheduler[neighbor].step()

            fed_logger.info(str(neighbor) + " sending gradients")
            msg = GlobalWeightMessage([inputs.grad])
            self.send_msg(self.get_exchange_name(), neighbor_rabbitmq_url, msg)

        fed_logger.info(str(neighbor) + ' offloading training end')

    def start_centralized_training(self):
        self.threads = {}
        client_neighbors = self.get_neighbors([NodeType.CLIENT])
        for neighbor in client_neighbors:
            self.threads[neighbor] = threading.Thread(target=self._thread_centralized_training,
                                                      args=(neighbor,), name=str(neighbor))
            fed_logger.info(str(neighbor) + ' offloading training start')
            self.threads[neighbor].start()

        fed_logger.info('waiting for offloading training to finish')
        for neighbor in client_neighbors:
            self.threads[neighbor].join()
        fed_logger.info('offloading training finished')

    def _thread_centralized_training(self, neighbor: NodeIdentifier):
        self._forward_propagation(neighbor)
        self._send_back_local_weight(neighbor)

    def _forward_propagation(self, neighbor: NodeIdentifier):
        msg: IterationFlagMessage = self.recv_msg(neighbor.get_exchange_name(), config.current_node_mq_url,
                                                  IterationFlagMessage.MESSAGE_TYPE)
        server_neighbor = self.get_neighbors([NodeType.SERVER])[0]
        edge_exchange = self.get_exchange_name(neighbor)
        flag: bool = msg.flag

        # Handle 2-tier split_point (int) - convert to list for 3-tier compatibility
        split_point = self.split_layers[neighbor]
        if not isinstance(split_point, list):
            split_layers_list = [split_point, split_point]  # Dummy for centralized
        else:
            split_layers_list = split_point

        if split_layers_list[1] < model_utils.get_unit_model_len() - 1:
            self.send_msg(edge_exchange, HTTPCommunicator.get_rabbitmq_url(server_neighbor),
                          IterationFlagMessage(flag))
        else:
            self.send_msg(edge_exchange, HTTPCommunicator.get_rabbitmq_url(server_neighbor),
                          IterationFlagMessage(False))

        while flag:
            if split_layers_list[0] < model_utils.get_unit_model_len() - 1:
                msg: IterationFlagMessage = self.recv_msg(neighbor.get_exchange_name(), config.current_node_mq_url,
                                                          IterationFlagMessage.MESSAGE_TYPE)
                flag: bool = msg.flag

                if not flag:
                    self.send_msg(edge_exchange, HTTPCommunicator.get_rabbitmq_url(server_neighbor),
                                  IterationFlagMessage(flag))
                    break

                msg: GlobalWeightMessage = self.recv_msg(neighbor.get_exchange_name(),
                                                         config.current_node_mq_url,
                                                         GlobalWeightMessage.MESSAGE_TYPE)
                smashed_layers = msg.weights[0]
                labels = msg.weights[1]

                inputs, targets = smashed_layers.to(self.device), labels.to(self.device)
                if split_layers_list[0] < split_layers_list[1]:
                    if neighbor in self.optimizers.keys():
                        self.optimizers[neighbor].zero_grad()
                    outputs = self.nets[neighbor](inputs)
                    if split_layers_list[1] < model_utils.get_unit_model_len() - 1:
                        self.send_msg(edge_exchange,
                                      HTTPCommunicator.get_rabbitmq_url(server_neighbor),
                                      IterationFlagMessage(flag))
                        msg: list = [outputs.to(self.device), targets.to(self.device)]
                        self.send_msg(edge_exchange,
                                      HTTPCommunicator.get_rabbitmq_url(server_neighbor),
                                      GlobalWeightMessage(msg))
                        msg: GlobalWeightMessage = self.recv_msg(edge_exchange,
                                                                 config.current_node_mq_url,
                                                                 GlobalWeightMessage.MESSAGE_TYPE)
                        gradients = msg.weights[0].to(self.device)
                        outputs.backward(gradients)
                        msg: list = [inputs.grad]
                        self.send_msg(self.get_exchange_name(), HTTPCommunicator.get_rabbitmq_url(neighbor),
                                      GlobalWeightMessage(msg))
                    else:
                        outputs = self.nets[neighbor](inputs)
                        loss = self.criterion(outputs, targets)
                        loss.backward()
                        if neighbor in self.optimizers.keys():
                            self.optimizers[neighbor].step()
                        msg: list = [inputs.grad]
                        self.send_msg(self.get_exchange_name(), HTTPCommunicator.get_rabbitmq_url(neighbor),
                                      GlobalWeightMessage(msg))
                else:
                    self.send_msg(edge_exchange, HTTPCommunicator.get_rabbitmq_url(server_neighbor),
                                  IterationFlagMessage(flag))
                    msg: list = [inputs.cpu(), targets.cpu()]
                    self.send_msg(edge_exchange, HTTPCommunicator.get_rabbitmq_url(server_neighbor),
                                  GlobalWeightMessage(msg))
                    msg: GlobalWeightMessage = self.recv_msg(neighbor.get_exchange_name(),
                                                             config.current_node_mq_url,
                                                             GlobalWeightMessage.MESSAGE_TYPE)
                    self.send_msg(self.get_exchange_name(), HTTPCommunicator.get_rabbitmq_url(neighbor), msg)
        fed_logger.info(str(neighbor) + ' offloading training end')

    def _send_back_local_weight(self, neighbor: NodeIdentifier):
        cweights = self.recv_msg(neighbor.get_exchange_name(), config.current_node_mq_url,
                                 GlobalWeightMessage.MESSAGE_TYPE).weights[0]
        server_neighbor = self.get_neighbors([NodeType.SERVER])[0]

        # Handle 2-tier split_point
        split_point = self.split_layers[neighbor]
        if isinstance(split_point, list):
            split_point = split_point[0]

        if split_point != (config.model_len - 1):
            w_local = model_utils.concat_weights(self.uninet.state_dict(), cweights,
                                                 self.nets[neighbor].state_dict())
        else:
            w_local = cweights
        msg = GlobalWeightMessage([w_local])
        self.send_msg(self.get_exchange_name(neighbor), HTTPCommunicator.get_rabbitmq_url(server_neighbor), msg)

    def gather_local_weights(self) -> dict[str, BaseModel]:
        client_local_weights = {}
        for neighbor in self.get_neighbors([NodeType.CLIENT]):
            msg: GlobalWeightMessage = self.recv_msg(neighbor.get_exchange_name(), config.current_node_mq_url,
                                                     GlobalWeightMessage.MESSAGE_TYPE)
            client_local_weights[neighbor] = msg.weights[0]
        return client_local_weights

    def aggregate(self, client_local_weights: dict[str, BaseModel]) -> None:
        zero_model = model_utils.zero_init(self.uninet).state_dict()
        w_local_list = self._concat_neighbor_local_weights(client_local_weights)
        aggregated_model = self.aggregator.aggregate(zero_model, w_local_list)
        self.uninet.load_state_dict(aggregated_model)

    def _concat_neighbor_local_weights(self, client_local_weights) -> list:
        w_local_list = []
        client_neighbors = self.get_neighbors([NodeType.CLIENT])
        for neighbor in client_neighbors:
            # Handle 2-tier split_point (int)
            split_point = self.split_layers[neighbor]
            if isinstance(split_point, list):
                split_point = split_point[0]

            w_local = (client_local_weights[neighbor], config.N / len(client_neighbors))
            if self.offload and split_point != (config.model_len - 1):
                w_local = (
                    model_utils.concat_weights(self.uninet.state_dict(), client_local_weights[str(neighbor)],
                                               self.nets[neighbor].state_dict()),
                    config.N / len(client_neighbors))
            w_local_list.append(w_local)
        return w_local_list

    def bandwidth(self) -> dict[NodeIdentifier, BandWidth]:
        return self.neighbor_bandwidth

    def gossip_with_neighbors(self):
        edge_neighbors = self.get_neighbors([NodeType.EDGE])
        if len(edge_neighbors) == 0:
            return
        msg = GlobalWeightMessage([self.uninet.to(self.device).state_dict()])
        self.scatter_msg(msg, [NodeType.EDGE])
        gathered_msgs = self.gather_msgs(GlobalWeightMessage.MESSAGE_TYPE, [NodeType.EDGE])
        gathered_models = [(msg.message.weights[0], config.N / len(edge_neighbors)) for msg in gathered_msgs]
        zero_model = model_utils.zero_init(self.uninet).state_dict()
        aggregated_model = self.aggregator.aggregate(zero_model, gathered_models)
        self.uninet.load_state_dict(aggregated_model)
