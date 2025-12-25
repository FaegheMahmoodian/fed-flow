import threading
import time
from abc import ABC

from app.config import config
from app.dto.bandwidth import BandWidth
from app.dto.message import BaseMessage, GlobalWeightMessage, SplitLayerConfigMessage, MessageType, NetworkTestMessage
from app.dto.received_message import ReceivedMessage
from app.entity.communicator import Communicator
from app.entity.http_communicator import HTTPCommunicator
from app.entity.node import Node
from app.entity.node_identifier import NodeIdentifier
from app.entity.node_type import NodeType
from app.util import data_utils
from app.config.logger import fed_logger


# noinspection PyTypeChecker
class FedBaseNodeInterface(ABC, Node, Communicator):
    """Base interface for federated learning nodes with bandwidth management"""

    def __init__(self, ip: str, port: int, node_type: NodeType, cluster, neighbors: list[NodeIdentifier] = None):
        Node.__init__(self, ip, port, node_type, cluster, neighbors)
        Communicator.__init__(self)
        self.neighbor_bandwidth: dict[NodeIdentifier, BandWidth] = {}
        self.uninet = None
        self.split_layers = None
        self.device = None
        self._edge_based = None

    def scatter_msg(self, msg: BaseMessage, neighbors_types: list[NodeType] = None):
        """Send message to all neighbors of specified types"""
        for neighbor in self.get_neighbors(neighbors_types):
            rabbitmq_url = HTTPCommunicator.get_rabbitmq_url(neighbor)
            self.send_msg(self.get_exchange_name(), rabbitmq_url, msg)

    def scatter_global_weights(self, neighbors_types: list[NodeType] = None):
        """Broadcast global model weights to neighbors"""
        msg = GlobalWeightMessage([self.uninet.state_dict()])
        self.scatter_msg(msg, neighbors_types)

    def scatter_split_layers(self, neighbors_types: list[NodeType] = None):
        """Send split layer configuration to each neighbor"""
        for neighbor in self.get_neighbors(neighbors_types):
            msg = SplitLayerConfigMessage(self.split_layers[neighbor])
            self.send_msg(self.get_exchange_name(), HTTPCommunicator.get_rabbitmq_url(neighbor), msg)

    def gather_msgs(self, msg_type: MessageType, neighbors_types: list[NodeType] = None) -> list[ReceivedMessage]:
        """Collect messages from all neighbors of specified types"""
        messages = []
        for neighbor in self.get_neighbors():
            neighbor_type = HTTPCommunicator.get_node_type(neighbor)
            if neighbors_types is None or neighbor_type in neighbors_types:
                msg = self.recv_msg(neighbor.get_exchange_name(), config.current_node_mq_url, msg_type)
                messages.append(ReceivedMessage(msg, neighbor))
        return messages

    def gather_neighbors_network_bandwidth(self, neighbors_type: NodeType = None):
        """Gather network bandwidth from neighbors (real measurement or hardcoded)"""

        # Check if using hardcoded bandwidth mode
        if config.USE_HARDCODED_BW:
            fed_logger.info("Using hardcoded bandwidth values")
            for neighbor in self.get_neighbors():
                neighbor_type = HTTPCommunicator.get_node_type(neighbor)
                if neighbors_type is None or neighbor_type == neighbors_type:
                    self.neighbor_bandwidth[neighbor] = self._get_hardcoded_bandwidth(neighbor)
                    fed_logger.info(
                        f"Hardcoded BW for {neighbor}: {self.neighbor_bandwidth[neighbor].bandwidth:.2f} Mbps")
        else:
            # Use threaded real measurement
            fed_logger.info("Measuring real bandwidth via network testing")
            net_threads = {}
            for neighbor in self.get_neighbors():
                neighbor_type = HTTPCommunicator.get_node_type(neighbor)
                if neighbors_type is None or neighbor_type == neighbors_type:
                    net_threads[neighbor] = threading.Thread(
                        target=self._thread_network_testing,
                        args=(neighbor,),
                        name=str(neighbor)
                    )
                    net_threads[neighbor].start()

            for _, thread in net_threads.items():
                thread.join()

    def _get_hardcoded_bandwidth(self, neighbor: NodeIdentifier) -> BandWidth:
        """Get hardcoded bandwidth value for specific neighbor with priority fallback"""
        neighbor_type = HTTPCommunicator.get_node_type(neighbor)

        # Priority 1: Custom mapping for specific neighbor
        if hasattr(config, 'HARDCODED_BW_MAP') and neighbor in config.HARDCODED_BW_MAP:
            return config.HARDCODED_BW_MAP[neighbor]

        # Priority 2: Node type defaults
        if neighbor_type == NodeType.CLIENT:
            return config.DEFAULT_CLIENT_BW
        elif neighbor_type == NodeType.EDGE:
            return config.DEFAULT_EDGE_BW

        # Priority 3: Final fallback
        fed_logger.warning(f"Using fallback bandwidth for {neighbor}")
        return config.DEFAULT_CLIENT_BW

    def _thread_network_testing(self, neighbor: NodeIdentifier):
        """Measure real bandwidth by sending test message and timing response"""
        network_time_start = time.time()
        msg = NetworkTestMessage([self.uninet.to(self.device).state_dict()])
        neighbor_rabbitmq_url = HTTPCommunicator.get_rabbitmq_url(neighbor)
        self.send_msg(self.get_exchange_name(), neighbor_rabbitmq_url, msg)
        msg: NetworkTestMessage = self.recv_msg(neighbor.get_exchange_name(), config.current_node_mq_url,
                                                NetworkTestMessage.MESSAGE_TYPE)
        network_time_end = time.time()
        self.neighbor_bandwidth[neighbor] = BandWidth(data_utils.sizeofmessage(msg.weights),
                                                      network_time_end - network_time_start)
        fed_logger.info(f"Measured BW for {neighbor}: {self.neighbor_bandwidth[neighbor].bandwidth:.2f} Mbps")

    @property
    def is_edge_based(self) -> bool:
        """Check if system operates in edge-based (3-tier) architecture"""
        if self._edge_based is not None:
            return self._edge_based
        self._edge_based = False
        for edge in self.get_neighbors([NodeType.EDGE]):
            server_neighbors = HTTPCommunicator.get_neighbors_from_neighbor(edge)
            if len(server_neighbors) > 0:
                self._edge_based = True
                break
        return self._edge_based

    def initialize_split_layers(self):
        """Initialize split layer configuration based on architecture type"""
        self.split_layers = {}
        if not self.is_edge_based:
            # 2-tier: Edge directly assigns split points to clients
            assert self._node_type == NodeType.EDGE
            for neighbor in self.get_neighbors([NodeType.CLIENT]):
                self.split_layers[neighbor] = len(self.uninet.cfg) - 1
        else:
            # 3-tier: Server manages split points for all clients via edges
            assert self._node_type == NodeType.SERVER
            for edge in self.get_neighbors([NodeType.EDGE]):
                for client in HTTPCommunicator.get_neighbors_from_neighbor(edge, [NodeType.CLIENT]):
                    self.split_layers[client] = [len(self.uninet.cfg) - 1, len(self.uninet.cfg) - 1]
