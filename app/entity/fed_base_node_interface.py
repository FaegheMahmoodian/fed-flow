"""
FedBaseNodeInterface - Base class for all federated learning nodes

Provides:
    - Message communication protocols
    - Bandwidth management (hardcoded/measured)
    - Split layer configuration
    - Architecture detection (2-tier vs 3-tier)
"""

import threading
import time
from abc import ABC
from typing import Dict, List

from app.config import config
from app.config.logger import fed_logger
from app.dto.bandwidth import BandWidth
from app.dto.message import (
    BaseMessage,
    GlobalWeightMessage,
    SplitLayerConfigMessage,
    MessageType,
    NetworkTestMessage
)
from app.dto.received_message import ReceivedMessage
from app.entity.communicator import Communicator
from app.entity.http_communicator import HTTPCommunicator
from app.entity.node import Node
from app.entity.node_identifier import NodeIdentifier
from app.entity.node_type import NodeType
from app.util import data_utils


class FedBaseNodeInterface(ABC, Node, Communicator):
    """Base interface for federated learning nodes with bandwidth management"""

    def __init__(
            self,
            ip: str,
            port: int,
            node_type: NodeType,
            cluster: str,
            neighbors: List[NodeIdentifier] = None
    ):
        Node.__init__(self, ip, port, node_type, cluster, neighbors)
        Communicator.__init__(self)

        # State tracking
        self.neighbor_bandwidth: Dict[NodeIdentifier, BandWidth] = {}
        self.uninet = None
        self.split_layers = None
        self.device = None
        self._edge_based = None

    # ================================================================
    # MESSAGE BROADCASTING
    # ================================================================

    def scatter_msg(
            self,
            msg: BaseMessage,
            neighbors_types: List[NodeType] = None
    ):
        """Send message to all neighbors of specified types"""
        for neighbor in self.get_neighbors(neighbors_types):
            rabbitmq_url = HTTPCommunicator.get_rabbitmq_url(neighbor)
            self.send_msg(self.get_exchange_name(), rabbitmq_url, msg)

    def scatter_global_weights(self, neighbors_types: List[NodeType] = None):
        """Broadcast global model weights to neighbors"""
        msg = GlobalWeightMessage([self.uninet.state_dict()])
        self.scatter_msg(msg, neighbors_types)

    def scatter_split_layers(self, neighbors_types: List[NodeType] = None):
        """Send split layer configuration to each neighbor"""
        for neighbor in self.get_neighbors(neighbors_types):
            msg = SplitLayerConfigMessage(self.split_layers[neighbor])
            rabbitmq_url = HTTPCommunicator.get_rabbitmq_url(neighbor)
            self.send_msg(self.get_exchange_name(), rabbitmq_url, msg)

    # ================================================================
    # MESSAGE GATHERING
    # ================================================================

    def gather_msgs(
            self,
            msg_type: MessageType,
            neighbors_types: List[NodeType] = None
    ) -> List[ReceivedMessage]:
        """Collect messages from all neighbors of specified types"""
        messages = []

        for neighbor in self.get_neighbors():
            neighbor_type = HTTPCommunicator.get_node_type(neighbor)

            if neighbors_types is None or neighbor_type in neighbors_types:
                msg = self.recv_msg(
                    neighbor.get_exchange_name(),
                    config.current_node_mq_url,
                    msg_type
                )
                messages.append(ReceivedMessage(msg, neighbor))

        return messages

    # ================================================================
    # BANDWIDTH MANAGEMENT
    # ================================================================

    def gather_neighbors_network_bandwidth(self, neighbors_type: NodeType = None):
        """
        Gather network bandwidth from neighbors.
        Uses hardcoded values if config.USE_HARDCODED_BW is True.
        """
        mode = "Hardcoded" if config.USE_HARDCODED_BW else "Measured"
        fed_logger.info(f"Gathering bandwidth ({mode} mode)")

        if config.USE_HARDCODED_BW:
            # Use hardcoded bandwidth values
            for neighbor in self.get_neighbors():
                neighbor_type_check = HTTPCommunicator.get_node_type(neighbor)

                if neighbors_type is None or neighbor_type_check == neighbors_type:
                    bandwidth = self._get_hardcoded_bandwidth(neighbor)
                    self.neighbor_bandwidth[neighbor] = bandwidth

                    fed_logger.info(
                        f"Hardcoded BW for {neighbor}: "
                        f"{bandwidth.to_mbps():.2f} Mbps"
                    )
        else:
            # Use threaded real measurement
            fed_logger.info("Measuring bandwidth via network testing")
            net_threads = {}

            for neighbor in self.get_neighbors():
                neighbor_type_check = HTTPCommunicator.get_node_type(neighbor)

                if neighbors_type is None or neighbor_type_check == neighbors_type:
                    net_threads[neighbor] = threading.Thread(
                        target=self._thread_network_testing,
                        args=(neighbor,),
                        name=str(neighbor)
                    )
                    net_threads[neighbor].start()

            # Wait for all measurements to complete
            for _, thread in net_threads.items():
                thread.join()

    def _get_hardcoded_bandwidth(self, neighbor: NodeIdentifier) -> BandWidth:
        """
        Get hardcoded bandwidth value for specific neighbor.
        Priority: CLIENT_BANDWIDTH_MAP > Node Type Default > Fallback
        """
        # Priority 1: Custom mapping for specific neighbor
        node_key = f"{neighbor.ip}_{neighbor.port}"

        if hasattr(config, 'CLIENT_BANDWIDTH_MAP') and node_key in config.CLIENT_BANDWIDTH_MAP:
            bw_value = config.CLIENT_BANDWIDTH_MAP[node_key]
            fed_logger.debug(
                f"Using custom hardcoded BW for {node_key}: {bw_value} Mbps"
            )
            return BandWidth(hardcoded_value=bw_value)

        # Priority 2: Node type defaults
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

    def _thread_network_testing(self, neighbor: NodeIdentifier):
        """
        Measure real bandwidth by sending test message and timing response.
        Falls back to hardcoded value on error.
        """
        try:
            # Start timing
            network_time_start = time.time()

            # Send test data
            msg = NetworkTestMessage([self.uninet.to(self.device).state_dict()])
            neighbor_rabbitmq_url = HTTPCommunicator.get_rabbitmq_url(neighbor)
            self.send_msg(self.get_exchange_name(), neighbor_rabbitmq_url, msg)

            # Wait for response
            response: NetworkTestMessage = self.recv_msg(
                neighbor.get_exchange_name(),
                config.current_node_mq_url,
                NetworkTestMessage.MESSAGE_TYPE
            )

            # End timing
            network_time_end = time.time()
            elapsed_time = network_time_end - network_time_start

            # Validate measurement
            if elapsed_time <= 0:
                fed_logger.warning(
                    f"Invalid elapsed time ({elapsed_time}s) for {neighbor}, "
                    f"using fallback"
                )
                self.neighbor_bandwidth[neighbor] = BandWidth(
                    hardcoded_value=config.HARDCODED_CLIENT_BW
                )
                return

            # Calculate bandwidth
            data_size = data_utils.sizeofmessage(response.weights)
            self.neighbor_bandwidth[neighbor] = BandWidth(
                transferred_bytes=data_size,
                time=elapsed_time
            )

            fed_logger.info(
                f"Measured BW for {neighbor}: "
                f"{self.neighbor_bandwidth[neighbor].to_mbps():.2f} Mbps"
            )

        except Exception as e:
            fed_logger.error(
                f"Bandwidth measurement failed for {neighbor}: {e}. "
                f"Using fallback: {config.HARDCODED_CLIENT_BW} Mbps"
            )
            self.neighbor_bandwidth[neighbor] = BandWidth(
                hardcoded_value=config.HARDCODED_CLIENT_BW
            )

    def get_neighbors_bandwidth(self) -> Dict[NodeIdentifier, BandWidth]:
        """Get current bandwidth measurements for all neighbors"""
        return self.neighbor_bandwidth

    # ================================================================
    # ARCHITECTURE DETECTION
    # ================================================================

    @property
    def is_edge_based(self) -> bool:
        """Check if system operates in edge-based (3-tier) architecture"""
        if self._edge_based is not None:
            return self._edge_based

        self._edge_based = False

        # Check if any edge has server neighbors (3-tier)
        for edge in self.get_neighbors([NodeType.EDGE]):
            server_neighbors = HTTPCommunicator.get_neighbors_from_neighbor(edge)
            if len(server_neighbors) > 0:
                self._edge_based = True
                break

        return self._edge_based

    # ================================================================
    # SPLIT LAYER INITIALIZATION
    # ================================================================

    def initialize_split_layers(self):
        """Initialize split layer configuration based on architecture type"""
        self.split_layers = {}

        if not self.is_edge_based:
            # 2-tier: Edge directly assigns split points to clients (int)
            assert self._node_type == NodeType.EDGE, "Only EDGE can init in 2-tier"

            default_split = len(self.uninet.cfg) - 1  # No offloading by default

            for neighbor in self.get_neighbors([NodeType.CLIENT]):
                self.split_layers[neighbor] = default_split

            fed_logger.info(
                f"Initialized 2-tier split layers with default: {default_split}"
            )
        else:
            # 3-tier: Server manages split points for all clients via edges (list)
            assert self._node_type == NodeType.SERVER, "Only SERVER can init in 3-tier"

            default_split = [
                len(self.uninet.cfg) - 1,
                len(self.uninet.cfg) - 1
            ]

            for edge in self.get_neighbors([NodeType.EDGE]):
                edge_clients = HTTPCommunicator.get_neighbors_from_neighbor(
                    edge,
                    [NodeType.CLIENT]
                )
                for client in edge_clients:
                    self.split_layers[client] = default_split

            fed_logger.info(
                f"Initialized 3-tier split layers with default: {default_split}"
            )