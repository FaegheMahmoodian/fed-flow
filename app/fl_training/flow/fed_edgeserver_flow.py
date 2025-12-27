"""
Edge Server Training Flow - Semi-Decentralized FL

Supports:
    - Decentralized mode (edge-client direct communication)
    - Centralized mode (cloud-based coordination)
    - Hardcoded bandwidth integration
    - Dynamic splitting and clustering
"""

import time
from typing import Dict, List

from app.config import config
from app.config.logger import fed_logger
from app.entity.aggregators.factory import create_aggregator
from app.entity.fed_edge_server import FedEdgeServer
from app.entity.http_communicator import HTTPCommunicator
from app.entity.node_type import NodeType
from app.util import graph_utils, model_utils


def run_decentralized(edge_server: FedEdgeServer, learning_rate: float, options: dict):
    """
    Run decentralized federated learning with edge-client collaboration.

    Args:
        edge_server: FedEdgeServer instance
        learning_rate: Initial learning rate
        options: Training configuration dict
    """
    # Initialize model and split configuration
    edge_server.initialize(learning_rate)
    fed_logger.info(f"Initial Split Config: {edge_server.split_layers}")
    edge_server.scatter_split_layers([NodeType.CLIENT])

    # Metrics tracking
    client_bw, edge_bw = [], []
    training_times = []
    rounds = []
    accuracy = []

    # Training loop
    for r in range(config.R):
        config.current_round = r
        rounds.append(r)
        fed_logger.info('====================================>')
        fed_logger.info(f'==> Round {r + 1} Start')

        # Distribute global model
        fed_logger.info("Sending global weights to clients")
        edge_server.scatter_global_weights([NodeType.CLIENT])

        s_time = time.time()

        # Gather bandwidth measurements (unless using hardcoded)
        if not config.USE_HARDCODED_BW:
            fed_logger.info("Gathering neighbors network bandwidth")
            edge_server.gather_neighbors_network_bandwidth()
        else:
            fed_logger.info("Using hardcoded bandwidth values")

        # Clustering (if enabled)
        fed_logger.info("Performing clustering")
        edge_server.clustering(options)

        # Extract bandwidth values for splitting
        client_bandwidths = []
        if config.USE_HARDCODED_BW:
            # Use hardcoded bandwidths
            clients = edge_server.get_neighbors([NodeType.CLIENT])
            for client in clients:
                bw = edge_server._get_hardcoded_bandwidth(client)
                client_bandwidths.append(bw.to_mbps())
        else:
            # Use measured bandwidths
            neighbors_bandwidth = edge_server.get_neighbors_bandwidth()
            for neighbor, bw in neighbors_bandwidth.items():
                neighbor_type = HTTPCommunicator.get_node_type(neighbor)
                if neighbor_type == NodeType.CLIENT:
                    client_bandwidths.append(bw.to_mbps())

        # Record average bandwidths for metrics
        if client_bandwidths:
            avg_client_bw = sum(client_bandwidths) / len(client_bandwidths)
            client_bw.append(avg_client_bw)
        else:
            client_bw.append(0.0)

        # Calculate edge bandwidth (if applicable)
        if not config.USE_HARDCODED_BW:
            neighbors_bandwidth = edge_server.get_neighbors_bandwidth()
            edge_bandwidths = []
            for neighbor, bw in neighbors_bandwidth.items():
                neighbor_type = HTTPCommunicator.get_node_type(neighbor)
                if neighbor_type == NodeType.EDGE:
                    edge_bandwidths.append(bw.to_mbps())

            if edge_bandwidths:
                edge_bw.append(sum(edge_bandwidths) / len(edge_bandwidths))
            else:
                edge_bw.append(0.0)
        else:
            edge_bw.append(0.0)

        # Perform splitting based on bandwidth
        fed_logger.info("Calculating optimal split points")
        edge_server.split(client_bandwidths, options)
        fed_logger.info(f"Updated Split Config: {edge_server.split_layers}")
        edge_server.scatter_split_layers([NodeType.CLIENT])

        # Start training round
        fed_logger.info("Starting decentralized training")
        edge_server.start_decentralized_training()

        # Gather local updates
        fed_logger.info("Receiving local weights from clients")
        local_weights = edge_server.gather_local_weights()

        # Aggregate updates
        fed_logger.info("Aggregating local weights")
        edge_server.aggregate(local_weights)

        # Gossip with neighbor edges (if any)
        fed_logger.info("Gossiping with neighbor edges")
        edge_server.gossip_with_neighbors()

        e_time = time.time()
        training_time = e_time - s_time
        training_times.append(training_time)

        # Evaluate global model
        fed_logger.info("Testing model accuracy")
        test_acc = model_utils.test(
            edge_server.uninet,
            edge_server.testloader,
            edge_server.device,
            edge_server.criterion
        )
        fed_logger.info(f"Test Accuracy: {test_acc:.4f}")
        accuracy.append(test_acc)

        fed_logger.info(f'==> Round {r + 1} End')
        fed_logger.info(f'==> Round Training Time: {training_time:.2f}s')

    # Report final results
    graph_utils.report_results(edge_server, training_times, client_bw, accuracy, edge_bw)


def run_centralized(edge_server: FedEdgeServer, learning_rate: float):
    """
    Run centralized federated learning (legacy mode).

    Args:
        edge_server: FedEdgeServer instance
        learning_rate: Initial learning rate
    """
    edge_server.gather_and_scatter_split_config()
    edge_server.initialize(learning_rate)

    for r in range(config.R):
        config.current_round = r
        fed_logger.info('====================================>')
        fed_logger.info(f'==> Round {r + 1} Start')

        # Exchange split configuration
        fed_logger.info("Exchanging split configuration")
        edge_server.gather_and_scatter_split_config()

        # Exchange global weights
        fed_logger.info("Exchanging global weights")
        edge_server.gather_and_scatter_global_weight()

        # Test network conditions
        fed_logger.info("Testing client network")
        edge_server.gather_neighbors_network_bandwidth()

        # Start training
        fed_logger.info("Starting centralized training")
        edge_server.start_centralized_training()

        fed_logger.info(f'==> Round {r + 1} End')


def run(options_ins: dict):
    """
    Main entry point for edge server training.

    Args:
        options_ins: Configuration dictionary containing:
            - ip: Edge server IP
            - port: Edge server port
            - model: Model architecture name
            - dataset: Dataset name
            - offload: Enable offloading
            - decentralized: Use decentralized mode
            - aggregation: Aggregation method
    """
    LR = config.learning_rate
    fed_logger.info('Preparing Edge Server')

    # Extract configuration
    offload = options_ins.get('offload')
    decentralized = options_ins.get('decentralized')
    aggregator = create_aggregator(options_ins.get('aggregation'))

    # Initialize edge server
    edge_server = FedEdgeServer(
        options_ins.get('ip'),
        options_ins.get('port'),
        options_ins.get('model'),
        options_ins.get('dataset'),
        offload,
        aggregator,
        config.CURRENT_NODE_NEIGHBORS
    )

    fed_logger.info(f"Neighbors: {config.CURRENT_NODE_NEIGHBORS}")
    fed_logger.info(f"Configuration: {list(options_ins.values())}")
    fed_logger.info(f"Hardcoded BW Mode: {config.USE_HARDCODED_BW}")

    # Run appropriate mode
    if decentralized:
        run_decentralized(edge_server, LR, options_ins)
    else:
        run_centralized(edge_server, LR)

    # Cleanup
    time.sleep(10)
    edge_server.stop_server()
    fed_logger.info("Edge server stopped successfully")
