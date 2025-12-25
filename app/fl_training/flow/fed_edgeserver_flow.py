import time

from app.config import config
from app.config.logger import fed_logger
from app.entity.aggregators.factory import create_aggregator
from app.entity.fed_edge_server import FedEdgeServer
from app.entity.http_communicator import HTTPCommunicator
from app.entity.node_type import NodeType
from app.util import graph_utils, model_utils


def run_decentralized(edge_server: FedEdgeServer, learning_rate, options: dict):
    """
    Execute decentralized federated learning on edge server.

    Args:
        edge_server: Edge server instance
        learning_rate: Learning rate for training
        options: Configuration options including splitting and clustering methods
    """
    edge_server.initialize(learning_rate)
    fed_logger.info(f"Split Config : {edge_server.split_layers}")
    edge_server.scatter_split_layers([NodeType.CLIENT])

    # Metrics tracking
    client_bw, edge_bw = [], []
    training_times = []
    rounds = []
    accuracy = []

    for r in range(config.R):
        config.current_round = r
        rounds.append(r)
        fed_logger.info('====================================>')
        fed_logger.info('==> Round {:} Start'.format(r + 1))

        # Broadcast global model weights to all clients
        fed_logger.info("sending global weights")
        edge_server.scatter_global_weights([NodeType.CLIENT])

        s_time = time.time()

        # Gather network bandwidth information from neighbors
        fed_logger.info("gathering neighbors network speed")
        edge_server.gather_neighbors_network_bandwidth()

        # Perform clustering on clients (if enabled in options)
        fed_logger.info("clustering")
        edge_server.clustering(options)

        # Collect and organize bandwidth data by node type
        fed_logger.info("getting neighbors bandwidth")
        neighbors_bandwidth = edge_server.get_neighbors_bandwidth()
        neighbors_bandwidth_by_type: dict[NodeType, list[float]] = {}

        for neighbor, bw in neighbors_bandwidth.items():
            neighbor_type = HTTPCommunicator.get_node_type(neighbor)
            if neighbor_type not in neighbors_bandwidth_by_type:
                neighbors_bandwidth_by_type[neighbor_type] = []
            neighbors_bandwidth_by_type[neighbor_type].append(bw.bandwidth)

        # Calculate average bandwidth for clients
        client_bw.append(
            sum(neighbors_bandwidth_by_type[NodeType.CLIENT]) / len(neighbors_bandwidth_by_type[NodeType.CLIENT]))

        # Calculate average bandwidth for edge servers (if any)
        if NodeType.EDGE in neighbors_bandwidth_by_type:
            edge_bw.append(
                sum(neighbors_bandwidth_by_type[NodeType.EDGE]) / len(neighbors_bandwidth_by_type[NodeType.EDGE]))
        else:
            edge_bw.append(0)

        # Determine split points based on bandwidth and clustering
        # Pass client bandwidth list and options to splitting function
        fed_logger.info("splitting")
        edge_server.split(neighbors_bandwidth_by_type.get(NodeType.CLIENT, []), options)
        fed_logger.info(f"Split Config : {edge_server.split_layers}")

        # Broadcast updated split configuration to clients
        edge_server.scatter_split_layers([NodeType.CLIENT])

        # Execute local training on clients with current split configuration
        fed_logger.info("start training")
        edge_server.start_decentralized_training()

        # Collect trained local model weights from all clients
        fed_logger.info("receiving local weights")
        local_weights = edge_server.gather_local_weights()

        # Aggregate collected weights using configured aggregation method
        fed_logger.info("aggregating weights")
        edge_server.aggregate(local_weights)

        # Exchange aggregated model with neighboring edge servers (if any)
        fed_logger.info("start gossiping with neighbors")
        edge_server.gossip_with_neighbors()

        e_time = time.time()

        # Calculate and record round metrics
        training_time = e_time - s_time
        training_times.append(training_time)

        # Evaluate global model accuracy on test set
        fed_logger.info("testing accuracy")
        test_acc = model_utils.test(edge_server.uninet, edge_server.testloader, edge_server.device,
                                    edge_server.criterion)
        fed_logger.info(f"Test Accuracy : {test_acc}")
        accuracy.append(test_acc)

        fed_logger.info('Round Finish')
        fed_logger.info('==> Round {:} End'.format(r + 1))
        fed_logger.info('==> Round Training Time: {:}'.format(training_time))

    # Generate and save performance reports
    graph_utils.report_results(edge_server, training_times, client_bw, accuracy, edge_bw)


def run_centralized(edge_server: FedEdgeServer, learning_rate):
    """
    Execute centralized federated learning through central server.

    Args:
        edge_server: Edge server instance acting as intermediary
        learning_rate: Learning rate for training
    """
    # Initial setup: receive split configuration from central server
    edge_server.gather_and_scatter_split_config()
    edge_server.initialize(learning_rate)

    for r in range(config.R):
        config.current_round = r
        fed_logger.info('====================================>')
        fed_logger.info('==> Round {:} Start'.format(r + 1))

        # Synchronize split configuration with central server
        fed_logger.info("receiving and sending splitting info")
        edge_server.gather_and_scatter_split_config()

        # Synchronize global model weights with central server
        fed_logger.info("receiving and sending global weights")
        edge_server.gather_and_scatter_global_weight()

        # Monitor client network conditions
        fed_logger.info("test clients network")
        edge_server.gather_neighbors_network_bandwidth()

        # Execute centralized training (forward/backward through server)
        fed_logger.info("start training")
        edge_server.start_centralized_training()

        fed_logger.info('==> Round {:} End'.format(r + 1))


def run(options_ins):
    """
    Main entry point for edge server federated learning.

    Args:
        options_ins: Dictionary containing all configuration options:
            - ip: Edge server IP address
            - port: Edge server port
            - model: Model architecture name
            - dataset: Dataset name
            - offload: Whether to offload computation
            - decentralized: Training mode (True=decentralized, False=centralized)
            - aggregation: Aggregation method name
            - splitting: Splitting strategy name
            - clustering: Clustering method name (optional)
    """
    LR = config.learning_rate
    fed_logger.info('Preparing Sever.')

    # Extract configuration options
    offload = options_ins.get('offload')
    decentralized = options_ins.get('decentralized')
    aggregator = create_aggregator(options_ins.get('aggregation'))

    # Initialize edge server with specified configuration
    edge_server = FedEdgeServer(
        options_ins.get('ip'),
        options_ins.get('port'),
        options_ins.get('model'),
        options_ins.get('dataset'),
        offload,
        aggregator,
        config.CURRENT_NODE_NEIGHBORS
    )

    fed_logger.info("neighbors: " + str(config.CURRENT_NODE_NEIGHBORS))
    fed_logger.info("start mode: " + str(options_ins.values()))

    # Execute appropriate training mode
    if decentralized:
        run_decentralized(edge_server, LR, options_ins)
    else:
        run_centralized(edge_server, LR)

    # Graceful shutdown
    time.sleep(10)
    edge_server.stop_server()
