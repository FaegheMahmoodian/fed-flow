import time

from app.config import config
from app.config.logger import fed_logger
from app.entity.aggregators.factory import create_aggregator
from app.entity.fed_edge_server import FedEdgeServer
from app.entity.http_communicator import HTTPCommunicator
from app.entity.node_type import NodeType
from app.util import graph_utils, model_utils

def run_decentralized(edge_server: FedEdgeServer, learning_rate, options: dict):
    edge_server.initialize(learning_rate)

    # Debug: ensure split_layers is initialized before the first scatter in decentralized mode
    if not edge_server.split_layers:
        # Debug: use a valid default state instead of {} to avoid splitting-method crashes
        default_state = [1.0] * len(edge_server.get_neighbors([NodeType.CLIENT]))
        edge_server.split(default_state, options)
        edge_server.initialize(learning_rate)  # Debug: rebuild nets/optimizers after setting initial split

    fed_logger.info(f"Split Config : {edge_server.split_layers}")
    edge_server.scatter_split_layers([NodeType.CLIENT])

    client_bw, edge_bw = [], []
    training_times = []
    rounds = []
    accuracy = []

    for r in range(config.R):
        config.current_round = r
        rounds.append(r)
        fed_logger.info('====================================>')
        fed_logger.info('==> Round {:} Start'.format(r + 1))

        fed_logger.info("sending global weights")
        edge_server.scatter_global_weights([NodeType.CLIENT])

        s_time = time.time()

        fed_logger.info("gathering neighbors network speed")
        edge_server.gather_neighbors_network_bandwidth()

        fed_logger.info("clustering")
        edge_server.clustering(options)

        fed_logger.info("getting neighbors bandwidth")
        neighbors_bandwidth = edge_server.get_neighbors_bandwidth()
        neighbors_bandwidth_by_type: dict[NodeType, list[float]] = {}
        for neighbor, bw in neighbors_bandwidth.items():
            neighbor_type = HTTPCommunicator.get_node_type(neighbor)
            if neighbor_type not in neighbors_bandwidth_by_type:
                neighbors_bandwidth_by_type[neighbor_type] = []
            neighbors_bandwidth_by_type[neighbor_type].append(bw.bandwidth)

        # Debug: guard against missing client bandwidth (keep arrays aligned; don't 'continue' silently)
        if NodeType.CLIENT not in neighbors_bandwidth_by_type or len(neighbors_bandwidth_by_type[NodeType.CLIENT]) == 0:
            fed_logger.warning("No client neighbors bandwidth available for splitting in this round")

            # Debug: keep result arrays aligned when skipping training for this round
            client_bw.append(0)
            edge_bw.append(0)
            training_times.append(0)
            accuracy.append(0)

            fed_logger.info('Round Finish (skipped)')
            fed_logger.info('==> Round {:} End'.format(r + 1))
            fed_logger.info('==> Round Training Time: 0')
            continue

        client_bw.append(
            sum(neighbors_bandwidth_by_type[NodeType.CLIENT]) / len(neighbors_bandwidth_by_type[NodeType.CLIENT])
        )
        if NodeType.EDGE in neighbors_bandwidth_by_type and len(neighbors_bandwidth_by_type[NodeType.EDGE]) > 0:
            edge_bw.append(
                sum(neighbors_bandwidth_by_type[NodeType.EDGE]) / len(neighbors_bandwidth_by_type[NodeType.EDGE])
            )
        else:
            edge_bw.append(0)

        fed_logger.info("splitting")

        # Debug: avoid unnecessary re-initialize when split config didn't change
        prev_split_layers = edge_server.split_layers
        edge_server.split(neighbors_bandwidth_by_type[NodeType.CLIENT], options)

        # Debug: rebuild per-client edge nets/optimizers only if the split actually changed
        if edge_server.split_layers != prev_split_layers:
            edge_server.initialize(learning_rate)

        fed_logger.info(f"Split Config : {edge_server.split_layers}")
        edge_server.scatter_split_layers([NodeType.CLIENT])

        fed_logger.info("start training")
        edge_server.start_decentralized_training()

        fed_logger.info("receiving local weights")
        local_weights = edge_server.gather_local_weights()

        fed_logger.info("aggregating weights")
        edge_server.aggregate(local_weights)

        fed_logger.info("start gossiping with neighbors")
        edge_server.gossip_with_neighbors()

        e_time = time.time()

        training_time = e_time - s_time
        training_times.append(training_time)

        fed_logger.info("testing accuracy")
        test_acc = model_utils.test(edge_server.uninet, edge_server.testloader, edge_server.device,
                                    edge_server.criterion)
        fed_logger.info(f"Test Accuracy : {test_acc}")
        accuracy.append(test_acc)
        fed_logger.info('Round Finish')
        fed_logger.info('==> Round {:} End'.format(r + 1))
        fed_logger.info('==> Round Training Time: {:}'.format(training_time))

    graph_utils.report_results(edge_server, training_times, client_bw, accuracy, edge_bw)


def run_centralized(edge_server: FedEdgeServer, learning_rate):
    edge_server.gather_and_scatter_split_config()
    edge_server.initialize(learning_rate)
    for r in range(config.R):
        config.current_round = r
        fed_logger.info('====================================>')
        fed_logger.info('==> Round {:} Start'.format(r + 1))
        fed_logger.info("receiving and sending splitting info")
        edge_server.gather_and_scatter_split_config()
        fed_logger.info("receiving and sending global weights")
        edge_server.gather_and_scatter_global_weight()
        fed_logger.info("test clients network")
        edge_server.gather_neighbors_network_bandwidth()
        fed_logger.info("start training")
        edge_server.start_centralized_training()
        fed_logger.info('==> Round {:} End'.format(r + 1))


def run(options_ins):
    LR = config.learning_rate
    fed_logger.info('Preparing Sever.')
    offload = options_ins.get('offload')
    decentralized = options_ins.get('decentralized')
    aggregator = create_aggregator(options_ins.get('aggregation'))
    edge_server = FedEdgeServer(options_ins.get('ip'), options_ins.get('port'),
                                options_ins.get('model'),
                                options_ins.get('dataset'), offload, aggregator,
                                config.CURRENT_NODE_NEIGHBORS)
    fed_logger.info("neighbors: " + str(config.CURRENT_NODE_NEIGHBORS))

    fed_logger.info("start mode: " + str(options_ins.values()))
    if decentralized:
        run_decentralized(edge_server, LR, options_ins)
    else:
        run_centralized(edge_server, LR)
    time.sleep(10)
    edge_server.stop_server()
