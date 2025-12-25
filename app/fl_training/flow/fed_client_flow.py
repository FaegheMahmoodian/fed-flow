import logging
import time
import warnings

from app.config import config
from app.config.config import *
from app.config.logger import fed_logger
from app.entity.aggregators.factory import create_aggregator
from app.entity.fed_client import FedClient
from app.entity.node_type import NodeType
from app.util import data_utils, energy_estimation, model_utils
from app.util.mobility_data_utils import start_mobility_simulation_thread

warnings.filterwarnings('ignore')
logging.getLogger("requests").setLevel(logging.WARNING)


def run_client(client: FedClient, learning_rate):
    """
    Execute standard federated learning client training flow.

    Args:
        client: Client instance for training
        learning_rate: Learning rate for local training
    """
    for r in range(config.R):
        config.current_round = r
        fed_logger.info('====================================>')
        fed_logger.info('ROUND: {} START'.format(r + 1))

        # Receive split configuration from edge server
        # Split config determines which layers run locally vs. on edge
        fed_logger.info("receiving splitting info")
        client.gather_split_config()

        # Download global model weights from edge server
        fed_logger.info("receiving global weights")
        client.gather_global_weights(NodeType.EDGE)

        # Measure network bandwidth to edge server
        # Bandwidth info is used by edge for adaptive splitting decisions
        fed_logger.info("test network")
        client.scatter_network_speed_to_edges()

        # Execute local training with current split configuration
        # May offload computation to edge based on split point
        fed_logger.info("start training")
        client.start_offloading_train()

        # Upload locally trained model weights to edge server
        fed_logger.info("sending local weights")
        client.scatter_local_weights()

        fed_logger.info('ROUND: {} END'.format(r + 1))


def run_d2d(client: FedClient):
    """
    Execute device-to-device (D2D) federated learning without edge offloading.
    Uses gossip protocol for peer-to-peer model exchange.

    Args:
        client: Client instance for D2D training
    """
    for r in range(config.R):
        config.current_round = r
        fed_logger.info('====================================>')
        fed_logger.info('ROUND: {} START'.format(r + 1))

        # Receive global model from central server (not edge)
        fed_logger.info("receiving global weights")
        client.gather_global_weights(NodeType.SERVER)

        # Train full model locally without offloading
        fed_logger.info("start training")
        client.no_offloading_train()

        # Exchange models with peer clients via gossip protocol
        fed_logger.info("gossip with neighbors")
        client.gossip_with_neighbors()

        # Upload aggregated model to central server
        # Only cluster leaders upload to reduce communication
        fed_logger.info("sending local weights")
        client.scatter_random_local_weights()

        fed_logger.info('ROUND: {} END'.format(r + 1))


def run(options_ins):
    """
    Main entry point for client federated learning.

    Args:
        options_ins: Dictionary containing all configuration options:
            - ip: Client IP address
            - port: Client port
            - model: Model architecture name
            - dataset: Dataset name
            - cluster: Cluster identifier for client grouping
            - aggregation: Aggregation method name
            - energy: Enable energy consumption estimation ("True"/"False")
            - mobility: Enable client mobility simulation
            - d2d: Enable device-to-device mode (True=D2D, False=standard)
    """
    fed_logger.info("start mode: " + str(options_ins.values()))

    # Get client index and learning rate from global config
    index = config.index
    learning_rate = config.learning_rate

    fed_logger.info('Preparing Client')
    fed_logger.info('Preparing Data.')

    # Partition dataset for this client based on index
    # Each client gets (N/K) samples where N=total samples, K=num clients
    indices = list(range(N))
    part_tr = indices[int((N / K) * index): int((N / K) * (index + 1))]
    train_loader = data_utils.get_trainloader(data_utils.get_trainset(), part_tr, 0)

    # Extract configuration options
    estimate_energy = options_ins.get("energy") == "True"
    mobility = options_ins.get('mobility')
    d2d = options_ins.get('d2d')

    # Initialize energy estimation if enabled
    if estimate_energy:
        energy_estimation.init(os.getpid())

    # Extract network and training configuration
    ip = options_ins.get('ip')
    port = options_ins.get('port')
    cluster = options_ins.get('cluster')
    aggregator = create_aggregator(options_ins.get('aggregation'))

    # Initialize client with specified configuration
    client = FedClient(
        ip=ip,
        port=port,
        model_name=options_ins.get('model'),
        dataset=options_ins.get('dataset'),
        train_loader=train_loader,
        LR=learning_rate,
        cluster=cluster,
        aggregator=aggregator,
        neighbors=config.CURRENT_NODE_NEIGHBORS
    )

    # Start mobility simulation thread if enabled
    # Handles dynamic client-edge association and handover
    if mobility:
        start_mobility_simulation_thread(client)
        # Uncomment for manual mobility control:
        # client.mobility_manager.discover_edges()
        # client.mobility_manager.monitor_and_migrate()

    # Execute appropriate training mode
    if d2d:
        run_d2d(client)
    else:
        run_client(client, learning_rate)

    # Graceful shutdown after training completion
    time.sleep(10)
    client.stop_server()
