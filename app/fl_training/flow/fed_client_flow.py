"""
Client Training Flow - Federated Learning Execution

Supports:
    - Standard FL with adaptive offloading
    - Device-to-Device (D2D) gossip protocol
    - Bandwidth-aware splitting
    - Energy estimation and mobility simulation
"""

import logging
import time
import warnings

from app.config import config
from app.config.logger import fed_logger
from app.entity.aggregators.factory import create_aggregator
from app.entity.fed_client import FedClient
from app.entity.node_type import NodeType
from app.util import data_utils, energy_estimation, model_utils
from app.util.mobility_data_utils import start_mobility_simulation_thread

warnings.filterwarnings('ignore')
logging.getLogger("requests").setLevel(logging.WARNING)


def run_client(client: FedClient, learning_rate: float):
    """
    Execute standard federated learning client training flow.

    Args:
        client: Client instance for training
        learning_rate: Learning rate for local training
    """
    for r in range(config.R):
        config.current_round = r
        fed_logger.info('====================================>')
        fed_logger.info(f'ROUND: {r + 1} START')

        # Receive split configuration from edge server
        fed_logger.info("Receiving split configuration from edge")
        client.gather_split_config()

        # Download global model weights from edge server
        fed_logger.info("Downloading global model weights")
        client.gather_global_weights(NodeType.EDGE)

        # Measure/report network bandwidth to edge server (if not using hardcoded)
        if not config.USE_HARDCODED_BW:
            fed_logger.info("Measuring network bandwidth to edge")
            client.scatter_network_speed_to_edges()
        else:
            fed_logger.info("Using hardcoded bandwidth - skipping measurement")

        # Execute local training with current split configuration
        fed_logger.info("Starting local training with offloading support")
        client.start_offloading_train()

        # Upload locally trained model weights to edge server
        fed_logger.info("Uploading local model weights to edge")
        client.scatter_local_weights()

        fed_logger.info(f'ROUND: {r + 1} END')


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
        fed_logger.info(f'ROUND: {r + 1} START')

        # Receive global model from central server (not edge)
        fed_logger.info("Downloading global model from server")
        client.gather_global_weights(NodeType.SERVER)

        # Train full model locally without offloading
        fed_logger.info("Starting full local training (no offloading)")
        client.no_offloading_train()

        # Exchange models with peer clients via gossip protocol
        fed_logger.info("Exchanging models with peer clients")
        client.gossip_with_neighbors()

        # Upload aggregated model to central server (cluster leaders only)
        fed_logger.info("Uploading to server (if cluster leader)")
        client.scatter_random_local_weights()

        fed_logger.info(f'ROUND: {r + 1} END')


def run(options_ins: dict):
    """
    Main entry point for client federated learning.

    Args:
        options_ins: Configuration dictionary containing:
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
    fed_logger.info(f"Starting client with config: {list(options_ins.values())}")

    # Get client index and learning rate from global config
    index = config.index
    learning_rate = config.learning_rate

    fed_logger.info('Preparing Client')
    fed_logger.info('Preparing Data Distribution')

    # Partition dataset for this client based on index
    N = config.N  # Total training samples
    K = config.K  # Total number of clients

    indices = list(range(N))
    part_tr = indices[int((N / K) * index): int((N / K) * (index + 1))]

    train_loader = data_utils.get_trainloader(
        data_utils.get_trainset(),
        part_tr,
        0  # Worker ID (0 for main process)
    )

    fed_logger.info(
        f"Client {index}: Loaded {len(part_tr)} samples "
        f"(indices {part_tr[0]}-{part_tr[-1]})"
    )

    # Extract configuration options
    estimate_energy = options_ins.get("energy") == "True"
    mobility = options_ins.get('mobility')
    d2d = options_ins.get('d2d')

    # Initialize energy estimation if enabled
    if estimate_energy:
        import os
        energy_estimation.init(os.getpid())
        fed_logger.info("Energy estimation enabled")

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

    fed_logger.info(f"Client initialized with neighbors: {config.CURRENT_NODE_NEIGHBORS}")
    fed_logger.info(f"Hardcoded BW mode: {config.USE_HARDCODED_BW}")

    # Start mobility simulation thread if enabled
    if mobility:
        fed_logger.info("Starting mobility simulation")
        start_mobility_simulation_thread(client)
        # Manual mobility control (uncomment if needed):
        # client.mobility_manager.discover_edges()
        # client.mobility_manager.monitor_and_migrate()

    # Execute appropriate training mode
    if d2d:
        fed_logger.info("Running in D2D mode")
        run_d2d(client)
    else:
        fed_logger.info("Running in standard FL mode")
        run_client(client, learning_rate)

    # Graceful shutdown after training completion
    fed_logger.info("Training completed - shutting down client")
    time.sleep(10)
    client.stop_server()
    fed_logger.info("Client stopped successfully")