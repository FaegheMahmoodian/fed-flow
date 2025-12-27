import os
import sys
from os import environ

from app.entity.node_coordinate import NodeCoordinate
from app.entity.node_identifier import NodeIdentifier

# ================================================================
# DEBUG & LOGGING
# ================================================================
DEBUG = os.getenv('DEBUG', 'True') == 'True'  # Enable for local testing

# ================================================================
# DATASET CONFIGURATION
# ================================================================
dataset_name = 'CIFAR10'  # Lighter than CIFAR100
home = sys.path[0].split('fed-flow')[0] + 'fed-flow' + "/app"
dataset_path = home + '/dataset/data/'
N = 1000  # REDUCED: Total training samples (was 50000)
index = 0

# ================================================================
# MODEL CONFIGURATION (Reduced VGG5)
# ================================================================
model_name = 'VGG5'
model_cfg = {
    # Lighter VGG5 for local testing (reduced channels)
    'VGG5': [
        ('C', 3, 16, 3, 16 * 32 * 32, 16 * 32 * 32 * 3 * 3 * 3),      # 32→16 channels
        ('M', 16, 16, 2, 16 * 16 * 16, 0),
        ('C', 16, 32, 3, 32 * 16 * 16, 32 * 16 * 16 * 3 * 3 * 16),    # 64→32 channels
        ('M', 32, 32, 2, 32 * 8 * 8, 0),
        ('C', 32, 32, 3, 32 * 8 * 8, 32 * 8 * 8 * 3 * 3 * 32),        # 64→32 channels
        ('D', 8 * 8 * 32, 64, 1, 64, 64 * 8 * 8 * 32),                # 128→64 units
        ('D', 64, 10, 1, 10, 64 * 10)                                 # 10 classes (CIFAR10)
    ]
}
model_len = 7

# Split layer (3-tier setup)
split_layer = [[5, 5]]  # Split earlier to reduce computation on client

# ================================================================
# FEDERATED LEARNING PARAMETERS (Optimized for Speed)
# ================================================================
R = int(environ.get("ROUND_COUNT", "3"))  # REDUCED: 3 rounds (was 2, but test with more)
current_round = 0

learning_rate = 0.05  # INCREASED: Faster convergence for small dataset
B = 50  # REDUCED: Batch size (was 100) - less memory usage
lr_step_size = 2  # REDUCED: Decay LR every 2 rounds
lr_gamma = 0.5  # ADJUSTED: Slower decay

K = int(environ.get("DEVICE_COUNT", "2"))  # REDUCED: 2 clients max (was 1)
G = 1
S = 1

# ================================================================
# BANDWIDTH CONFIGURATION (HARDCODED MODE)
# ================================================================

# Enable hardcoded bandwidth (no real network measurement)
USE_HARDCODED_BW = os.getenv('USE_HARDCODED_BW', 'True') == 'True'  # DEFAULT: True

# Local network speeds (simulated LAN)
HARDCODED_CLIENT_BW = float(environ.get('HARDCODED_CLIENT_BW', '50.0'))   # 50 Mbps (local LAN)
HARDCODED_EDGE_BW = float(environ.get('HARDCODED_EDGE_BW', '100.0'))      # 100 Mbps (server)

# Custom bandwidth map (for heterogeneous testing)
CLIENT_BANDWIDTH_MAP = {
    # Example: Simulate slow client
    # "127.0.0.1_5001": 10.0,   # Client 1: 10 Mbps
    # "127.0.0.1_5002": 50.0,   # Client 2: 50 Mbps
}

# Bandwidth fluctuation (disable for consistent testing)
BW_FLUCTUATION_ENABLED = os.getenv('BW_FLUCTUATION_ENABLED', 'False') == 'True'
BW_FLUCTUATION_PERCENT = float(environ.get('BW_FLUCTUATION_PERCENT', '5.0'))  # ±5%

# Legacy (deprecated)
CLIENTS_BANDWIDTH = []

# ================================================================
# RABBITMQ CONFIGURATION (Local)
# ================================================================
mq_url = "amqp://rabbitmq:rabbitmq@localhost:5672/"
current_node_mq_url = "Will be set by input options"
cluster = "fed-flow-local"

# ================================================================
# NETWORK TOPOLOGY
# ================================================================
CURRENT_NODE_NEIGHBORS: list[NodeIdentifier] = []
INITIAL_NODE_COORDINATE = NodeCoordinate
SCENARIO_DESCRIPTION = environ.get("SCENARIO_DESCRIPTION", "Local Laptop Test - 1000 samples")

# ================================================================
# SIMULATION MODE
# ================================================================
simnet = False

# ================================================================
# HELPER FUNCTIONS
# ================================================================

def get_hardcoded_bw_for_node(node_id: NodeIdentifier) -> float:
    """Get hardcoded bandwidth for specific node"""
    node_key = f"{node_id.ip}_{node_id.port}"
    
    if node_key in CLIENT_BANDWIDTH_MAP:
        return CLIENT_BANDWIDTH_MAP[node_key]
    
    try:
        from app.entity.http_communicator import HTTPCommunicator
        from app.entity.node_type import NodeType
        
        node_type = HTTPCommunicator.get_node_type(node_id)
        return HARDCODED_EDGE_BW if node_type == NodeType.EDGE else HARDCODED_CLIENT_BW
    except:
        return HARDCODED_CLIENT_BW


def apply_bandwidth_fluctuation(bandwidth_mbps: float) -> float:
    """Apply random fluctuation to bandwidth"""
    if not BW_FLUCTUATION_ENABLED:
        return bandwidth_mbps
    
    import random
    fluctuation = random.uniform(
        -BW_FLUCTUATION_PERCENT / 100,
        BW_FLUCTUATION_PERCENT / 100
    )
    
    return bandwidth_mbps * (1 + fluctuation)


def log_bandwidth_config():
    """Log bandwidth configuration"""
    from app.config.logger import fed_logger
    
    fed_logger.info("=" * 50)
    fed_logger.info("BANDWIDTH CONFIGURATION")
    fed_logger.info("=" * 50)
    fed_logger.info(f"Mode: {'HARDCODED' if USE_HARDCODED_BW else 'MEASURED'}")
    
    if USE_HARDCODED_BW:
        fed_logger.info(f"Default Client BW: {HARDCODED_CLIENT_BW} Mbps")
        fed_logger.info(f"Default Edge BW: {HARDCODED_EDGE_BW} Mbps")
        fed_logger.info(f"Custom Mappings: {len(CLIENT_BANDWIDTH_MAP)}")
        fed_logger.info(f"Fluctuation: {'ON' if BW_FLUCTUATION_ENABLED else 'OFF'}")
        if BW_FLUCTUATION_ENABLED:
            fed_logger.info(f"Fluctuation Range: ±{BW_FLUCTUATION_PERCENT}%")
    
    fed_logger.info("=" * 50)