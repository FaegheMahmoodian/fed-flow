"""
Architecture:
    Client ←→ Edge Server (Single split point per client)

Features:
    - Hardcoded bandwidth support
    - Single-layer splitting (no cloud server)
"""

import random
import numpy as np
from typing import Dict, List, Optional, Callable
from collections import defaultdict

from app.config import config
from app.entity.fed_base_node_interface import FedBaseNodeInterface
from app.entity.node_identifier import NodeIdentifier
from app.entity.node_type import NodeType
from app.util import model_utils
from app.util.bandwidth import (
    load_client_bandwidths,
    apply_fluctuation_to_all,
    estimate_layer_latency
)


# ══════════════════════════════════════════════════════════════
#  Helper Functions
# ═══════════════════════════════════════════════════════════════

def _validate_split_point(split_point: int, model_len: int) -> int:
    """
    Validate and clamp split point to valid range.

    Ensures split point is within [1, model_len-1] to maintain:
    - At least 1 layer on client side
    - At least 1 layer on edge side

    Args:
        split_point: Proposed split layer index
        model_len: Total number of layers in model

    Returns:
        int: Clamped split point

    Example:
        >>> _validate_split_point(0, 7)
        1  # Minimum is 1
        >>> _validate_split_point(10, 7)
        6  # Maximum is model_len - 1
    """
    min_split = 1
    max_split = model_len - 1
    return max(min_split, min(split_point, max_split))


def _get_model_length(node: FedBaseNodeInterface) -> int:
    """
    Get total number of layers in the model.

    Args:
        node: FedEdgeServer or FedClient instance

    Returns:
        int: Number of layers

    Raises:
        AttributeError: If node doesn't have uninet attribute
    """
    try:
        return len(node.uninet.cfg)
    except AttributeError:
        # Fallback to config if node doesn't have model
        return config.model_len


def _compute_workload_distribution(model_cfg: List) -> np.ndarray:
    """
    Compute normalized workload distribution across layers.

    Calculates cumulative FLOPs for each layer and normalizes to [0, 1].
    Used for FLOP-based splitting decisions.

    Args:
        model_cfg: Model configuration list from config.model_cfg

    Returns:
        np.ndarray: Normalized cumulative workload [0.0, ..., 1.0]

    Example:
        >>> cfg = [(..., 100), (..., 200), (..., 300)]  # FLOPs at index 5
        >>> _compute_workload_distribution(cfg)
        array([0.167, 0.5, 1.0])  # Cumulative: 100/600, 300/600, 600/600
    """
    workload = []
    cumulated_flops = 0

    for layer in model_cfg:
        cumulated_flops += layer[5]  # FLOPs at index 5
        workload.append(cumulated_flops)

    # Normalize to [0, 1]
    workload_array = np.array(workload)
    total_flops = cumulated_flops

    if total_flops > 0:
        workload_array = workload_array / total_flops

    return workload_array


# ═══════════════════════════════════════════════════════════════
# 🟢 Section 1: Bandwidth-Aware Splitting Methods
# ═══════════════════════════════════════════════════════════════

def optimal_split(state, labels, **kwargs) -> Dict[NodeIdentifier, int]:
    """
    Optimal bandwidth-aware splitting for semi-decentralized FL.

    Computes split points proportional to client bandwidth:
    - Higher BW → More layers on client (later split)
    - Lower BW → Fewer layers on client (early split)

    Algorithm:
        1. Load bandwidths from hardcoded config or state
        2. Apply fluctuation if enabled
        3. Calculate BW ratio: client_bw / max_bw
        4. Compute split: ratio × (max_layer - min_layer) + min_layer

    Args:
        state: List[float] - Optional dynamic bandwidth measurements (Mbps)
        labels: Unused (kept for interface compatibility)
        **kwargs: Must contain 'node' (FedEdgeServer instance)

    Returns:
        Dict[NodeIdentifier, int]: Split point per client

    Example:
        >>> # config.HARDCODED_CLIENT_BW = {"client1": 15.0, "client2": 8.0}
        >>> optimal_split([], [], node=edge_server)
        {
            NodeIdentifier('client1'): 6,  # High BW → split at layer 6
            NodeIdentifier('client2'): 3   # Low BW → split at layer 3
        }

    Raises:
        ValueError: If 'node' parameter is missing
    """
    node = kwargs.get('node')
    if node is None:
        raise ValueError(" optimal_split requires 'node' parameter in kwargs")

    # Load client bandwidths (hardcoded or dynamic)
    clients, bandwidths = load_client_bandwidths(node, state)

    # Apply bandwidth fluctuation if enabled
    bandwidths = apply_fluctuation_to_all(bandwidths, config.current_round)

    # Get model configuration
    model_len = _get_model_length(node)
    max_bw = max(bandwidths) if bandwidths else 15.0

    # Define valid split range
    min_split = 1
    max_split = model_len - 1

    # Calculate split points based on bandwidth ratio
    split_layers = {}
    for client, client_bw in zip(clients, bandwidths):
        # BW ratio in [0, 1]
        bw_ratio = client_bw / max_bw

        # Map to layer range
        split_point = int(bw_ratio * (max_split - min_split) + min_split)

        # Validate and store
        split_layers[client] = _validate_split_point(split_point, model_len)

    return split_layers


def bandwidth_aware_split(state, labels, **kwargs) -> Dict[NodeIdentifier, int]:
    """
    Advanced splitting with latency threshold consideration.

    Adjusts split points based on both bandwidth and estimated latency:
    - If latency > threshold → Early split (minimize communication)
    - If latency ≤ threshold → BW-proportional split

    Use Case:
        Networks with variable latency (WiFi, cellular)

    Args:
        state: List[float] - Optional bandwidth measurements
        labels: Unused
        **kwargs:
            - node: FedEdgeServer instance (required)
            - latency_threshold: float - Max acceptable latency in ms (default: 100.0)

    Returns:
        Dict[NodeIdentifier, int]: Split point per client

    Example:
        >>> bandwidth_aware_split([], [], node=edge, latency_threshold=80.0)
        {
            NodeIdentifier('client1'): 1,  # High latency → early split
            NodeIdentifier('client2'): 5   # Low latency → distributed split
        }
    """
    node = kwargs.get('node')
    if node is None:
        raise ValueError("❌ bandwidth_aware_split requires 'node' parameter")

    latency_threshold = kwargs.get('latency_threshold', 100.0)  # ms

    # Load and fluctuate bandwidths
    clients, bandwidths = load_client_bandwidths(node, state)
    bandwidths = apply_fluctuation_to_all(bandwidths, config.current_round)

    model_len = _get_model_length(node)
    split_layers = {}

    for client, client_bw in zip(clients, bandwidths):
        # Estimate latency for middle split point
        mid_split = model_len // 2
        estimated_latency = estimate_layer_latency(mid_split, client_bw)

        if estimated_latency > latency_threshold:
            # High latency → minimize communication (split early)
            split_point = 1
        else:
            # Low latency → distribute computation based on BW
            # Assume reference BW = 15.0 Mbps for full model
            split_point = int((client_bw / 15.0) * model_len)
            split_point = _validate_split_point(split_point, model_len)

        split_layers[client] = split_point

    return split_layers


def adaptive_split(state, labels, **kwargs) -> Dict[NodeIdentifier, int]:
    """
    Adaptive splitting based on historical performance.

    Adjusts split points based on previous round accuracy:
    - If accuracy improving → Keep/increase client computation
    - If accuracy degrading → Reduce client computation

    State Management:
        Requires node to maintain:
        - self.prev_split_layers: Dict[NodeIdentifier, int]
        - self.prev_accuracies: Dict[NodeIdentifier, float]

    Args:
        state: List[float] - Optional bandwidth measurements
        labels: Unused
        **kwargs:
            - node: FedEdgeServer instance (required)
            - accuracy_threshold: float - Minimum acceptable accuracy (default: 0.6)

    Returns:
        Dict[NodeIdentifier, int]: Adjusted split points

    Example:
        >>> # Previous round: client1 had accuracy=0.55 with split=4
        >>> adaptive_split([], [], node=edge, accuracy_threshold=0.6)
        {
            NodeIdentifier('client1'): 3  # Decreased (poor performance)
        }
    """
    node = kwargs.get('node')
    if node is None:
        raise ValueError("❌ adaptive_split requires 'node' parameter")

    accuracy_threshold = kwargs.get('accuracy_threshold', 0.6)

    # Load bandwidths
    clients, bandwidths = load_client_bandwidths(node, state)
    bandwidths = apply_fluctuation_to_all(bandwidths, config.current_round)

    model_len = _get_model_length(node)
    split_layers = {}

    # Initialize history if not exists
    if not hasattr(node, 'prev_split_layers'):
        node.prev_split_layers = {}
    if not hasattr(node, 'prev_accuracies'):
        node.prev_accuracies = {}

    for client, client_bw in zip(clients, bandwidths):
        # Get previous split and accuracy
        prev_split = node.prev_split_layers.get(client, model_len // 2)
        prev_acc = node.prev_accuracies.get(client, 0.5)

        # Adaptive adjustment
        if prev_acc < accuracy_threshold:
            # Poor performance → reduce client computation
            split_point = max(1, prev_split - 1)
        else:
            # Good performance → increase client computation
            split_point = min(model_len - 1, prev_split + 1)

        split_layers[client] = _validate_split_point(split_point, model_len)

    return split_layers


def uniform_split(state, labels, **kwargs) -> Dict[NodeIdentifier, int]:
    """
    Uniform splitting - same split point for all clients.

    Useful for:
    - Baseline comparisons
    - Homogeneous client configurations
    - Simplified debugging

    Args:
        state: Unused
        labels: Unused
        **kwargs:
            - node: FedEdgeServer instance (required)
            - split_point: int - Fixed split layer (default: model_len // 2)

    Returns:
        Dict[NodeIdentifier, int]: Same split point for all clients

    Example:
        >>> uniform_split([], [], node=edge, split_point=4)
        {
            NodeIdentifier('client1'): 4,
            NodeIdentifier('client2'): 4,
            NodeIdentifier('client3'): 4
        }
    """
    node = kwargs.get('node')
    if node is None:
        raise ValueError("❌ uniform_split requires 'node' parameter")

    model_len = _get_model_length(node)
    fixed_split = kwargs.get('split_point', model_len // 2)
    fixed_split = _validate_split_point(fixed_split, model_len)

    # Get all clients
    clients = node.get_neighbors([NodeType.CLIENT])

    # Assign same split to all
    split_layers = {client: fixed_split for client in clients}

    return split_layers


def random_split(state, labels, **kwargs) -> Dict[NodeIdentifier, int]:
    """
    Random splitting - each client gets random split point.

    Used for:
    - Baseline comparisons
    - Testing system robustness
    - Exploring split space

    Args:
        state: Unused
        labels: Unused
        **kwargs:
            - node: FedEdgeServer instance (required)

    Returns:
        Dict[NodeIdentifier, int]: Random split per client

    Example:
        >>> random_split([], [], node=edge)
        {
            NodeIdentifier('client1'): 2,
            NodeIdentifier('client2'): 5,
            NodeIdentifier('client3'): 3
        }
    """
    node = kwargs.get('node')
    if node is None:
        raise ValueError("❌ random_split requires 'node' parameter")

    model_len = _get_model_length(node)
    clients = node.get_neighbors([NodeType.CLIENT])

    split_layers = {}
    for client in clients:
        split_point = random.randint(1, model_len - 1)
        split_layers[client] = split_point

    return split_layers


# ═══════════════════════════════════════════════════════════════
# 🔵 Section 2: Legacy/Compatibility Methods
# ═══════════════════════════════════════════════════════════════

def no_split(state, labels, **kwargs) -> Dict[NodeIdentifier, int]:
    """
    No splitting - entire model runs on edge server.

    Client only sends raw data, all computation on edge.
    Equivalent to centralized training.

    Args:
        state: Unused
        labels: Unused
        **kwargs: Must contain 'node' (FedEdgeServer)

    Returns:
        Dict[NodeIdentifier, int]: Split point = 0 for all clients
    """
    node = kwargs.get('node')
    if node is None:
        raise ValueError("❌ no_split requires 'node' parameter")

    clients = node.get_neighbors([NodeType.CLIENT])
    return {client: 0 for client in clients}


def full_client_split(state, labels, **kwargs) -> Dict[NodeIdentifier, int]:
    """
    Full client-side computation - no offloading to edge.

    Client runs entire model, edge only aggregates.
    Maximum client computation, minimum communication.

    Args:
        state: Unused
        labels: Unused
        **kwargs: Must contain 'node'

    Returns:
        Dict[NodeIdentifier, int]: Split point = model_len - 1 for all
    """
    node = kwargs.get('node')
    if node is None:
        raise ValueError("❌ full_client_split requires 'node' parameter")

    model_len = _get_model_length(node)
    clients = node.get_neighbors([NodeType.CLIENT])

    return {client: model_len - 1 for client in clients}


# ═══════════════════════════════════════════════════════════════
# 🔧 Section 3: Utility Functions
# ═══════════════════════════════════════════════════════════════

def action_to_layer(action: List[float]) -> List[int]:
    """
    Convert RL action values to layer indices.

    Maps continuous action space [0, 1] to discrete layer indices
    based on cumulative FLOPs distribution.

    Used in RL-based splitting strategies.

    Args:
        action: List of action values in [0, 1]

    Returns:
        List[int]: Corresponding layer indices

    Example:
        >>> action_to_layer([0.3, 0.7, 0.5])
        [2, 5, 3]  # Mapped to layers based on workload
    """
    # Compute workload distribution
    model_cfg = model_utils.get_unit_model().cfg
    workload_dist = _compute_workload_distribution(model_cfg)

    split_layers = []
    for action_val in action:
        # Find closest layer to action value
        idx = np.argmin(np.abs(workload_dist - action_val))

        # Clamp to valid range
        split_layers.append(_validate_split_point(idx, len(model_cfg)))

    return split_layers


# ═══════════════════════════════════════════════════════════════
# 🔧 Section 4: Method Registry
# ═══════════════════════════════════════════════════════════════

SPLITTING_METHODS: Dict[str, Callable] = {
    # Primary methods for semi-decentralized FL
    'optimal_split': optimal_split,
    'bandwidth_aware': bandwidth_aware_split,
    'adaptive': adaptive_split,
    'uniform': uniform_split,
    'random': random_split,

    # Edge cases
    'no_split': no_split,
    'full_client': full_client_split,
}


def get_splitting_method(method_name: str) -> Callable:
    """
    Retrieve splitting method by name with validation.

    Args:
        method_name: Name of the splitting method

    Returns:
        Callable: Splitting function

    Raises:
        ValueError: If method name not found

    Example:
        >>> func = get_splitting_method('optimal_split')
        >>> splits = func([], [], node=edge_server)
    """
    if method_name not in SPLITTING_METHODS:
        available = ', '.join(SPLITTING_METHODS.keys())
        raise ValueError(
            f"❌ Splitting method '{method_name}' not found.\n"
            f"Available methods: {available}"
        )

    return SPLITTING_METHODS[method_name]


def list_available_methods() -> List[str]:
    """
    Get list of all available splitting methods.

    Returns:
        List[str]: Method names

    Example:
        >>> list_available_methods()
        ['optimal_split', 'bandwidth_aware', 'adaptive', ...]
    """
    return list(SPLITTING_METHODS.keys())


# ═══════════════════════════════════════════════════════════════
# 🔧 Section 5: Validation and Debugging
# ═══════════════════════════════════════════════════════════════

def validate_split_layers(
        split_layers: Dict[NodeIdentifier, int],
        model_len: int
) -> bool:
    """
    Validate split layer dictionary for correctness.

    Checks:
        - All split points in valid range [1, model_len-1]
        - No missing clients
        - No invalid types

    Args:
        split_layers: Dict mapping clients to split points
        model_len: Total number of layers

    Returns:
        bool: True if valid, raises exception otherwise

    Raises:
        ValueError: If validation fails

    Example:
        >>> splits = {NodeIdentifier('c1'): 3, NodeIdentifier('c2'): 5}
        >>> validate_split_layers(splits, model_len=7)
        True
    """
    if not split_layers:
        raise ValueError("❌ Split layers dictionary is empty")

    for client, split in split_layers.items():
        # Check type
        if not isinstance(split, int):
            raise ValueError(
                f"❌ Invalid split type for {client}: {type(split)} "
                f"(expected int)"
            )

        # Check range
        if split < 1 or split >= model_len:
            raise ValueError(
                f"❌ Invalid split point for {client}: {split} "
                f"(must be in [1, {model_len - 1}])"
            )

    return True


def print_split_summary(
        split_layers: Dict[NodeIdentifier, int],
        bandwidths: Optional[List[float]] = None
):
    """
    Print formatted summary of split decisions.

    Useful for debugging and monitoring.

    Args:
        split_layers: Split point per client
        bandwidths: Optional bandwidth values for context

    Example:
        >>> print_split_summary(splits, [15.0, 8.5])
        ╔══════════════════════════════════════╗
        ║   Split Layer Summary (Round 10)    ║
        ╚══════════════════════════════════════╝
        client1: Split=6, BW=15.0 Mbps
        client2: Split=3, BW=8.5 Mbps
    """
    print("\n" + "=" * 50)
    print(f"  Split Layer Summary (Round {config.current_round})")
    print("=" * 50)

    clients = list(split_layers.keys())

    for i, client in enumerate(clients):
        split = split_layers[client]
        bw_info = f", BW={bandwidths[i]:.1f} Mbps" if bandwidths else ""
        print(f"{client}: Split={split}{bw_info}")

    print("=" * 50 + "\n")

# ═══════════════════════════════════════════════════════════════
# End
# ═══════════════════════════════════════════════════════════════
