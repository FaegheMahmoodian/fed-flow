"""
Semi-Decentralized FL Splitting Module

Architecture:
    Client ←→ Edge Server (Single split point per client)

Features:
    - Hardcoded bandwidth support via config.USE_HARDCODED_BW
    - Single-layer splitting (no cloud server)
    - Multiple splitting strategies (optimal, adaptive, uniform, etc.)
"""

import random
import numpy as np
from typing import Dict, List, Optional, Callable

from app.config import config
from app.entity.fed_base_node_interface import FedBaseNodeInterface
from app.entity.node_identifier import NodeIdentifier
from app.entity.node_type import NodeType
from app.util import model_utils
from app.dto.bandwidth import BandWidth


# ═══════════════════════════════════════════════════════════════
#  Helper Functions
# ═══════════════════════════════════════════════════════════════

def _validate_split_point(split_point: int, model_len: int) -> int:
    """
    Clamp split point to valid range [1, model_len-1].

    Args:
        split_point: Proposed split layer index
        model_len: Total number of layers

    Returns:
        int: Valid split point
    """
    min_split = 1
    max_split = model_len - 1
    return max(min_split, min(split_point, max_split))


def _get_model_length(node: FedBaseNodeInterface) -> int:
    """
    Get total number of model layers.

    Args:
        node: FedEdgeServer or FedClient instance

    Returns:
        int: Number of layers
    """
    try:
        return len(node.uninet.cfg)
    except AttributeError:
        return config.model_len


def _compute_workload_distribution(model_cfg: List) -> np.ndarray:
    """
    Compute normalized cumulative workload (FLOPs) per layer.

    Args:
        model_cfg: Model configuration list

    Returns:
        np.ndarray: Normalized cumulative workload [0.0, ..., 1.0]
    """
    workload = []
    cumulated_flops = 0

    for layer in model_cfg:
        cumulated_flops += layer[5]  # FLOPs at index 5
        workload.append(cumulated_flops)

    workload_array = np.array(workload)
    total_flops = cumulated_flops

    if total_flops > 0:
        workload_array = workload_array / total_flops

    return workload_array


def _load_client_bandwidths(node: FedBaseNodeInterface, state: Optional[List[float]]) -> tuple:
    """
    Load client bandwidths from hardcoded config or measured state.

    Priority:
        1. Hardcoded mode (config.USE_HARDCODED_BW = True)
        2. State parameter (measured bandwidths)
        3. node.neighbor_bandwidth (cached measurements)

    Args:
        node: Edge server instance
        state: Optional list of measured bandwidths (Mbps)

    Returns:
        tuple: (clients: List[NodeIdentifier], bandwidths: List[float in Mbps])
    """
    clients = node.get_neighbors([NodeType.CLIENT])

    if config.USE_HARDCODED_BW:
        # Use hardcoded bandwidths from node._get_hardcoded_bandwidth()
        bandwidths = []
        for client in clients:
            bw = node._get_hardcoded_bandwidth(client)
            bandwidths.append(bw.to_mbps())
    else:
        # Use measured bandwidths
        if state and len(state) > 0:
            bandwidths = state
        else:
            # Fallback to cached neighbor_bandwidth
            bandwidths = []
            for client in clients:
                if client in node.neighbor_bandwidth:
                    bw = node.neighbor_bandwidth[client]
                    bandwidths.append(bw.to_mbps())
                else:
                    # Default fallback: 15 Mbps
                    bandwidths.append(15.0)

    return clients, bandwidths


def _apply_fluctuation(bandwidths: List[float], round_num: int) -> List[float]:
    """
    Apply random fluctuation to simulate network variance.

    Only active if config.USE_BW_FLUCTUATION = True.
    Uses deterministic seed based on round number for reproducibility.

    Args:
        bandwidths: List of bandwidth values (Mbps)
        round_num: Current training round

    Returns:
        List[float]: Fluctuated bandwidths (±10% variance)
    """
    if not getattr(config, 'USE_BW_FLUCTUATION', False):
        return bandwidths

    fluctuation_rate = 0.1  # ±10%
    fluctuated = []

    for i, bw in enumerate(bandwidths):
        # Deterministic seed per client per round
        random.seed(hash(f"{round_num}_{i}_{bw}"))
        variance = random.uniform(-fluctuation_rate, fluctuation_rate)
        new_bw = bw * (1 + variance)
        fluctuated.append(max(1.0, new_bw))  # Min 1 Mbps

    return fluctuated


def _estimate_layer_latency(bandwidth_mbps: float, layer_idx: int, model_cfg: List) -> float:
    """
    Estimate communication latency for transmitting layer output.

    Args:
        bandwidth_mbps: Client bandwidth in Mbps
        layer_idx: Layer index
        model_cfg: Model configuration

    Returns:
        float: Estimated latency in milliseconds
    """
    if layer_idx >= len(model_cfg):
        return 0.0

    # Get layer output size (bytes) at index 6
    layer_output_size = model_cfg[layer_idx][6]

    # Convert bandwidth to bytes/sec
    bw_bytes_per_sec = (bandwidth_mbps * 1_000_000) / 8

    # Calculate transmission time
    if bw_bytes_per_sec > 0:
        transmission_time_sec = layer_output_size / bw_bytes_per_sec
        return transmission_time_sec * 1000  # Convert to ms
    else:
        return float('inf')


# ═══════════════════════════════════════════════════════════════
#  Bandwidth-Aware Splitting Methods
# ═══════════════════════════════════════════════════════════════

def optimal_split(state, labels, **kwargs) -> Dict[NodeIdentifier, int]:
    """
    Optimal bandwidth-proportional splitting.

    Higher BW → More layers on client (later split)
    Lower BW → Fewer layers on client (earlier split)

    Algorithm:
        split = (client_bw / max_bw) × (max_layer - min_layer) + min_layer

    Args:
        state: Optional list of measured bandwidths (Mbps)
        labels: Unused (interface compatibility)
        **kwargs: Must contain 'node' (FedEdgeServer)

    Returns:
        Dict[NodeIdentifier, int]: Split point per client

    Raises:
        ValueError: If 'node' not in kwargs
    """
    node = kwargs.get('node')
    if node is None:
        raise ValueError("optimal_split requires 'node' parameter in kwargs")

    # Load bandwidths (hardcoded or measured)
    clients, bandwidths = _load_client_bandwidths(node, state)

    # Apply fluctuation if enabled
    bandwidths = _apply_fluctuation(bandwidths, config.current_round)

    # Get model length
    model_len = _get_model_length(node)
    max_bw = max(bandwidths) if bandwidths else 15.0

    # Define valid split range
    min_split = 1
    max_split = model_len - 1

    # Calculate split points
    split_layers = {}
    for client, client_bw in zip(clients, bandwidths):
        bw_ratio = client_bw / max_bw
        split_point = int(bw_ratio * (max_split - min_split) + min_split)
        split_layers[client] = _validate_split_point(split_point, model_len)

    return split_layers


def bandwidth_aware_split(state, labels, **kwargs) -> Dict[NodeIdentifier, int]:
    """
    Splitting with latency threshold consideration.

    If estimated latency > threshold → Early split (minimize communication)
    Else → Bandwidth-proportional split

    Args:
        state: Optional bandwidth measurements
        labels: Unused
        **kwargs:
            - node: FedEdgeServer (required)
            - latency_threshold: Max acceptable latency in ms (default: 100.0)

    Returns:
        Dict[NodeIdentifier, int]: Split point per client
    """
    node = kwargs.get('node')
    if node is None:
        raise ValueError("bandwidth_aware_split requires 'node' parameter")

    latency_threshold = kwargs.get('latency_threshold', 100.0)  # ms

    # Load and fluctuate bandwidths
    clients, bandwidths = _load_client_bandwidths(node, state)
    bandwidths = _apply_fluctuation(bandwidths, config.current_round)

    model_len = _get_model_length(node)
    model_cfg = node.uninet.cfg if hasattr(node, 'uninet') else config.model_cfg.get(config.model_name, [])

    split_layers = {}

    for client, client_bw in zip(clients, bandwidths):
        # Estimate latency for middle split
        mid_split = model_len // 2
        estimated_latency = _estimate_layer_latency(client_bw, mid_split, model_cfg)

        if estimated_latency > latency_threshold:
            # High latency → early split
            split_point = 1
        else:
            # Low latency → BW-proportional split
            split_point = int((client_bw / 15.0) * model_len)
            split_point = _validate_split_point(split_point, model_len)

        split_layers[client] = split_point

    return split_layers


def adaptive_split(state, labels, **kwargs) -> Dict[NodeIdentifier, int]:
    """
    Adaptive splitting based on historical accuracy.

    Adjusts split based on previous round performance:
        - Accuracy < threshold → Reduce client computation
        - Accuracy ≥ threshold → Increase client computation

    Requires node attributes:
        - self.prev_split_layers: Dict[NodeIdentifier, int]
        - self.prev_accuracies: Dict[NodeIdentifier, float]

    Args:
        state: Optional bandwidth measurements
        labels: Unused
        **kwargs:
            - node: FedEdgeServer (required)
            - accuracy_threshold: Minimum acceptable accuracy (default: 0.6)

    Returns:
        Dict[NodeIdentifier, int]: Adjusted split points
    """
    node = kwargs.get('node')
    if node is None:
        raise ValueError("adaptive_split requires 'node' parameter")

    accuracy_threshold = kwargs.get('accuracy_threshold', 0.6)

    # Load bandwidths
    clients, bandwidths = _load_client_bandwidths(node, state)
    bandwidths = _apply_fluctuation(bandwidths, config.current_round)

    model_len = _get_model_length(node)
    split_layers = {}

    # Initialize history if not exists
    if not hasattr(node, 'prev_split_layers'):
        node.prev_split_layers = {}
    if not hasattr(node, 'prev_accuracies'):
        node.prev_accuracies = {}

    for client in clients:
        # Get previous split and accuracy
        prev_split = node.prev_split_layers.get(client, model_len // 2)
        prev_acc = node.prev_accuracies.get(client, 0.5)

        # Adaptive adjustment
        if prev_acc < accuracy_threshold:
            split_point = max(1, prev_split - 1)  # Reduce client load
        else:
            split_point = min(model_len - 1, prev_split + 1)  # Increase client load

        split_layers[client] = _validate_split_point(split_point, model_len)

    return split_layers


def uniform_split(state, labels, **kwargs) -> Dict[NodeIdentifier, int]:
    """
    Uniform splitting - same split point for all clients.

    Use cases:
        - Baseline comparisons
        - Homogeneous environments
        - Debugging

    Args:
        state: Unused
        labels: Unused
        **kwargs:
            - node: FedEdgeServer (required)
            - split_point: Fixed split layer (default: model_len // 2)

    Returns:
        Dict[NodeIdentifier, int]: Same split for all clients
    """
    node = kwargs.get('node')
    if node is None:
        raise ValueError("uniform_split requires 'node' parameter")

    model_len = _get_model_length(node)
    fixed_split = kwargs.get('split_point', model_len // 2)
    fixed_split = _validate_split_point(fixed_split, model_len)

    clients = node.get_neighbors([NodeType.CLIENT])
    split_layers = {client: fixed_split for client in clients}

    return split_layers


def random_split(state, labels, **kwargs) -> Dict[NodeIdentifier, int]:
    """
    Random splitting - each client gets random split point.

    Use cases:
        - Baseline comparisons
        - Robustness testing
        - Exploration

    Args:
        state: Unused
        labels: Unused
        **kwargs: Must contain 'node'

    Returns:
        Dict[NodeIdentifier, int]: Random split per client
    """
    node = kwargs.get('node')
    if node is None:
        raise ValueError("random_split requires 'node' parameter")

    model_len = _get_model_length(node)
    clients = node.get_neighbors([NodeType.CLIENT])

    split_layers = {}
    for client in clients:
        split_point = random.randint(1, model_len - 1)
        split_layers[client] = split_point

    return split_layers


# ═══════════════════════════════════════════════════════════════
#  Edge Case Methods
# ═══════════════════════════════════════════════════════════════

def no_split(state, labels, **kwargs) -> Dict[NodeIdentifier, int]:
    """
    No splitting - entire model on edge server.

    Client sends raw data, edge does all computation.
    Equivalent to centralized training.

    Args:
        state: Unused
        labels: Unused
        **kwargs: Must contain 'node'

    Returns:
        Dict[NodeIdentifier, int]: Split = 0 for all clients
    """
    node = kwargs.get('node')
    if node is None:
        raise ValueError("no_split requires 'node' parameter")

    clients = node.get_neighbors([NodeType.CLIENT])
    return {client: 0 for client in clients}


def full_client_split(state, labels, **kwargs) -> Dict[NodeIdentifier, int]:
    """
    Full client-side computation - no offloading.

    Client runs entire model, edge only aggregates.

    Args:
        state: Unused
        labels: Unused
        **kwargs: Must contain 'node'

    Returns:
        Dict[NodeIdentifier, int]: Split = model_len - 1 for all
    """
    node = kwargs.get('node')
    if node is None:
        raise ValueError("full_client_split requires 'node' parameter")

    model_len = _get_model_length(node)
    clients = node.get_neighbors([NodeType.CLIENT])

    return {client: model_len - 1 for client in clients}


# ═══════════════════════════════════════════════════════════════
#  Utility Functions
# ═══════════════════════════════════════════════════════════════

def action_to_layer(action: List[float]) -> List[int]:
    """
    Convert RL action values [0, 1] to discrete layer indices.

    Maps continuous action space to layers based on cumulative FLOPs.

    Args:
        action: List of action values in [0, 1]

    Returns:
        List[int]: Corresponding layer indices
    """
    model_cfg = model_utils.get_unit_model().cfg
    workload_dist = _compute_workload_distribution(model_cfg)

    split_layers = []
    for action_val in action:
        idx = np.argmin(np.abs(workload_dist - action_val))
        split_layers.append(_validate_split_point(idx, len(model_cfg)))

    return split_layers


# ═══════════════════════════════════════════════════════════════
#  Method Registry
# ═══════════════════════════════════════════════════════════════

SPLITTING_METHODS: Dict[str, Callable] = {
    # Primary methods
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
    Retrieve splitting method by name.

    Args:
        method_name: Name of splitting method

    Returns:
        Callable: Splitting function

    Raises:
        ValueError: If method not found
    """
    if method_name not in SPLITTING_METHODS:
        available = ', '.join(SPLITTING_METHODS.keys())
        raise ValueError(
            f"Splitting method '{method_name}' not found. "
            f"Available: {available}"
        )

    return SPLITTING_METHODS[method_name]


def list_available_methods() -> List[str]:
    """
    Get list of available splitting methods.

    Returns:
        List[str]: Method names
    """
    return list(SPLITTING_METHODS.keys())


# ═══════════════════════════════════════════════════════════════
#  Validation and Debugging
# ═══════════════════════════════════════════════════════════════

def validate_split_layers(split_layers: Dict[NodeIdentifier, int], model_len: int) -> bool:
    """
    Validate split layer dictionary.

    Checks:
        - All splits in valid range [1, model_len-1]
        - Correct data types

    Args:
        split_layers: Dict mapping clients to splits
        model_len: Total number of layers

    Returns:
        bool: True if valid

    Raises:
        ValueError: If validation fails
    """
    if not split_layers:
        raise ValueError("Split layers dictionary is empty")

    for client, split in split_layers.items():
        if not isinstance(split, int):
            raise ValueError(
                f"Invalid split type for {client}: {type(split)} (expected int)"
            )

        if split < 1 or split >= model_len:
            raise ValueError(
                f"Invalid split point for {client}: {split} "
                f"(must be in [1, {model_len - 1}])"
            )

    return True


def print_split_summary(split_layers: Dict[NodeIdentifier, int],
                       bandwidths: Optional[List[float]] = None):
    """
    Print formatted summary of split decisions.

    Args:
        split_layers: Split point per client
        bandwidths: Optional bandwidth values
    """
    print("\n" + "=" * 50)
    print(f"  Split Layer Summary (Round {config.current_round})")
    print("=" * 50)

    clients = list(split_layers.keys())

    for i, client in enumerate(clients):
        split = split_layers[client]
        bw_info = f", BW={bandwidths[i]:.1f} Mbps" if bandwidths and i < len(bandwidths) else ""
        print(f"{client}: Split={split}{bw_info}")

    print("=" * 50 + "\n")
