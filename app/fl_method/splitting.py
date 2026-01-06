import random
import numpy as np
import torch
# from stable_baselines3 import PPO

from app.config import config
from app.entity.fed_base_node_interface import FedBaseNodeInterface
from app.entity.http_communicator import HTTPCommunicator
from app.entity.node_type import NodeType
from app.util import model_utils, graph_utils

def _client_neighbors(node):
    # FIX: always use actual connected client neighbors (prevents mismatch with config.K/CLIENTS_LIST)
    return list(node.get_neighbors([NodeType.CLIENT])) if node is not None else []

# ------------------ No offloading => full model on client -------------------------
def none(state, labels, node=None, **kwargs):
    # FIX: accept edge server instance for compatibility with fl_method_parser
    # FIX: one-cut + return dict keyed by neighbor for scatter_split_layers compatibility
    neighbors = _client_neighbors(node)
    cut = model_utils.get_unit_model_len() - 1
    return {n: cut for n in neighbors}


def no_edge_fake(state, labels, node=None, **kwargs):
    # Debug: accept optional 'node' because FedEdgeServer.split passes (state, labels, node)
    # FIX: one-cut + return dict keyed by neighbor (avoids TypeError + KeyError in scatter)
    neighbors = _client_neighbors(node)
    splitting = {}
    for n in neighbors:
        splitting[n] = random.randint(1, model_utils.get_unit_model_len() - 1)  # avoid model_len mismatch

    return splitting


def fake(state, labels, node=None, **kwargs):
    # FIX: accept edge server instance for compatibility with fl_method_parser
    # FIX: one-cut version to keep split learning in two parts only
    neighbors = _client_neighbors(node)
    cut = 3
    return {n: cut for n in neighbors}


def fake_decentralized(state, labels, node=None, **kwargs):
    # FIX: accept edge server instance for compatibility with fl_method_parser
    # FIX: one-cut + always return dict keyed by client neighbor
    split_layers = {}
    if node is None:
        return split_layers

    if node.is_edge_based:
        for edge in node.get_neighbors([NodeType.EDGE]):
            for client in HTTPCommunicator.get_neighbors_from_neighbor(edge, [NodeType.CLIENT]):
                split_layers[client] = len(node.uninet.cfg) - 1  # one-cut: no offloading / full on client
    else:
        for neighbor in node.get_neighbors([NodeType.CLIENT]):
            split_layers[neighbor] = len(node.uninet.cfg) - 1  # one-cut
    return split_layers

# ------------------- choose last layer so effectively no offloading --------
def no_splitting(state, labels, node=None, **kwargs):
    # FIX: accept edge server instance for compatibility with fl_method_parser
    # FIX: one-cut version
    neighbors = _client_neighbors(node)
    cut = model_utils.get_unit_model_len() - 1
    return {n: cut for n in neighbors}


#---------------------- cut very early so most work on edge --------------
def only_edge_splitting(state, labels, node=None, **kwargs):
    # FIX: accept edge server instance for compatibility with fl_method_parser
    # FIX: one-cut version
    neighbors = _client_neighbors(node)
    cut = 1  # keep >=1 to avoid invalid split depending on model implementation
    return {n: cut for n in neighbors}



# ----------------  random splitting (HFLP used random) -----------
def randomSplitting(state, labels, node=None, **kwargs):
    # FIX: one-cut split for client-edge (returns int per client)
    neighbors = _client_neighbors(node)
    splitting = {}
    for n in neighbors:
        splitting[n] = random.randint(1, model_utils.get_unit_model_len() - 1)  # 1..model_len-1
    return splitting


# ------------ FedMec:  empirically deploys the convolutional layers of a DNN on the device-side -------
# assigning the remaining part to the edge server
def FedMec(state, labels, node=None, **kwargs):
    # FIX: accept edge server instance for compatibility with fl_method_parser
    if node is None:
        return {}  # safe fallback

    cfg = config.model_cfg["VGG5"]
    last = 0
    for idx, layer in enumerate(cfg):
        if isinstance(layer, (list, tuple)) and len(layer) > 0 and layer[0] == 'C':
            last = idx

    neighbors = node.get_neighbors([NodeType.CLIENT])
    return {n: last for n in neighbors}


# ---------------------------------for RL training ----------------------------------
# def expand_actions(actions, clients_list, group_labels):  # Expanding group actions to each device
#     full_actions = []
#
#     for i in range(len(clients_list)):
#         full_actions.append(actions[group_labels[i]])
#
#     return full_actions
#
#
# def action_to_layer(action):  # Expanding group actions to each device
#     # first caculate cumulated flops
#     model_state_flops = []
#     cumulated_flops = 0
#
#     for l in model_utils.get_unit_model().cfg:
#         cumulated_flops += l[5]
#         model_state_flops.append(cumulated_flops)
#
#     model_flops_list = np.array(model_state_flops)
#     model_flops_list = model_flops_list / cumulated_flops
#
#     split_layer = []
#     for v in action:
#         idx = np.where(np.abs(model_flops_list - v) == np.abs(model_flops_list - v).min())
#
#         idx = idx[0][-1]
#         if idx >= 5:  # all FC layers combine to one option
#             idx = 6
#         split_layer.append(idx)
#     return split_layer
#
#
# def actionToLayerEdgeBase(splitDecision: list[float]) -> tuple[int, int]:
#     """ It returns the offloading points for the given action ( op1 , op2 )"""
#     op1: int
#     op2: int  # Offloading points op1, op2
#     workLoad = []
#     model_state_flops = []
#
#     for l in model_utils.get_unit_model().cfg:
#         workLoad.append(l[5])
#         model_state_flops.append(sum(workLoad))
#
#     totalWorkLoad = sum(workLoad)
#     model_flops_list = np.array(model_state_flops)
#     model_flops_list = model_flops_list / totalWorkLoad
#     idx = np.where(np.abs(model_flops_list - splitDecision[0]) == np.abs(model_flops_list - splitDecision[0]).min())
#     op1 = idx[0][-1]
#
#     op2_totalWorkload = sum(workLoad[op1:])
#     model_state_flops = []
#     for l in range(op1, model_utils.get_unit_model_len()):
#         model_state_flops.append(sum(workLoad[op1:l + 1]))
#     model_flops_list = np.array(model_state_flops)
#     model_flops_list = model_flops_list / op2_totalWorkload
#
#     idx = np.where(np.abs(model_flops_list - splitDecision[1]) == np.abs(model_flops_list - splitDecision[1]).min())
#     op2 = idx[0][-1] + op1
#
#     return op1, op2
