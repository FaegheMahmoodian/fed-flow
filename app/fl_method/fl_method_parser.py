from app.fl_method import clustering, splitting

# a mapping of fl methods to make function call easier
fl_methods = {
    # ============================================================
    # CLUSTERING METHODS
    # ============================================================
    "bandwidth": clustering.bandwidth,
    "none_clustering": clustering.none,

    # ============================================================
    # SPLITTING METHODS - Basic/Legacy
    # ============================================================
    "none_splitting": splitting.none,
    "no_edge_fake_splitting": splitting.no_edge_fake,

    # ============================================================
    # SPLITTING METHODS - Test/Fake Methods
    # ============================================================
    "fake_splitting": splitting.fake,
    "fake_decentralized_splitting": splitting.fake_decentralized,

    # ============================================================
    # SPLITTING METHODS - 3-Tier Architecture
    # ============================================================
    "no_splitting": splitting.no_splitting,
    "only_edge_splitting": splitting.only_edge_splitting,
    "only_server_splitting": splitting.only_server_splitting,
    "random_splitting": splitting.random_split,
    "fedmec_splitting": splitting.FedMec,

    # ============================================================
    # UTILITY/HELPER METHODS
    # ============================================================
    "expand_actions": splitting.expand_actions,
    "action_to_layer": splitting.action_to_layer,
    "action_to_layer_edge_base": splitting.actionToLayerEdgeBase,
}
