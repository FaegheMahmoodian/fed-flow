from app.fl_method import clustering, splitting

# a mapping of fl methods to make function call easier
fl_methods = {
    "bandwidth": clustering.bandwidth,
    "none_clustering": clustering.none,
    "none_splitting": splitting.none,
    "fake_splitting": splitting.fake,
    "fake_decentralized_splitting": splitting.fake_decentralized,
    "no_splitting": splitting.no_splitting,
    "no_edge_fake_splitting": splitting.no_edge_fake,
    "only_edge_splitting": splitting.only_edge_splitting,
    "random_splitting": splitting.randomSplitting,
    "fedmec_splitting": splitting.FedMec,
}
