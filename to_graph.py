import networkx as nx
import xarray as xr
import numpy as np

dataset_cache = None


def load_dataset():
    global dataset_cache
    if dataset_cache is not None:
        return dataset_cache

    dataset = xr.open_dataset("network_status_dataset_preview.nc")

    # Extract relevant variables
    lp_feat_values = dataset["lp_feat"].values  # List of lightpath features
    data_values = dataset["data"].values  # [sample, lp_feat, link, freq]
    samples = dataset["sample"].values  # Identifiers for samples
    metrics = dataset["metric"].values  # Names of metrics
    target_values = dataset["target"].values  # [sample, metric]

    # Indexes of the features with the format 'feature_name_index'
    feature_indexes = {feature: idx for idx, feature in enumerate(lp_feat_values)}
    dataset_cache = (
        lp_feat_values,
        data_values,
        samples,
        metrics,
        target_values,
        feature_indexes,
    )
    return dataset_cache


def create_baseline_graph(sample_index, features_to_consider, dataset_path):
    # Load dataset
    (
        lp_feat_values,
        data_values,
        samples,
        metrics,
        target_values,
        feature_indexes,
    ) = load_dataset()

    sample_data = data_values[sample_index]  # [lp_feat, link, freq]
    G = nx.Graph()

    # Add network nodes without features (IDs from 1 to 75)
    num_nodes = 75
    node_ids = range(1, num_nodes + 1)
    G.add_nodes_from(node_ids)

    # Set to store the lightpaths already processed (to avoid duplicates)
    processed_lightpaths = set()

    # Go over all links and frequencies to identify the lightpaths
    num_links = sample_data.shape[1]
    num_freqs = sample_data.shape[2]

    for link_index in range(num_links):
        for freq_index_in_loop in range(num_freqs):
            lp_feat_vector = sample_data[:, link_index, freq_index_in_loop]

            # Verify if the channel is occupied
            if not np.all(lp_feat_vector == 0):
                conn_id = int(lp_feat_vector[feature_indexes["conn_id"]])

                # Ignore if conn_id is zero
                if conn_id == 0:
                    continue

                # Avoid processing the same lightpath multiple times
                if conn_id in processed_lightpaths:
                    continue

                processed_lightpaths.add(conn_id)

                src_id = int(lp_feat_vector[feature_indexes["src_id"]])
                dst_id = int(lp_feat_vector[feature_indexes["dst_id"]])

                # Add the features to the edge if they are in the list of features to consider
                edge_features = {}
                for feature in features_to_consider:
                    edge_features[feature] = lp_feat_vector[feature_indexes[feature]]

                # Add the edge to the graph if the lightpath starts in src_id and ends in dst_id
                G.add_edge(src_id, dst_id, **edge_features)

    # Extract the labels for the current sample
    target_vector = target_values[sample_index]

    labels = dict(zip(metrics, target_vector))

    # Assign the labels to the graph as attributes
    G.graph["labels"] = labels

    return G
