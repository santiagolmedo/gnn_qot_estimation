import networkx as nx
import xarray as xr
import numpy as np

dataset_cache = None


def load_dataset(dataset_path):
    global dataset_cache
    if dataset_cache is not None:
        return dataset_cache

    dataset = xr.open_dataset(dataset_path)

    # Extract relevant variables
    lp_feat_values = dataset["lp_feat"].values  # List of lightpath features
    data_values = dataset["data"].values  # [sample, lp_feat, link, freq]
    metrics = dataset["metric"].values  # Names of metrics
    target_values = dataset["target"].values  # [sample, metric]

    # Indexes of the features
    feature_indexes = {feature: idx for idx, feature in enumerate(lp_feat_values)}
    dataset_cache = (
        data_values,
        metrics,
        target_values,
        feature_indexes,
    )
    return dataset_cache


def create_baseline_graph(sample_index, features_to_consider, dataset_path):
    """
    This method generates a graph representation of the network state for a given sample. The network is modeled
    as a set of nodes (representing network nodes) and edges (representing lightpaths between source and destination nodes).
    The key transformation here involves converting the 3D tensor representing the lightpath activity in the network
    into a graph, where:

    - **Nodes**: Each graph node corresponds to a physical node in the network (IDs from 1 to 75). These nodes are
      added without specific features (i.e., they are featureless), representing the 75 physical network nodes.
      Each node serves as an anchor for the lightpaths and edges between nodes.

    - **Edges**: An edge is added between two nodes (source and destination) if there is an active lightpath
      connecting them. The presence of an edge signifies that data is being transmitted between these two network nodes.
      Each edge represents the link between a pair of nodes, and it carries several key features from the dataset that
      describe the characteristics of the lightpath utilizing the link at a particular frequency.

    The graph is constructed using the following steps:

    1. **Iterating through lightpaths**: For each network sample, the lightpath data is structured as a 3D tensor
       (`lp_feat`, `link`, `freq`), where:
       - `lp_feat`: Represents the features of each lightpath (e.g., modulation order, path length, OSNR, etc.).
       - `link`: Indicates which network links are used by the lightpath.
       - `freq`: Represents the frequency at which the lightpath operates.

    2. **Identifying active lightpaths**: Each lightpath feature vector is examined for a given link and frequency.
       If the lightpath is active (i.e., the feature vector is not all zeros), the lightpath is processed and represented
       in the graph as an edge between the source node (`src_id`) and the destination node (`dst_id`).

    3. **Edge features**: The key characteristics of each lightpath (e.g., modulation order, number of spans, path length,
       frequency, OSNR, SNR, and BER) are extracted and assigned as attributes to the edge between the source and destination
       nodes. These features provide valuable information about the performance and configuration of the optical transmission
       link.

    4. **Avoiding duplicate lightpaths**: To ensure that each lightpath is only processed once, the `conn_id` (connection ID)
       is tracked in a set of processed lightpaths. This prevents redundant edges from being added between the same source
       and destination nodes for the same lightpath.

    5. **Graph labels**: The overall graph (representing the network state) is assigned target labels (`OSNR`, `SNR`, `BER`,
       and class) from the dataset, which serve as the outputs that will be predicted by models trained on this graph
       representation.

    By constructing the graph in this way, we capture the structure and functionality of the optical network in a format
    suitable for graph-based learning models like Graph Neural Networks (GNNs). The transformation from a 3D tensor of
    lightpath activity into a graph allows the use of powerful graph-based models to predict quality metrics (OSNR, SNR, BER)
    for the network state.

    Parameters:
    - sample_index (int): The index of the sample in the dataset to convert into a graph.
    - features_to_consider (list): List of lightpath features to use as edge attributes in the graph.
    - dataset_path (str): Path to the dataset file.

    Returns:
    - G (networkx.Graph): A graph representing the network state for the specified sample.
    """

    # Load dataset from the cache or the file
    (
        data_values,
        metrics,
        target_values,
        feature_indexes,
    ) = load_dataset(dataset_path)

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
