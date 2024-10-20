import networkx as nx
import xarray as xr
import numpy as np

# Global cache for dataset metadata
dataset_metadata_cache = None


def load_dataset_metadata(dataset_path):
    """
    Loads dataset metadata and caches it to avoid redundant loading.

    Parameters:
    - dataset_path (str): Path to the dataset file.

    Returns:
    - metadata (dict): Dictionary containing dataset metadata.
    """
    global dataset_metadata_cache
    if dataset_metadata_cache is not None:
        return dataset_metadata_cache

    dataset = xr.open_dataset(dataset_path)

    # Extract relevant variables
    lp_feat_values = dataset["lp_feat"].values  # List of lightpath features
    metrics = dataset["metric"].values  # Names of metrics
    feature_indexes = {feature: idx for idx, feature in enumerate(lp_feat_values)}

    # Indexes for specific features
    conn_id_index = feature_indexes["conn_id"]
    osnr_index = feature_indexes["osnr"]
    snr_index = feature_indexes["snr"]
    ber_index = feature_indexes["ber"]

    # Lists of links and frequencies
    links = dataset["link"].values  # List of links
    freqs = dataset["freq"].values  # List of frequencies

    # Close the dataset after extracting metadata
    dataset.close()

    # Cache the metadata
    dataset_metadata_cache = {
        "lp_feat_values": lp_feat_values,
        "metrics": metrics,
        "feature_indexes": feature_indexes,
        "conn_id_index": conn_id_index,
        "osnr_index": osnr_index,
        "snr_index": snr_index,
        "ber_index": ber_index,
        "links": links,
        "freqs": freqs,
    }
    return dataset_metadata_cache


def spectral_distance(freq1, freq2):
    return abs(freq1 - freq2)


def create_topological_graph(sample_index, features_to_consider, dataset_path):
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
    print("Creating topological graph for sample", sample_index)

    # Load dataset metadata
    metadata = load_dataset_metadata(dataset_path)
    feature_indexes = metadata["feature_indexes"]
    metrics = metadata["metrics"]

    # Open the dataset and access the sample data
    dataset = xr.open_dataset(dataset_path)
    sample_data = dataset["data"].isel(
        sample=sample_index
    )  # Shape: [lp_feat, link, freq]
    target_vector = dataset["target"].isel(sample=sample_index).values
    dataset.close()

    G = nx.Graph()

    # Add network nodes without features (IDs from 1 to 75)
    num_nodes = 75
    G.add_nodes_from(range(1, num_nodes + 1))

    # Convert sample_data to NumPy array
    sample_data_np = sample_data.values  # Shape: [lp_feat, num_links, num_freqs]

    # Identify occupied channels
    occupied = np.any(sample_data_np != 0, axis=0)  # Shape: [num_links, num_freqs]
    occupied_indices = np.argwhere(occupied)  # Shape: [N_occupied, 2]

    # Extract lightpath features for occupied channels
    lp_feat_vectors = sample_data_np[:, occupied]  # Shape: [lp_feat, N_occupied]

    # Flatten link and freq indices
    link_indices = occupied_indices[:, 0]
    freq_indices = occupied_indices[:, 1]

    # Extract conn_id, src_id, dst_id for occupied channels
    conn_ids = lp_feat_vectors[feature_indexes["conn_id"], :].astype(int)
    src_ids = lp_feat_vectors[feature_indexes["src_id"], :].astype(int)
    dst_ids = lp_feat_vectors[feature_indexes["dst_id"], :].astype(int)

    # Create a mask for unique conn_ids
    _, unique_indices = np.unique(conn_ids, return_index=True)

    # Extract unique lightpaths
    unique_conn_ids = conn_ids[unique_indices]
    unique_src_ids = src_ids[unique_indices]
    unique_dst_ids = dst_ids[unique_indices]
    unique_lp_feat_vectors = lp_feat_vectors[:, unique_indices]

    # Collect edge features for unique lightpaths
    edge_features_list = []
    for i in range(len(unique_conn_ids)):
        edge_features = {
            feature: unique_lp_feat_vectors[feature_indexes[feature], i]
            for feature in features_to_consider
        }
        edge_features_list.append(edge_features)

    # Add edges to the graph
    for src_id, dst_id, edge_attrs in zip(
        unique_src_ids, unique_dst_ids, edge_features_list
    ):
        G.add_edge(src_id, dst_id, **edge_attrs)

    # Assign labels to the graph
    labels = dict(zip(metrics, target_vector))
    G.graph["labels"] = labels

    return G


def create_lightpath_graph(
    sample_index, features_to_consider, dataset_path, freq_threshold=0.05
):
    """
    Generates a graph where each node represents a lightpath, and edges represent interactions between lightpaths.

    Parameters:
    - sample_index (int): Index of the sample in the dataset.
    - features_to_consider (list): List of lightpath features to use as node attributes.
    - dataset_path (str): Path to the dataset file.
    - freq_threshold (float): Threshold for spectral distance to consider an interaction.

    Returns:
    - G (networkx.Graph): Graph representing the interactions between lightpaths for the specified sample.
    """
    print("Creating lightpath graph for sample", sample_index)

    # Load dataset metadata
    metadata = load_dataset_metadata(dataset_path)
    feature_indexes = metadata["feature_indexes"]
    metrics = metadata["metrics"]
    conn_id_index = metadata["conn_id_index"]
    osnr_index = metadata["osnr_index"]
    snr_index = metadata["snr_index"]
    ber_index = metadata["ber_index"]
    freqs = metadata["freqs"]
    links = metadata["links"]

    # Open the dataset and access the sample data
    dataset = xr.open_dataset(dataset_path)
    sample_data = dataset["data"].isel(sample=sample_index)
    target_vector = dataset["target"].isel(sample=sample_index).values
    dataset.close()

    G = nx.Graph()
    G.graph["labels"] = dict(zip(metrics, target_vector))

    # Convert sample_data to NumPy array for efficient processing
    sample_data_np = sample_data.values  # Shape: [lp_feat, num_links, num_freqs]

    # Identify occupied channels (where any feature is non-zero)
    occupied = np.any(sample_data_np != 0, axis=0)  # Shape: [num_links, num_freqs]

    # Get indices of occupied channels
    occupied_indices = np.argwhere(occupied)  # Shape: [N_occupied, 2]

    # Initialize dictionaries to store lightpath information
    lightpaths = {}

    # Build a mapping from links to lightpaths
    link_to_lightpaths = {}

    # Process each occupied channel
    for link_index, freq_index in occupied_indices:
        lp_feat_vector = sample_data_np[:, link_index, freq_index]

        conn_id = int(lp_feat_vector[conn_id_index])

        if conn_id not in lightpaths:
            # Determine if it is the LUT (is_lut)
            is_lut = int(
                lp_feat_vector[osnr_index] == -1
                and lp_feat_vector[snr_index] == -1
                and lp_feat_vector[ber_index] == -1
            )

            # Collect node features
            node_features = {
                feature: lp_feat_vector[feature_indexes[feature]]
                for feature in features_to_consider
            }
            node_features["is_lut"] = is_lut

            # Initialize lightpath information
            lightpaths[conn_id] = {
                "features": node_features,
                "freqs_per_link": {},  # {link_index: set of frequencies}
            }

        # Add frequency used on the link
        freqs_per_link = lightpaths[conn_id]["freqs_per_link"]
        freqs_per_link.setdefault(link_index, set()).add(freqs[freq_index])

        # Map link to lightpaths
        link_to_lightpaths.setdefault(link_index, set()).add(conn_id)

    # Add lightpath nodes to the graph
    for conn_id, info in lightpaths.items():
        node_id = f"lightpath_{conn_id}"
        G.add_node(node_id, **info["features"])

    # Build interactions between lightpaths per link
    for link_index, conn_ids in link_to_lightpaths.items():
        conn_ids = list(conn_ids)
        if len(conn_ids) < 2:
            continue  # No interactions possible

        # For this link, collect frequencies and lightpaths
        frequencies = []
        lp_ids = []
        for conn_id in conn_ids:
            freqs_on_link = lightpaths[conn_id]["freqs_per_link"][link_index]
            for freq in freqs_on_link:
                frequencies.append(freq)
                lp_ids.append(conn_id)

        frequencies = np.array(frequencies)
        lp_ids = np.array(lp_ids)

        # Compute pairwise spectral distances
        freq_diff = np.abs(frequencies[:, None] - frequencies[None, :])

        # Find pairs with spectral distance below threshold (excluding self-pairs)
        i_indices, j_indices = np.where((freq_diff < freq_threshold) & (freq_diff > 0))
        edges = set()
        for i, j in zip(i_indices, j_indices):
            lp_id_i = lp_ids[i]
            lp_id_j = lp_ids[j]
            edge = tuple(sorted((lp_id_i, lp_id_j)))
            if edge not in edges:
                node_id1 = f"lightpath_{lp_id_i}"
                node_id2 = f"lightpath_{lp_id_j}"
                G.add_edge(node_id1, node_id2)
                edges.add(edge)

    return G
