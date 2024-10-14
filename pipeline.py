import networkx as nx
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pdb

# Load the dataset file
dataset = xr.open_dataset("network_status_dataset_preview_20.nc")
pdb.set_trace()

# Load the relevant dataset variables
lp_feat_values = dataset['lp_feat'].values
ntd_values = dataset['ntd'].values
data_values = dataset['data'].values
links = dataset['link'].values
freqs = dataset['freq'].values

# Extract network topology descriptor (ntd) features
nt_feat_values = dataset['nt_feat'].values

# Function to calculate spectral distance
def spectral_distance(freq1, freq2):
    return abs(freq1 - freq2)

# Function to create a graph for one sample using NetworkX, including LUT identification
def create_graph_nx(sample_index, freq_threshold=0.05):
    sample_data = data_values[sample_index]
    osnr_index = list(lp_feat_values).index('osnr')
    snr_index = list(lp_feat_values).index('snr')
    ber_index = list(lp_feat_values).index('ber')

    # Initialize a directed graph (or undirected if necessary)
    G = nx.Graph()

    # Dictionary to store lightpaths by link
    link_lightpaths = {}  # Stores lightpaths per link for easier comparison

    # Step 1: Add Lightpath Nodes and Lightpath-Link Edges
    for link_index in range(len(links)):
        link_lightpaths[link_index] = []  # Initialize the list of lightpaths per link
        for freq_index in range(len(freqs)):
            # Extract the complete feature vector for each lightpath
            lp_feat_vector = sample_data[:, link_index, freq_index]

            # Check if the lightpath is the LUT (OSNR, SNR, BER all -1)
            is_lut = int(lp_feat_vector[osnr_index] == -1 and lp_feat_vector[snr_index] == -1 and lp_feat_vector[ber_index] == -1)

            # If there is an active lightpath
            if np.any(lp_feat_vector > 0):
                lp_node_id = f"lightpath_{link_index}_{freq_index}"  # Unique ID for each lightpath node
                G.add_node(lp_node_id, type='lightpath', features=lp_feat_vector, is_lut=is_lut)

                # Add the link node if not already added
                link_node_id = f"link_{link_index}"
                if not G.has_node(link_node_id):
                    link_feat_vector = ntd_values[link_index]
                    G.add_node(link_node_id, type='link', features=link_feat_vector)

                # Add edge between lightpath and link
                G.add_edge(lp_node_id, link_node_id)

                # Track this lightpath for later direct connections
                link_lightpaths[link_index].append((lp_node_id, freq_index))

    # Step 2: Add Direct Connections Between Lightpaths
    for link_index, lightpaths in link_lightpaths.items():
        # Compare every pair of lightpaths in this link
        for i in range(len(lightpaths)):
            for j in range(i + 1, len(lightpaths)):
                lp_node_id1, freq1 = lightpaths[i]
                lp_node_id2, freq2 = lightpaths[j]

                # Calculate the spectral distance between the two lightpaths
                dist = spectral_distance(freqs[freq1], freqs[freq2])

                # Add an edge if the spectral distance is below the threshold
                if dist < freq_threshold:
                    G.add_edge(
                        lp_node_id1,
                        lp_node_id2,
                        shared_link=link_index,
                        spectral_distance=dist
                    )

    return G

# Step 3: Create NetworkX graphs for the first 20 samples
graphs_nx = []
for i in range(20):
    graph_nx = create_graph_nx(i)
    graphs_nx.append(graph_nx)

# Now you can visualize or inspect the NetworkX graphs
print(graphs_nx[0].nodes(data=True))  # Show nodes with attributes
print(graphs_nx[0].edges(data=True))  # Show edges with attributes
