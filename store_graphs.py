import to_graph as to_graph
import networkx as nx
import xarray as xr
import os
import pickle

DIRECTORY_FOR_GRAPHS = "networkx_graphs"

if __name__ == "__main__":
    dataset_path = "network_status_dataset_preview.nc"
    dataset = xr.open_dataset(dataset_path)
    samples = dataset["sample"].values  # Identifiers for samples

    features_to_consider = [
        "mod_order",
        "path_len",
        "num_spans",
        "freq",
        "lp_linerate",
        "osnr",
        "snr",
        "ber",
        "conn_id",
    ]

    graphs = []
    for i in range(len(samples)):
        G = to_graph.create_baseline_graph(i, features_to_consider, dataset_path)
        graphs.append(G)

    # Create the directory for the graphs and remove the previous graphs
    if os.path.exists(DIRECTORY_FOR_GRAPHS):
        for file in os.listdir(DIRECTORY_FOR_GRAPHS):
            os.remove(f"{DIRECTORY_FOR_GRAPHS}/{file}")
    os.makedirs(DIRECTORY_FOR_GRAPHS, exist_ok=True)

    for i, graph in enumerate(graphs):
        filepath = os.path.join(DIRECTORY_FOR_GRAPHS, f"graph_{i}.gpickle")
        with open(filepath, 'wb') as f:
            pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Graphs stored in {DIRECTORY_FOR_GRAPHS}")
