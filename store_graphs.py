import to_graph as to_graph
import xarray as xr
import os
import pickle
import argparse
import networkx as nx

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Store networkx graphs in a directory."
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="preview",
        help="Type of dataset to use (preview or full).",
    )
    parser.add_argument(
        "--representation",
        type=str,
        default="lightpath",
        help="Type of graph representation to use (lightpath or topological).",
    )
    parser.add_argument(
        "--storage_type",
        type=str,
        default="pickle",
        help="Type of storage to use (pickle or gexf).",
    )
    args = parser.parse_args()

    dataset_type = args.dataset_type
    representation = args.representation
    storage_type = args.storage_type

    directory_for_graphs = (
        "networkx_graphs_lightpath"
        if representation == "lightpath"
        else "networkx_graphs_topological"
    )

    dataset_path = (
        "network_status_dataset_preview.nc"
        if dataset_type == "preview"
        else "../antel-repo/hhi/datasets/network_status_dataset.nc"
    )
    dataset = xr.open_dataset(dataset_path)
    samples = dataset["sample"].values  # Identifiers for samples
    dataset.close()

    features_to_consider = [
        "mod_order",
        "path_len",
        "num_spans",
        "freq",
    ]

    # Create the directory for the graphs and remove the previous graphs
    if os.path.exists(directory_for_graphs):
        for file in os.listdir(directory_for_graphs):
            os.remove(f"{directory_for_graphs}/{file}")
    os.makedirs(directory_for_graphs, exist_ok=True)

    for i in range(len(samples)):
        G = (
            to_graph.create_lightpath_graph(i, features_to_consider, dataset_path)
            if representation == "lightpath"
            else to_graph.create_topological_graph(
                i, features_to_consider, dataset_path
            )
        )

        if storage_type == "pickle":
            filepath = os.path.join(directory_for_graphs, f"graph_{i}.gpickle")
            with open(filepath, "wb") as f:
                pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
        elif storage_type == "gexf":
            filepath = os.path.join(directory_for_graphs, f"graph_{i}.gexf")
            nx.write_gexf(G, filepath)

    print(f"Graphs stored in {directory_for_graphs}")
