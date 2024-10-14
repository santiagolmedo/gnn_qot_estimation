import networkx as nx
import xarray as xr
import numpy as np

# Cargar el dataset
dataset = xr.open_dataset("network_status_dataset_preview_20.nc")

# Extraer variables relevantes
lp_feat_values = dataset['lp_feat'].values  # Lista de nombres de características de los lightpaths
data_values = dataset['data'].values  # [sample, lp_feat, link, freq]
samples = dataset['sample'].values  # Identificadores de muestras
metrics = dataset['metric'].values  # Nombres de las métricas
target_values = dataset['target'].values  # [sample, metric]

# Índices de las características que nos interesan
conn_id_index = list(lp_feat_values).index('conn_id')
mod_order_index = list(lp_feat_values).index('mod_order')
path_len_index = list(lp_feat_values).index('path_len')
num_spans_index = list(lp_feat_values).index('num_spans')
freq_index = list(lp_feat_values).index('freq')
lp_linerate_index = list(lp_feat_values).index('lp_linerate')
osnr_index = list(lp_feat_values).index('osnr')
snr_index = list(lp_feat_values).index('snr')
ber_index = list(lp_feat_values).index('ber')
src_id_index = list(lp_feat_values).index('src_id')
dst_id_index = list(lp_feat_values).index('dst_id')

# Función para crear el grafo para una muestra
def create_graph(sample_index):
    sample_data = data_values[sample_index]  # [lp_feat, link, freq]
    G = nx.Graph()

    # Añadir nodos de la red sin características (IDs de 1 a 75)
    num_nodes = 75  # Según el rango de src_id y dst_id
    node_ids = range(1, num_nodes + 1)
    G.add_nodes_from(node_ids)

    # Diccionario para almacenar los lightpaths ya procesados (para evitar duplicados)
    processed_lightpaths = set()

    # Recorrer todos los enlaces y frecuencias para identificar los lightpaths
    num_links = sample_data.shape[1]
    num_freqs = sample_data.shape[2]

    for link_index in range(num_links):
        for freq_index_in_loop in range(num_freqs):
            lp_feat_vector = sample_data[:, link_index, freq_index_in_loop]

            # Verificar si el canal está ocupado
            if not np.all(lp_feat_vector == 0):
                conn_id = int(lp_feat_vector[conn_id_index])

                # Ignorar si conn_id es cero
                if conn_id == 0:
                    continue

                # Evitar procesar el mismo lightpath varias veces
                if conn_id in processed_lightpaths:
                    continue

                processed_lightpaths.add(conn_id)

                src_id = int(lp_feat_vector[src_id_index])
                dst_id = int(lp_feat_vector[dst_id_index])

                # Características de la arista
                edge_features = {
                    'mod_order': lp_feat_vector[mod_order_index],
                    'path_len': lp_feat_vector[path_len_index],
                    'num_spans': lp_feat_vector[num_spans_index],
                    'freq': lp_feat_vector[freq_index],
                    'lp_linerate': lp_feat_vector[lp_linerate_index],
                    'osnr': lp_feat_vector[osnr_index],
                    'snr': lp_feat_vector[snr_index],
                    'ber': lp_feat_vector[ber_index],
                    'conn_id': conn_id
                }

                # Añadir arista entre src_id y dst_id
                G.add_edge(src_id, dst_id, **edge_features)

    # Extraer las etiquetas para la muestra actual
    # Las etiquetas están en target_values[sample_index], con las métricas en 'metric'
    metric_names = dataset['metric'].values
    target_vector = target_values[sample_index]

    # Crear un diccionario con las etiquetas
    labels = dict(zip(metric_names, target_vector))

    # Asignar las etiquetas al grafo como atributos
    G.graph['labels'] = labels

    return G

# Crear una lista de grafos para todas las muestras
graphs = []
for i in range(len(samples)):
    G = create_graph(i)
    graphs.append(G)

for i, graph in enumerate(graphs):
    nx.write_gexf(graph, f"graph_{i}.gexf")