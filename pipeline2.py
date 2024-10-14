import networkx as nx
import xarray as xr
import numpy as np

# Cargar el conjunto de datos
dataset = xr.open_dataset("network_status_dataset_preview_20.nc")

# Extraer variables relevantes
lp_feat_values = dataset['lp_feat'].values
ntd_values = dataset['ntd'].values
data_values = dataset['data'].values
links = dataset['link'].values
freqs = dataset['freq'].values
nt_feat_values = dataset['nt_feat'].values

# Índices de las características
conn_id_index = list(lp_feat_values).index('conn_id')
osnr_index = list(lp_feat_values).index('osnr')
snr_index = list(lp_feat_values).index('snr')
ber_index = list(lp_feat_values).index('ber')

# Función para calcular la distancia espectral
def spectral_distance(freq1, freq2):
    return abs(freq1 - freq2)

# Función para crear el grafo para una muestra
def create_graph_nx(sample_index, freq_threshold=0.05):
    sample_data = data_values[sample_index]  # [lp_feat, link, freq]

    # Inicializar el grafo
    G = nx.Graph()

    # Crear nodos de enlaces con sus características
    for link_index in range(len(links)):
        link_node_id = f"link_{link_index}"
        link_feat_vector = ntd_values[link_index]
        G.add_node(link_node_id, type='link', features=link_feat_vector)

    # Diccionario para almacenar información de los lightpaths
    lightpaths = {}

    # Recorrer todos los enlaces y frecuencias para identificar los lightpaths
    for link_index in range(len(links)):
        for freq_index in range(len(freqs)):
            # Extraer el vector de características
            lp_feat_vector = sample_data[:, link_index, freq_index]

            # Verificar si el canal está ocupado (no es todo cero)
            if not np.all(lp_feat_vector == 0):
                conn_id = lp_feat_vector[conn_id_index]

                conn_id = int(conn_id)

                # Añadir el lightpath al diccionario si no está
                if conn_id not in lightpaths:
                    # Determinar si es el LUT
                    is_lut = int(lp_feat_vector[osnr_index] == -1 and lp_feat_vector[snr_index] == -1 and lp_feat_vector[ber_index] == -1)
                    lightpaths[conn_id] = {
                        'features': lp_feat_vector,
                        'links': set(),
                        'freqs': set(),
                        'is_lut': is_lut
                    }

                # Añadir el enlace y frecuencia utilizados
                lightpaths[conn_id]['links'].add(link_index)
                lightpaths[conn_id]['freqs'].add(freqs[freq_index])

    # Crear nodos de lightpaths y aristas con enlaces
    for conn_id, lp_info in lightpaths.items():
        lp_node_id = f"lightpath_{conn_id}"
        G.add_node(lp_node_id, type='lightpath', features=lp_info['features'], is_lut=lp_info['is_lut'])

        # Conectar el lightpath con los enlaces que utiliza
        for link_index in lp_info['links']:
            link_node_id = f"link_{link_index}"
            G.add_edge(lp_node_id, link_node_id)

    # Añadir aristas entre lightpaths según los criterios
    lp_ids = list(lightpaths.keys())
    for i in range(len(lp_ids)):
        for j in range(i + 1, len(lp_ids)):
            lp1 = lightpaths[lp_ids[i]]
            lp2 = lightpaths[lp_ids[j]]

            # Verificar si comparten enlaces
            shared_links = lp1['links'].intersection(lp2['links'])
            if shared_links:
                # Calcular distancia espectral media en los enlaces compartidos
                spectral_distances = []
                for freq1 in lp1['freqs']:
                    for freq2 in lp2['freqs']:
                        dist = spectral_distance(freq1, freq2)
                        spectral_distances.append(dist)
                mean_spectral_distance = np.mean(spectral_distances)

                # Si la distancia espectral media es menor que el umbral, conectar
                if mean_spectral_distance < freq_threshold:
                    lp_node_id1 = f"lightpath_{lp_ids[i]}"
                    lp_node_id2 = f"lightpath_{lp_ids[j]}"
                    G.add_edge(
                        lp_node_id1,
                        lp_node_id2,
                        shared_links=len(shared_links),
                        mean_spectral_distance=mean_spectral_distance
                    )

    return G

# Crear grafos para las primeras 20 muestras
graphs_nx = []
for i in range(2):
    graph_nx = create_graph_nx(i)
    graphs_nx.append(graph_nx)
    nx.write_gexf(graph_nx, f"graph_{i}.gexf")

# Inspeccionar el primer grafo
print(graphs_nx[0].nodes(data=True))
print(graphs_nx[0].edges(data=True))

# def draw_instance(G):
#     # Crear una posición para cada nodo
#     pos = nx.spring_layout(G, seed=42)  # Puedes cambiar el layout si prefieres

#     # Listas para nodos y aristas según su tipo
#     lightpath_nodes = [n for n, attr in G.nodes(data=True) if attr['type'] == 'lightpath']
#     link_nodes = [n for n, attr in G.nodes(data=True) if attr['type'] == 'link']

#     # Colores y formas para los nodos
#     node_colors = []
#     node_shapes = []
#     for n in G.nodes():
#         attr = G.nodes[n]
#         if attr['type'] == 'lightpath':
#             if attr.get('is_lut', 0):
#                 node_colors.append('red')  # LUT en rojo
#             else:
#                 node_colors.append('orange')  # Otros lightpaths en naranja
#             node_shapes.append('o')  # Círculo para lightpaths
#         else:
#             node_colors.append('skyblue')  # Enlaces en azul claro
#             node_shapes.append('s')  # Cuadrado para enlaces

#     # Dibujar nodos y aristas
#     plt.figure(figsize=(12, 8))
#     ax = plt.gca()

#     # Dibujar enlaces entre lightpaths y enlaces
#     edge_colors = []
#     for u, v, attr in G.edges(data=True):
#         if G.nodes[u]['type'] == 'lightpath' and G.nodes[v]['type'] == 'link' or \
#            G.nodes[v]['type'] == 'lightpath' and G.nodes[u]['type'] == 'link':
#             edge_colors.append('gray')
#         else:
#             edge_colors.append('green')  # Aristas entre lightpaths
#     nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors)

#     # Dibujar nodos con sus formas y colores
#     for shape in set(node_shapes):
#         node_list = [n for n, s in zip(G.nodes(), node_shapes) if s == shape]
#         color_list = [c for n, c, s in zip(G.nodes(), node_colors, node_shapes) if s == shape]
#         nx.draw_networkx_nodes(G, pos, nodelist=node_list, node_color=color_list, node_shape=shape, node_size=300, ax=ax)

#     # Añadir etiquetas a los nodos
#     labels = {}
#     for n in G.nodes():
#         attr = G.nodes[n]
#         if attr['type'] == 'lightpath':
#             labels[n] = f"LP {n.split('_')[1]}"
#         else:
#             labels[n] = f"L {n.split('_')[1]}"
#     nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)

#     # Crear una leyenda personalizada
#     from matplotlib.lines import Line2D
#     legend_elements = [
#         Line2D([0], [0], marker='o', color='w', label='Lightpath',
#                markerfacecolor='orange', markersize=10),
#         Line2D([0], [0], marker='o', color='w', label='LUT',
#                markerfacecolor='red', markersize=10),
#         Line2D([0], [0], marker='s', color='w', label='Enlace',
#                markerfacecolor='skyblue', markersize=10),
#         Line2D([0], [0], color='gray', lw=2, label='Conexión LP-Enlace'),
#         Line2D([0], [0], color='green', lw=2, label='Interacción entre LPs')
#     ]
#     plt.legend(handles=legend_elements, loc='best')

#     plt.title("Grafo de la Red para una Instancia")
#     plt.axis('off')
#     plt.show()

# # Dibujar el primer grafo
# draw_instance(graphs_nx[0])


