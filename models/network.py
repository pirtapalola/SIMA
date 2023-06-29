import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx

PATH = 'C:/Users/pirtapalola/Documents/Applications/EconomicNetworks/ss_project/final_project/'
# Read the csv file
stations = pd.read_csv('C:/Users/pirtapalola/Documents/Applications/EconomicNetworks/ss_project/final_project/comunas.csv',
                       usecols=['Station', 'Longitude', 'Latitude'])
connections = pd.read_csv('C:/Users/pirtapalola/Documents/Applications/EconomicNetworks/ss_project/final_project/comunas_commutes.csv')

'''
# Create point geometries
#stations_geometry = gpd.points_from_xy(stations['Longitude'],
 #                                  stations['Latitude'])

# Create a pandas geodataframe
#site_gdf = gpd.GeoDataFrame(stations, geometry=stations_geometry)
#print(site_gdf)

# Create an empty graph
graph = nx.Graph()
#graph.add_node('Itagui-Comuna 03', pos=(-75.63346, 6.16956))

def add_nodes_from_coordinates(list1):
    graph.add_node(list1[0], pos=(list1[1], list1[2]))

list_station_inputs = []
no_stations = [i for i in range(0, 66)]

for i in no_stations:
    list_station_inputs.append(((stations.iloc[i])))

for n in list_station_inputs:
    add_nodes_from_coordinates(n)

#station_indices = [i for i in no_stations+1]

#graph = nx.from_pandas_edgelist(connections, '1', '1.1')

edges = pd.DataFrame(
    {"source": connections['source'],
     "target": connections['target'],
     "weight": connections['weight']})
graph = nx.from_pandas_edgelist(edges, edge_attr=True)'''


metro_data = pd.read_csv('C:/Users/pirtapalola/Documents/Applications/EconomicNetworks/ss_project/final_project/medellin_metro.csv', encoding='cp1252')
# creating a new network for each line
lines = metro_data['line'].unique()
network_dict = {}
for line in lines:
    G = nx.Graph()
    metro_data_line = metro_data[metro_data['line'] == line]
    for index in range(len(metro_data_line)):
        current_row = metro_data_line.iloc[index]
        node_name = current_row['stop']
        G.add_node(node_name)
        if index < len(metro_data_line)-1:
            next_row = metro_data_line.iloc[index + 1]
            G.add_edge(current_row['stop'], next_row['stop'])
            network_dict[line] = G
# now we need to crawl through to find connections to another line, and match up which lines connect
main_network = nx.Graph()
for line, network in network_dict.items():
    for node in network.nodes():
        main_network.add_node(node)
        node_edges_in_line = network.edges(node)
        for edge in node_edges_in_line:
            main_network.add_edge(*edge, attribute=line)
#now adding metadata to the network nodes in a hopefully sensible way
for node in main_network.nodes():
    station_info = metro_data[metro_data['stop'] == node]
    main_network.nodes[node]['line'] = station_info['line'].tolist()
    main_network.nodes[node]['comuna_name'] = station_info['comuna_name'].tolist()
    main_network.nodes[node]['macrozona'] = station_info['macrozona'].to_list()
    main_network.nodes[node]['municipality'] = station_info['municipality'].to_list()
    main_network.nodes[node]['muni_index'] = station_info['muni_index'].to_list()
    main_network.nodes[node]['Lattitude'] = station_info['Lattitude'].to_list()
    main_network.nodes[node]['Longitude'] = station_info['Longitude'].to_list()
    main_network.nodes[node]['type'] = station_info['type'].to_list()

# Plot the shapefile
comunas_boundaries = gpd.GeoDataFrame.from_file(PATH + 'comunas_boundaries/MACROZONAS.shp')
ax = comunas_boundaries.plot(linewidth=1, edgecolor='grey', facecolor='lightblue')
nx.draw(G, nx.get_node_attributes(G, 'pos'), with_labels=False, node_size=5, node_color='r')
comunas_boundaries.plot()
plt.show()
