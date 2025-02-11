import copy
import json
import os
from datetime import timedelta
from warnings import warn

import geojson
import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from geopy.distance import geodesic
from networkx.readwrite import json_graph
from shapely import wkt
from shapely.geometry import LineString, Point, box, mapping
from shapely.geometry.base import BaseGeometry


def generate_lattice_arrays(map_limits, grid_size):
    # Extract map limits
    north, south, east, west = map_limits['north'], map_limits['south'], map_limits['east'], map_limits['west']

    # Create a bounding box
    bounding_box = gpd.GeoSeries(
        [box(west, south, east, north)], crs="EPSG:4326"
    )

    # Reproject to a metric CRS
    bounding_box_metric = bounding_box.to_crs(epsg=3857)

    # Get bounds in metric CRS
    min_x, min_y, max_x, max_y = bounding_box_metric.total_bounds

    # Generate points in metric coordinates
    x_coords = np.arange(min_x, max_x, grid_size)
    y_coords = np.arange(min_y, max_y, grid_size)

    return x_coords, y_coords


def generate_lattice(map_limits, grid_size):
    """
    Generate a lattice of points within the given map limits.

    Parameters:
    - map_limits (dict): Dictionary containing north, south, east, west bounds.
    - grid_size (float): Grid spacing in meters.

    Returns:
    - points (list of tuples): List of (longitude, latitude) points in the lattice.
    """

    x_coords, y_coords = generate_lattice_arrays(map_limits, grid_size)

    xx, yy = np.meshgrid(x_coords, y_coords)

    # Create points in metric CRS
    points_metric = [Point(x, y) for x, y in zip(xx.ravel(), yy.ravel())]

    # Convert metric points back to geographic coordinates
    points_geographic = gpd.GeoSeries(
        points_metric, crs="EPSG:3857").to_crs(epsg=4326)

    # Return points as (longitude, latitude) tuples
    return [(point.x, point.y) for point in points_geographic]


def compute_length(linestring):
    """
    Compute the geodesic length of a LINESTRING geometry in kilometers.
    """
    coords = list(linestring.coords)
    return sum(geodesic(coords[i], coords[i + 1]).kilometers for i in range(len(coords) - 1))


def print_graph_info(G: nx.Graph, n=5):
    graph_type = "Directed" if isinstance(G, nx.DiGraph) else "Undirected" if isinstance(
        G, nx.Graph) else "MultiGraph" if isinstance(G, nx.MultiGraph) else "MultiDiGraph" if isinstance(G, nx.MultiDiGraph) else "Unknown"
    print(
        f'{graph_type} has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges')
    print("+----------------------+")
    print("Nodes")
    print("+----------------------+")
    for i, (node, data) in enumerate(G.nodes(data=True)):
        print(f"*** Node: {node} ***")
        print(f"  Data: {data}")
        if i >= n:
            break

    print("+----------------------+")
    print("Edges")
    print("+----------------------+")
    for i, (source, target, data) in enumerate(G.edges(data=True)):
        print(f"*** Edge: {source} -> {target} ***")
        print(f"  Data: {data}")
        if i >= n:
            break


def osmnx_to_geopandas(G):
    def sanitize_columns(df):
        """
        Sanitize GeoDataFrame columns to ensure all values are compatible with GeoJSON.
        Converts unsupported types (e.g., lists) to strings.
        """
        for col in df.columns:
            if col == "geometry":
                continue  # Skip geometry column
            df[col] = df[col].apply(
                lambda x: str(x) if isinstance(x, (list, dict)) else x
            )
        return df

    edge_data = []
    for u, v, data in G.edges(data=True):
        # Process geometry
        geometry = data.get('geometry')
        if geometry:
            if isinstance(geometry, str):
                geometry = wkt.loads(geometry)
        else:
            point_u = G.nodes[u].get('geometry')
            point_v = G.nodes[v].get('geometry')
            geometry = LineString(
                [point_u, point_v]) if point_u and point_v else None

        if isinstance(geometry, BaseGeometry) and geometry.is_valid:
            # Collect edge data
            edge_entry = data.copy()
            edge_entry['edge_id'] = f"{u}->{v}"
            edge_entry['geometry'] = geometry
            edge_data.append(edge_entry)

    # Create GeoDataFrame for edges
    edges_gdf = gpd.GeoDataFrame(
        edge_data, geometry='geometry', crs="EPSG:4326")
    edges_gdf = sanitize_columns(edges_gdf)

    node_data = []
    for node_id, data in G.nodes(data=True):
        # Collect node data
        node_entry = data.copy()
        node_entry['node_id'] = node_id
        node_data.append(node_entry)

    # Create GeoDataFrame for nodes
    nodes_gdf = gpd.GeoDataFrame(
        node_data, geometry='geometry', crs="EPSG:4326")
    nodes_gdf = sanitize_columns(nodes_gdf)

    return nodes_gdf, edges_gdf


def railroad_geojson_to_networkx(filename):
    """
    Converts a GeoJSON file into a NetworkX directed graph with special formatting
    for known fields while storing all data.

    Parameters:
        filename (str): Path to the GeoJSON file.

    Returns:
        nx.DiGraph: The resulting graph with all data from the GeoJSON stored.
    """
    # Read the GeoJSON file
    gdf = gpd.read_file(filename)

    # Initialize a directed graph
    graph = nx.DiGraph()

    # Process points (stations) with special formatting for "coordinates"
    points_gdf = gdf[gdf.geometry.type == "Point"]
    for _, row in points_gdf.iterrows():
        station_id = int(row["station_id"])
        waiting_time = row.get("waiting_time", 60)
        transfer_time = row.get("transfer_time", 120)
        coordinates = row.geometry.coords[0]
        # Store all attributes, overriding 'coordinates' with a formatted version
        node_data = row.to_dict()
        node_data["station_coordinates"] = coordinates
        node_data["station_id"] = station_id
        node_data["waiting_time"] = timedelta(seconds=waiting_time)
        node_data["transfer_time"] = timedelta(seconds=transfer_time)
        graph.add_node(station_id, **node_data)

    # Process LineStrings (connections) with special formatting for known fields
    lines_gdf = gdf[gdf.geometry.type == "LineString"]
    for _, row in lines_gdf.iterrows():
        if "track_from" not in row or "track_to" not in row:
            warn(f"Connection missing 'track_from' or 'track_to': {row}")
            continue

        start, end = int(row["track_from"]), int(row["track_to"])
        path = list(row.geometry.coords)
        length_km = row["length_km"]
        travel_time = row["travel_time"]
        oneway = row.get("oneway", True)
        if isinstance(oneway, str):
            oneway = oneway.lower() not in ['0', 'false', 'no', 'off', 'n']
        else:
            oneway = bool(oneway)
        # Store all attributes, overriding specific fields with formatted versions
        edge_data = row.to_dict()
        edge_data["path"] = path
        edge_data["length_km"] = length_km
        edge_data["travel_time"] = timedelta(seconds=travel_time)
        edge_data['track_from'] = start
        edge_data['track_to'] = end
        edge_data['oneway'] = oneway
        graph.add_edge(start, end, **edge_data)
        if not oneway:
            inverse_edge_data = copy.deepcopy(edge_data)
            inverse_edge_data['track_from'] = end
            inverse_edge_data['track_to'] = start
            inverse_edge_data['path'] = list(reversed(path))
            inverse_edge_data['geometry'] = LineString(
                list(inverse_edge_data['geometry'].coords)[::-1])

            graph.add_edge(end, start, **inverse_edge_data)

    return graph


def find_closest_point_id(x, y, x_coords, y_coords):
    """
    Finds the ID of the closest point in the grid for a given x and y coordinate,
    without requiring a DataFrame.

    Parameters:
    - x: float, the x-coordinate (longitude)
    - y: float, the y-coordinate (latitude)
    - x_coords: sorted array of unique longitudes
    - y_coords: sorted array of unique latitudes

    Returns:
    - closest_id: int, the ID of the closest point in the flattened meshgrid
    """
    # Find the indices of the closest x and y coordinates
    x_idx = np.argmin(np.abs(np.array(x_coords) - x))
    y_idx = np.argmin(np.abs(np.array(y_coords) - y))

    # Convert 2D grid indices to 1D index
    num_x = len(x_coords)
    closest_id = y_idx * num_x + x_idx

    return closest_id


def get_interest_matrixes_from_graph_with_scores(G):
    # Step 1: Extract node scores from the graph
    nodes = list(G.nodes)
    n = len(nodes)

    # Create an ID-to-index mapping for the adjacency matrices
    node_idx = {node: i for i, node in enumerate(nodes)}

    # Initialize matrices
    common_interest_matrix = np.zeros((n, n))
    tourist_matrix = np.zeros((n, n))

    # Extract the total score and tourism score for each node
    total_scores = {node: G.nodes[node].get(
        "total_normalized_score", 0) for node in nodes}
    tourism_scores = {node: G.nodes[node].get(
        "Tourism_score", 0) for node in nodes}

    # Step 2: Fill matrices based on edges
    for u, v in G.edges:
        i, j = node_idx[u], node_idx[v]

        # Common Interest Matrix: Depends only on the "to" node
        common_interest_matrix[i, j] = total_scores[v]

        # Tourist Matrix: Product of tourism scores of both nodes
        tourist_matrix[i, j] = tourism_scores[u] * tourism_scores[v]

    # Convert to Pandas DataFrame for readability
    common_interest_df = pd.DataFrame(
        common_interest_matrix, index=nodes, columns=nodes)
    tourist_df = pd.DataFrame(tourist_matrix, index=nodes, columns=nodes)

    # Fill the diagonal values of the matrices with zeros
    np.fill_diagonal(common_interest_matrix, 0)
    np.fill_diagonal(tourist_matrix, 0)

    return common_interest_df, tourist_df, common_interest_matrix, tourist_matrix
