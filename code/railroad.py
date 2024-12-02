from matplotlib.colors import Normalize
import itertools
import json
import os
import shutil
from collections import defaultdict
from copy import deepcopy
from datetime import date, datetime, timedelta

import contextily as ctx
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import cm
from shapely import wkt
from shapely.geometry import LineString, Point

from geopy.distance import geodesic


import matplotlib.pyplot as plt
import itertools
import networkx as nx
from matplotlib.cm import tab10
from shapely.geometry import LineString


class CustomJSONEncoder(json.JSONEncoder):
    def iterencode(self, obj, _one_shot=False):
        def convert(obj):
            if isinstance(obj, (datetime, date)):
                return {"__type__": "datetime", "value": obj.isoformat()}
            elif isinstance(obj, timedelta):
                return {"__type__": "timedelta", "value": obj.total_seconds()}
            elif isinstance(obj, np.ndarray):
                return {"__type__": "ndarray", "value": obj.tolist()}
            elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                return int(obj)
            elif isinstance(obj, LineString):
                return {"__type__": "LineString", "value": obj.wkt}
            elif isinstance(obj, Point):
                return {"__type__": "Point", "value": obj.wkt}
            elif isinstance(obj, tuple):
                return {"__type__": "tuple", "value": [convert(i) for i in obj]}
            elif isinstance(obj, dict):
                if any(not isinstance(k, (str, int, float, bool, type(None))) for k in obj.keys()):
                    # Dict with non-string keys
                    return {
                        "__type__": "tuple_dict",
                        "value": [(convert(k), convert(v)) for k, v in obj.items()]
                    }
                else:
                    return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(i) for i in obj]
            else:
                return obj  # Let the default method handle other types

        # Preprocess the entire object to convert non-serializable types
        obj = convert(obj)
        # Call the parent class's iterencode method
        return super(CustomJSONEncoder, self).iterencode(obj, _one_shot=_one_shot)


def custom_json_decoder(dct):
    if isinstance(dct, dict) and "__type__" in dct:
        if dct["__type__"] == "datetime":
            return datetime.fromisoformat(dct["value"])
        elif dct["__type__"] == "timedelta":
            return timedelta(seconds=dct["value"])
        elif dct["__type__"] == "ndarray":
            return np.array(dct["value"])
        elif dct["__type__"] == "LineString":
            return wkt.loads(dct["value"])
        elif dct["__type__"] == "Point":
            return wkt.loads(dct["value"])
        elif dct["__type__"] == "tuple_dict":
            return {custom_json_decoder(k): custom_json_decoder(v) for k, v in dct["value"]}
        elif dct["__type__"] == "tuple":
            return tuple(custom_json_decoder(i) for i in dct["value"])
    return dct  # Return as is if no special type


def save_graph(graph, filename):
    """
    Save a NetworkX graph to a JSON file, encoding custom data types.

    Args:
        graph (networkx.Graph): The graph to save.
        filename (str): The JSON file to save to.
    """
    data = nx.node_link_data(graph)  # Convert graph to node-link format
    with open(filename, "w") as f:
        json.dump(data, f, cls=CustomJSONEncoder)


def load_graph(filename=None, data=None):
    """
    Load a NetworkX graph from a JSON file, decoding custom data types.

    Args:
        filename (str): The JSON file to load from.

    Returns:
        networkx.Graph: The reconstructed graph.
    """
    if data is None:
        with open(filename, "r") as f:
            data = json.load(f, object_hook=custom_json_decoder)
    else:
        data = json.loads(json.dumps(data), object_hook=custom_json_decoder)
    return nx.node_link_graph(data)  # Convert back to a graph


def create_subgraph(G, xlim=None, ylim=None):
    """
    Creates a subgraph containing only the nodes within the specified x and y limits.
    """

    # Filter nodes based on xlim and ylim
    filtered_nodes = [
        node for node, data in G.nodes(data=True)
        if (xlim is None or (xlim[0] <= data['geometry'].x <= xlim[1])) and
           (ylim is None or (ylim[0] <= data['geometry'].y <= ylim[1]))
    ]

    # Create a subgraph containing only the filtered nodes
    subgraph = G.subgraph(filtered_nodes).copy()

    return subgraph


def plot_node_graph(G, xlim=None, ylim=None, show_node_ids=False, draw_edges=True,
                    show_nodes_labels=False, cluster_attribute=None, node_size=50,
                    road_color='black', add_map=False, show=True, fig=None, ax=None,
                    color_segments=False, cmap='viridis'):
    """
    Plots the graph, focusing on a specific area by first creating a subgraph.
    Each cluster is colored differently. Adds a mode to color each segment of
    edge geometries in a gradient.

    Parameters:
    - G: The input graph.
    - xlim: (min, max) tuple for x-axis limits.
    - ylim: (min, max) tuple for y-axis limits.
    - show_node_ids: Boolean, whether to display node IDs.
    - cluster_attribute: Node attribute that indicates the cluster (e.g., 'cluster').
    - node_size: Size of nodes in the plot.
    - add_map: Boolean, whether to add a basemap (requires xlim and ylim).
    - show: Boolean, whether to display the plot.
    - fig: Matplotlib figure object (optional).
    - ax: Matplotlib axis object (optional).
    - color_segments: Boolean, whether to color edge segments in a gradient.
    - cmap: Colormap to use for segment coloring.
    """
    # Step 1: Create the subgraph
    subgraph = create_subgraph(G, xlim=xlim, ylim=ylim)

    # Step 2: Extract node positions and cluster information
    pos = {node: (data['geometry'].x, data['geometry'].y)
           for node, data in subgraph.nodes(data=True)}

    if cluster_attribute:
        clusters = {data[cluster_attribute]
                    for _, data in subgraph.nodes(data=True)}
    else:
        clusters = {None}

    # Assign colors for each cluster
    cluster_colors = itertools.cycle(cm.tab10.colors)  # Use tab10 color map
    cluster_color_map = {cluster: color for cluster,
                         color in zip(clusters, cluster_colors)}

    # Step 3: Create figure and axis if not provided
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))

    # Draw nodes by cluster
    for cluster, color in cluster_color_map.items():
        cluster_nodes = [node for node, data in subgraph.nodes(data=True)
                         if data.get(cluster_attribute) == cluster]
        nx.draw_networkx_nodes(
            subgraph, pos, nodelist=cluster_nodes, node_color=[
                color] * len(cluster_nodes),
            node_size=node_size, label=f"Cluster {cluster}" if cluster else "Unclustered", ax=ax
        )

    # Draw edges with segment gradient coloring
    if draw_edges:
        colormap = cm.get_cmap(cmap)
        norm = Normalize(vmin=0, vmax=1)  # Normalize segment progression

        for source, target, data in subgraph.edges(data=True):
            if 'geometry' not in data:
                print(f"Edge {source} -> {target} does not have geometry.")
                continue
            geometry = data['geometry']
            if not isinstance(geometry, LineString):
                print(
                    f"Geometry of edge {source} -> {target} is not a LineString.")
                continue

            x, y = geometry.xy
            if len(x) < 2:
                print(
                    f"Skipping edge {source} -> {target} with insufficient points.")
                continue

            if color_segments:
                # Draw each segment in a gradient
                num_segments = len(x) - 1
                for i in range(num_segments):
                    segment_x = [x[i], x[i + 1]]
                    segment_y = [y[i], y[i + 1]]
                    # Color based on segment's relative position along the edge
                    color = colormap(
                        norm(i / (num_segments - 1) if num_segments > 1 else 0))
                    ax.plot(segment_x, segment_y, color=color, alpha=0.8)
            else:
                # Plot the entire geometry as a single line
                ax.plot(x, y, color=road_color, alpha=0.5)

    # Show node IDs if the parameter is True
    if show_node_ids:
        for node, (x, y) in pos.items():
            ax.text(x, y, str(node), fontsize=8, ha='center', color='black')

    # Add basemap if requested
    if add_map:
        if xlim and ylim:
            ctx.add_basemap(ax,
                            crs="EPSG:4326",  # Assumes the geometry is in WGS84
                            source=ctx.providers.CartoDB.PositronNoLabels)
        else:
            raise ValueError(
                "To add a map, you must specify both xlim and ylim.")

    # Set x and y limits explicitly if specified
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    # Add legend for clusters
    ax.legend(loc='best', fontsize=10)

    # Force the display of gridlines, ticks, and tick labels
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xticks(ax.get_xticks())  # Force x-axis ticks
    ax.set_yticks(ax.get_yticks())  # Force y-axis ticks
    ax.tick_params(axis='both', which='both', direction='in', length=5)

    # Set axis labels
    ax.set_xlabel("Longitude")  # Customize as needed
    ax.set_ylabel("Latitude")   # Customize as needed

    # Explicitly enable tick labels
    ax.tick_params(labelbottom=True, labelleft=True)

    ax.axis('on')  # Ensure axis is visible
    if show:
        plt.show()

    return fig, ax


def test_road_graph(G: nx.Graph):
    for from_id, to_id, data in G.edges(data=True):
        line = data['geometry']
        from_point = G.nodes[from_id]['geometry']
        to_point = G.nodes[to_id]['geometry']
        if line.coords[0] == (from_point.x, from_point.y):
            assert line.coords[-1] == (
                to_point.x, to_point.y), f'first point match, but the last point does not match {line.coords[-1]} != {(to_point.x, to_point.y)}, distance = {geodesic((line.coords[0][1], line.coords[0][0]), (to_point.x, to_point.y)).meters}m'
        elif line.coords[-1] == (from_point.x, from_point.y):
            assert line.coords[0] == (
                to_point.x, to_point.y), f'last point match, but the first point does not match {line.coords[0]} != {(to_point.x, to_point.y)}, distance = {geodesic((line.coords[0][1], line.coords[0][0]), (to_point.x, to_point.y)).meters}m'
        else:
            raise Exception(
                f'{line.coords[0] = }\n{line.coords[-1] = }\n{(from_point.x, from_point.y) = }\n{(to_point.x, to_point.y) = }')


class Train:
    def __init__(self, schedule: list[tuple[tuple[datetime, datetime], int]], capacity: int, number_of_stations: int, style: dict | None = None, route_name: str = ''):
        """
        Initializes a Train object with its schedule, capacity, and passengers.

        Parameters:
        - schedule (list[tuple[tuple[datetime, datetime], int]]): A list of tuples ( (arrival_time, departure_time), station_id).
        - capacity (int): Maximum passenger capacity of the train.
        - number_of_stations (int): Total number of stations in the system.
        """
        # Extract arrival and departure times
        self.arrival_times = [s[0][0] for s in schedule]
        self.departure_times = [s[0][1] for s in schedule]
        # Ensure that departure time is less than or equal to the next arrival time
        for arrival, departure in zip(self.arrival_times[1:], self.departure_times[:-1]):
            assert departure <= arrival, f'Next arrival time must be greater than previous departure time: {departure} <= {arrival}'
        # Extract the route (list of station IDs)
        self.route = [s[1] for s in schedule]
        self.schedule = sorted(schedule, key=lambda x: x[0])

        self.capacity = capacity
        self.passengers_on_board = np.zeros(number_of_stations, dtype=int)
        self.number_of_stations = number_of_stations
        self.route_name = route_name
        self.style = style or {}

        # Initialize event list for the train
        self.events = []
        for idx, ((arrival_time, departure_time), station_id) in enumerate(self.schedule):
            self.events.append(
                {'time': arrival_time, 'event_type': 'arrival', 'station_index': idx})
            self.events.append(
                {'time': departure_time, 'event_type': 'departure', 'station_index': idx})
        self.events.sort(key=lambda x: x['time'])

        self.current_event_index = 0  # To keep track of the next event
        self.at_station = False
        # Index of the current station in the route (None before first arrival)
        self.previous_station_index = None
        self.current_station_index = None
        self.next_station_index = 0

        # Flag to indicate if passengers have disembarked
        self.passengers_disembarked = False

    def get_pandas_schedule(self):
        """
        Returns the train schedule as a pandas DataFrame.

        Columns:
        - 'departure_station': Station ID of the departure station.
        - 'arrival_station': Station ID of the arrival station.
        - 'departure_time': Departure time from the station.
        - 'arrival_time': Arrival time at the station.
        """
        data = {
            # Exclude the last station for departure
            "departure_station": self.route[:-1],
            # Exclude the first station for arrival
            "arrival_station": self.route[1:],
            # Exclude the last time
            "departure_time": self.departure_times[:-1],
            # Exclude the first time
            "arrival_time": self.arrival_times[1:],
        }
        return pd.DataFrame(data)

    @property
    def status(self):
        if self.route_finished:
            return "Finished"
        elif self.at_station:
            return f"At station {self.current_station}"
        elif self.previous_station_index is None:
            return "Not started"
        else:
            return f"Between stations {self.previous_station} and {self.next_station}"

    def get_event_queue(self):
        return [event['time'] for event in self.events]

    @property
    def previous_station(self) -> int:
        """
        Returns the previous station ID.
        """
        if self.previous_station_index is not None:
            return self.route[self.previous_station_index]
        else:
            return None

    @property
    def current_station(self) -> int:
        """
        Returns the current station ID.
        """
        if self.at_station and self.current_station_index is not None and self.current_station_index < len(self.route):
            return self.route[self.current_station_index]
        else:
            return None

    @property
    def next_station(self) -> int:
        """
        Returns the next station ID.
        """
        next_index = self.next_station_index
        if next_index is not None:
            return self.route[next_index]
        else:
            return None

    @property
    def next_arrival_time(self) -> datetime:
        """
        Returns the arrival time at the next station.
        """
        for event in self.events[self.current_event_index:]:
            if event['event_type'] == 'arrival':
                return event['time']
        return None

    @property
    def next_departure_time(self) -> datetime:
        """
        Returns the departure time from the next station.
        """
        for event in self.events[self.current_event_index:]:
            if event['event_type'] == 'departure':
                return event['time']
        return None

    @property
    def previous_departure_time(self) -> datetime:
        """
        Returns the departure time from the previous station.
        """
        idx = self.current_event_index - 1
        while idx >= 0:
            event = self.events[idx]
            if event['event_type'] == 'departure':
                return event['time']
            idx -= 1
        return None

    @property
    def previous_arrival_time(self) -> datetime:
        """
        Returns the arrival time at the previous station.
        """
        idx = self.current_event_index - 1
        while idx >= 0:
            event = self.events[idx]
            if event['event_type'] == 'arrival':
                return event['time']
            idx -= 1
        return None

    @property
    def number_of_passengers(self) -> int:
        """
        Returns the total number of passengers on board.
        """
        return self.passengers_on_board.sum()

    @property
    def route_finished(self) -> bool:
        """
        Checks if the train has reached the end of its route.
        """
        return self.current_event_index >= len(self.events)

    @staticmethod
    def generate_schedule(
        G: nx.DiGraph,
        departure_time: datetime,
        stations: list[int]
    ) -> list[tuple[tuple[datetime, datetime], int]]:
        """
        Generates a train schedule based on the graph and route.

        Parameters:
        - G (nx.DiGraph): The railroad graph with travel times for edges and waiting times for nodes.
        - departure_time (datetime): The initial departure time for the train.
        - stations (list[int]): An ordered list of station IDs representing the train's route.

        Returns:
        - list[tuple[tuple[datetime, datetime], int]]: A list of tuples ((arrival_time, departure_time), station_id).
        """
        schedule = []
        current_time = departure_time

        for i in range(len(stations) - 1):
            station_id = stations[i]

            # Arrival time is the current time
            arrival_time = current_time

            # Add waiting time at the station for departure
            waiting_time = G.nodes[station_id].get(
                'waiting_time', timedelta(minutes=1))
            departure_time = arrival_time + waiting_time

            # Append (arrival_time, departure_time) and station_id to the schedule
            schedule.append(((arrival_time, departure_time), station_id))

            # Travel time to the next station
            travel_time = G[station_id][stations[i + 1]]['travel_time']
            current_time = departure_time + travel_time

        # Handle the last station
        last_station = stations[-1]
        arrival_time = current_time
        waiting_time = G.nodes[last_station].get(
            'waiting_time', timedelta(minutes=1))
        departure_time = arrival_time + waiting_time
        schedule.append(((arrival_time, departure_time), last_station))

        return schedule

    @staticmethod
    def generate_schedules(
        G: nx.DiGraph,
        initial_time: datetime,
        route: list[int],
        num_trains: int,
        simulation_length: timedelta
    ) -> list[list[tuple[tuple[datetime, datetime], int]]]:
        """
        Generates schedules for multiple trains on a cyclic route, starting at different positions
        along the route so that they are equally spaced.

        Parameters:
        - G (nx.DiGraph): The railroad graph with travel times for edges and waiting times for nodes.
        - initial_time (datetime): The initial time for the simulation.
        - route (List[int]): An ordered list of station IDs forming a cycle.
        - num_trains (int): The number of trains to schedule.
        - simulation_length (timedelta): The total length of the simulation.

        Returns:
        - List[List[Tuple[Tuple[datetime, datetime], int]]]: A list of schedules for each train.
        """

        # Compute cumulative times along the route
        cumulative_times = [timedelta(0)]
        for i in range(len(route)):
            station_id = route[i]
            waiting_time = G.nodes[station_id].get(
                'waiting_time', timedelta(minutes=1))
            next_station = route[(i + 1) % len(route)]
            travel_time = G[station_id][next_station]['travel_time']
            cumulative_times.append(
                cumulative_times[-1] + waiting_time + travel_time)
        # Total cycle time
        cycle_time = cumulative_times[-1]
        # Initialize list to hold schedules for all trains
        all_schedules = []

        # Simulation end time
        simulation_end_time = initial_time + simulation_length

        for i in range(num_trains):
            time_offset = i * cycle_time / num_trains
            # Find the starting point along the route where cumulative_time >= time_offset
            idx = 0
            while cumulative_times[idx] < time_offset:
                idx += 1
            # Adjust the starting station index
            station_idx = idx % len(route)
            # Compute the time difference to adjust arrival time at starting station
            time_diff = cumulative_times[idx] - time_offset

            # Generate schedule starting from station_idx
            current_time = initial_time - time_diff  # Adjust current_time backward
            train_schedule = []
            while current_time < simulation_end_time:
                for j in range(len(route)):
                    idx = (station_idx + j) % len(route)
                    station_id = route[idx]
                    # Arrival time at station
                    arrival_time = current_time
                    # Waiting time at station
                    waiting_time = G.nodes[station_id].get(
                        'waiting_time', timedelta(minutes=1))
                    # Departure time from station
                    departure_time = arrival_time + waiting_time
                    # Append to schedule
                    if arrival_time >= initial_time:
                        if arrival_time > simulation_end_time:
                            break
                        if departure_time > simulation_end_time:
                            departure_time = simulation_end_time
                        train_schedule.append(
                            ((arrival_time, departure_time), station_id))
                    # Travel time to next station
                    next_idx = (idx + 1) % len(route)
                    next_station = route[next_idx]
                    travel_time = G[station_id][next_station]['travel_time']
                    current_time = departure_time + travel_time
                    if current_time > simulation_end_time:
                        break
                else:
                    continue  # continue the while loop to start next loop
                break  # Break if the for loop was broken due to time constraints
            all_schedules.append(train_schedule)
        return all_schedules

    def disembark_passengers(self, strategy_matrix: np.ndarray) -> np.ndarray:
        """
        Disembarks passengers whose next destination (via strategy matrix) does not match the train's next station.

        Parameters:
        - strategy_matrix (np.ndarray): Matrix defining where passengers should go next.

        Returns:
        - np.ndarray: Array of passengers disembarking at the current station.
        """
        assert not self.passengers_disembarked, 'Passengers have already disembarked'

        # Create a boolean array indicating whether passengers stay on the train
        staying_on_train = np.array(
            [strategy_matrix[self.current_station, dest] == self.next_station
             for dest in range(self.number_of_stations)]
        )

        # Passengers leaving the train
        leaving_passengers = self.passengers_on_board * ~staying_on_train

        # Keep only the passengers who are staying on the train
        self.passengers_on_board *= staying_on_train

        return leaving_passengers

    def board_passengers(self, station_passengers: np.ndarray, strategy_matrix: np.ndarray) -> np.ndarray:
        """
        Boards passengers whose next destination matches the train's next station.

        Parameters:
        - station_passengers (np.ndarray): Array representing passengers waiting at the station.
        - strategy_matrix (np.ndarray): Matrix defining where passengers should go next.

        Returns:
        - np.ndarray: Array representing how many passengers of each type were taken on board.
        """
        assert self.current_station is not None, 'Train is not at a station'
        if self.route_finished:
            raise ValueError(
                "Train has reached the end of its route, yet the board_passengers method was called.")

        available_space = self.capacity - self.passengers_on_board.sum()

        passengers_next_step = strategy_matrix[self.current_station, :]
        passengers_will_board = passengers_next_step == self.next_station

        passengers_intending_to_board = station_passengers * passengers_will_board
        if passengers_intending_to_board.sum() <= available_space:
            # All passengers intending to board can do so
            self.passengers_on_board += passengers_intending_to_board
            return passengers_intending_to_board

        else:
            # Not all passengers can board due to space constraints
            passengers_taken = np.zeros_like(
                station_passengers)  # Initialize tracking array

            # Sort passengers by their destination to fill the train efficiently
            sorted_indices = np.argsort(passengers_intending_to_board)[::-1]
            for idx in sorted_indices:
                if passengers_intending_to_board[idx] > 0:
                    if passengers_intending_to_board[idx] <= available_space:
                        passengers_taken[idx] = passengers_intending_to_board[idx]
                        available_space -= passengers_intending_to_board[idx]
                    else:
                        passengers_taken[idx] = available_space
                        available_space = 0
                        break

            self.passengers_on_board += passengers_taken
            return passengers_taken

    def tick(self, railroad: 'Railroad') -> None:
        """
        Advances the train to the next event (arrival or departure) and processes it.

        Parameters:
        - railroad (Railroad): The railroad system containing strategies and passenger information.
        """
        if self.current_event_index >= len(self.events):
            raise ValueError(
                "Train has reached the end of its route, yet the tick method was called.")

        event = self.events[self.current_event_index]
        self.current_event_index += 1

        current_time = event['time']
        event_type = event['event_type']
        station_index = event['station_index']

        if event_type == 'arrival':
            # Handle arrival
            assert not self.at_station, 'Train is already at a station'
            self.at_station = True
            self.current_station_index = self.next_station_index
            self.next_station_index += 1
            self.previous_station_index = self.current_station_index
            # Move to the station index in the route
            self.current_station_index = station_index

            # Disembark passengers
            strategy_matrix = railroad.strategy.strategy_matrix
            leaving_passengers = self.disembark_passengers(strategy_matrix)
            railroad.add_passengers_to_buffer(
                leaving_passengers, self.current_station)
            self.passengers_disembarked = True

        elif event_type == 'departure':
            assert self.passengers_disembarked, 'Passengers have not disembarked yet'
            assert self.at_station, 'Train is not at a station'
            # Handle departure

            if self.route_finished:
                # Train has reached the end of its route
                self.current_station_index = None
                self.previous_station_index = None
                self.next_station_index = None
                self.at_station = False
                self.passengers_disembarked = False
            else:
                # Board passengers
                station_passengers = railroad.get_passengers_at_station(
                    self.current_station)
                strategy_matrix = railroad.strategy.strategy_matrix
                taken_passengers = self.board_passengers(
                    station_passengers, strategy_matrix)
                railroad.board_passengers_from_station(
                    taken_passengers, self.current_station)

                self.passengers_disembarked = False

                self.at_station = False
                self.current_station_index = None

        # Print train status after processing the event
        print(
            f"Train {id(self)} processed event '{event_type}' at {current_time}")

    def __str__(self):
        """
        Returns a string representation of the Train object, including its route,
        current and next stations, capacity, passengers onboard, and schedule.
        """
        onboard_passengers = ", ".join(
            f"to {dest}: {count}" for dest, count in enumerate(self.passengers_on_board) if count > 0
        )
        schedule_str = "\n".join(
            f"   Station {station}: {arrival_time} - {departure_time} ->" for ((arrival_time, departure_time), station) in self.schedule
        )
        remaining_stops = self.route[self.current_station_index +
                                     1:] if self.current_station_index is not None else self.route
        return (
            f"Train:\n"
            f'  Route Name: {self.route_name}\n'
            f"  Route: {self.route}\n"
            f"  Status: {self.status}\n"
            f"  Current Event Index: {self.current_event_index}\n"
            f"  Previous Station: {self.previous_station}\n"
            f"  Current Station: {self.current_station}\n"
            f"  Next Station: {self.next_station}\n"
            f"  Next Arrival Time: {self.next_arrival_time}\n"
            f"  Next Departure Time: {self.next_departure_time}\n"
            f"  Previous Departure Time: {self.previous_departure_time}\n"
            f"  Previous Arrival Time: {self.previous_arrival_time}\n"
            f"  Capacity: {self.capacity}\n"
            f"  Passengers On Board: {self.passengers_on_board.sum()} "
            f"({onboard_passengers if onboard_passengers else 'None'})\n"
            f"  Remaining Stops: {remaining_stops}\n"
            f"  Schedule:\n{schedule_str}"
        )

    def to_json(self) -> dict:
        """
        Serializes the Train object to a JSON-compatible dictionary.

        Returns:
            dict: A dictionary representation of the Train object.
        """
        return {
            "schedule": [
                {
                    "arrival_time": arrival_time,
                    "departure_time": departure_time,
                    "station_id": station_id
                }
                for ((arrival_time, departure_time), station_id) in self.schedule
            ],
            "capacity": self.capacity,
            "number_of_stations": self.number_of_stations,
            "route": self.route,
            "route_name": self.route_name,
            "status": self.status,
            "at_station": self.at_station,
            "number_of_passengers": self.number_of_passengers,
            "current_event_index": self.current_event_index,
            "current_station_index": self.current_station_index,
            "previous_station": self.previous_station,
            "current_station": self.current_station,
            "next_station": self.next_station,
            "next_arrival_time": self.next_arrival_time if self.next_arrival_time else None,
            "next_departure_time": self.next_departure_time if self.next_departure_time else None,
            "previous_departure_time": self.previous_departure_time if self.previous_departure_time else None,
            "previous_arrival_time": self.previous_arrival_time if self.previous_arrival_time else None,
            "passengers_on_board": self.passengers_on_board,
            "style": self.style,
        }


class Strategy:
    def __init__(self, graph: nx.DiGraph, station_timestamps: dict, number_of_stations: int):
        """
        Initialize the Strategy object with a pre-built time-expanded graph.

        Parameters:
        - graph: A directed graph (DiGraph) representing the time-expanded train schedule.
        - station_timestamps: Dictionary mapping station IDs to sorted tuples of timestamps.
        """
        self.num_stations = number_of_stations
        self.G = graph
        self.station_timestamps = station_timestamps
        self.strategy_matrix = np.zeros(
            (self.num_stations, self.num_stations), dtype=int)
        self.stop_moving()
        self.last_timestamp = None
        # print(f'{station_timestamps = }')

        self.compute_reachability()

    def stop_moving(self):
        for i in range(self.num_stations):
            self.strategy_matrix[i, :] = i

    @classmethod
    def from_schedule(cls, schedule: pd.DataFrame, number_of_stations: int):
        """
        Build a Strategy instance from a schedule.

        Parameters:
        - schedule: DataFrame containing train schedules with columns:
            'departure_station', 'arrival_station', 'departure_time', 'arrival_time'.

        Returns:
        - A Strategy instance containing the time-expanded graph and station timestamps.
        """
        G = nx.DiGraph()

        # Group trips by station for later processing
        # Use a set to avoid duplicate timestamps
        station_times = defaultdict(set)

        # Add train connections (travel edges)
        for _, row in schedule.iterrows():
            dep_station = row['departure_station']
            arr_station = row['arrival_station']
            dep_time = row['departure_time']
            arr_time = row['arrival_time']

            # Add travel edge
            G.add_edge((dep_station, dep_time), (arr_station, arr_time))

            # Record both departure and arrival times for each station
            station_times[dep_station].add(dep_time)
            station_times[arr_station].add(arr_time)

        # Add waiting time edges at each station
        for station, times in station_times.items():
            sorted_times = sorted(times)
            for i in range(len(sorted_times) - 1):
                if sorted_times[i] < sorted_times[i + 1]:  # Prevent self-loops
                    G.add_edge(
                        (station, sorted_times[i]),
                        (station, sorted_times[i + 1])
                    )

        # Prepare station timestamps dictionary (convert sets to sorted tuples)
        station_timestamps = {
            station: tuple(sorted(times)) for station, times in station_times.items()
        }

        return cls(G, station_timestamps, number_of_stations)

    def compute_reachability(self):
        """
        Computes and stores the earliest reachable timestamp and path to each station for every node in the graph.

        Updates:
        - Each node in the graph will have an attribute "reachability", a dictionary mapping target station IDs
        to the earliest reachable timestamp and the path to reach it.

        Returns:
        - reachability: Dictionary where keys are source nodes, and values are dictionaries mapping
                        target station IDs to the earliest reachable timestamp and path.
        """
        # Compute all shortest paths in the graph
        all_paths = nx.shortest_path(self.G)

        # Dictionary to store reachability for all nodes
        reachability = {}

        # Iterate through the source nodes and their reachable nodes
        for source_node, target_nodes in all_paths.items():
            # Initialize reachability for this source node
            reachability[source_node] = {}

            # Iterate over all reachable nodes from the source node
            for target_node, path in target_nodes.items():
                # Extract the station ID and timestamp of the target node
                target_station = target_node[0]
                target_time = target_node[1]
                pass
                # If this station is not yet in reachability or this timestamp is earlier, update it
                if (
                    target_station not in reachability[source_node]
                    or target_time < reachability[source_node][target_station]["reachable_time"]
                ):
                    reachability[source_node][target_station] = {
                        "reachable_time": target_time,
                        "path": path,
                        "next_instruction": path[1][0] if len(path) > 1 else None,
                    }

        # Add the reachability data as a node attribute in the graph
        nx.set_node_attributes(self.G, reachability, "reachability")

        return reachability

    def get_next_instructions(self, station_id, timestamp):
        """
        Computes the next instructions for all reachable stations from a given station at a specific timestamp.

        Parameters:
        - station_id: The ID of the station to start from.
        - timestamp: The current timestamp (datetime object) to find the departure time.

        Returns:
        - instructions: Dictionary where keys are reachable station IDs, and values are the next instructions
                        (next station ID or None if staying at the same station).
        """
        instructions = {}

        # Find the departure timestamp for the station
        timestamps = self.station_timestamps.get(station_id, [])
        departure_time = next(
            (time for time in timestamps if time >= timestamp), None)

        if departure_time is None:
            # If no valid departure time exists, return empty instructions
            return instructions

        # The node corresponding to the departure
        departure_node = (station_id, departure_time)

        # Retrieve the reachability data for this node
        reachability = self.G.nodes[departure_node].get("reachability", {})

        # Extract next instructions for each reachable station
        for target_station, data in reachability.items():
            instructions[target_station] = data["next_instruction"]

        return instructions

    def update_strategy(self, timestamp: datetime):
        if self.last_timestamp == timestamp:
            return

        self.strategy_matrix = np.zeros(
            (self.num_stations, self.num_stations), dtype=int)
        for i in range(self.num_stations):
            instructions = self.get_next_instructions(i, timestamp)
            instructions_array = np.array(
                [j if instructions.get(j, i) is None else instructions.get(j, i) for j in range(self.num_stations)], dtype=int)
            self.strategy_matrix[i, :] = instructions_array

        assert self.strategy_matrix.shape == (
            self.num_stations, self.num_stations)
        print(f"Strategy matrix updated at {timestamp}")
        print(self.num_stations)
        print(self.strategy_matrix.shape)

        self.last_timestamp = timestamp

    def visualize_graph(self):
        """
        Visualize the time-expanded graph using time as the y-coordinate
        and station IDs as the x-coordinate, with directed arrows.
        """
        # Generate positions for the graph
        pos = {
            node: (hash(node[0]) % 10, node[1].timestamp())
            for node in self.G.nodes
        }

        # Draw the graph with directed arrows
        plt.figure(figsize=(16, 12))
        nx.draw_networkx_nodes(
            self.G, pos, node_size=300, node_color="lightblue")
        nx.draw_networkx_labels(
            self.G, pos,
            labels={
                node: f"{node[0]}@{node[1].strftime('%H:%M')}" for node in self.G.nodes},
            font_size=8
        )
        nx.draw_networkx_edges(self.G, pos, edge_color="gray",
                               alpha=0.7, arrows=True, arrowstyle="-|>", arrowsize=10)

        # Set axis labels
        plt.xlabel("Station ID (hashed)", fontsize=14)
        plt.ylabel("Time", fontsize=14)
        plt.title("Time-Expanded Train Schedule Visualization", fontsize=16)
        plt.show()

    # def update_strategy(self, current_time: datetime):
    #     """
    #     Manually updates the strategy matrix based on the train’s loop:
    #     0 -> 1 -> 2 -> 3 -> 2 -> 0 ...
    #     Passengers stay at their current station if their destination matches it.
    #     """
    #     self.strategy_matrix = np.array([
    #         [0, 1, 1, 1],  # Passengers at station 0
    #         [2, 1, 2, 2],  # Passengers at station 1
    #         [0, 0, 2, 3],  # Passengers at station 2
    #         [2, 2, 2, 3]   # Passengers at station 3
    #     ])

    #     # Assert that diagonal elements are 0, 1, 2, ...
    #     diagonal = np.diag(self.strategy_matrix)
    #     assert np.array_equal(diagonal, np.arange(self.strategy_matrix.shape[0])), (
    #         f"Diagonal elements of the strategy matrix are incorrect: {diagonal}"
    #     )

    def to_json(self) -> dict:
        """
        Serializes the Strategy object to a JSON-compatible dictionary.

        Returns:
            dict: A dictionary representation of the Strategy object.
        """
        return {
            "strategy_matrix": self.strategy_matrix,
        }


class Demand:
    def __init__(self, demand_dict, number_of_stations):
        """
        Initializes the Demand object with a dictionary where keys are timedelta objects
        representing times, and values are demand matrices.
        """
        self.initial_demand = demand_dict
        self._number_of_stations = number_of_stations
        self.reset()

    def get_event_queue(self):
        return list(self.initial_demand.keys())

    def reset(self):
        """Resets the internal demand storage to a copy of the initial demand dictionary."""
        self.active_demand = deepcopy(self.initial_demand)

    def get_demand_until(self, current_time):
        """
        Summarizes and returns the demand for all times less than or equal to the specified current_time.
        Removes these entries from the active demand dictionary.

        Parameters:
        current_time (timedelta): The current simulation time.

        Returns:
        np.ndarray: The cumulative demand matrix up to the current time.
        """
        # Initialize a cumulative demand matrix with zeros
        cumulative_demand = np.zeros(
            (self._number_of_stations, self._number_of_stations), dtype=int)

        # Collect keys to delete after summing their matrices
        keys_to_delete = [
            time for time in self.active_demand if time <= current_time]

        # Sum demand matrices that match the keys and delete them from the internal dictionary
        for time in keys_to_delete:
            cumulative_demand += self.active_demand[time]
            del self.active_demand[time]

        return cumulative_demand

    def to_json(self) -> dict:
        """
        Serializes the Demand object to a JSON-compatible dictionary.

        Returns:
            dict: A dictionary representation of the Demand object.
        """
        return {
            "initial_demand": {k.isoformat(): v.tolist() for k, v in self.initial_demand.items()},
            "active_demand": {k.isoformat(): v.tolist() for k, v in self.active_demand.items()}
        }


class Railroad:
    def __init__(self, graph: nx.DiGraph, start_time: datetime, demand: Demand, trains: dict[str, Train], strategy: Strategy, passenger_matrix: np.ndarray | None = None, report_interval: timedelta | None = None, report_plot: bool = False, report_json: bool = False, savedir: str = None, max_simulation_time: timedelta = timedelta(hours=1), style: dict | None = None):
        """
        Initializes the Railroad object.

        Parameters:
        graph (nx.DiGraph): A directed graph representing the railroad network.
        start_time (datetime): The initial time for the simulation.
        demand (Demand): An instance of the Demand class to manage passenger demand over time.
        trains (dict[str, Train]): A dictionary of Train objects keyed by train IDs.
        strategy (Strategy): An instance of the Strategy class to manage passenger routing.
        passenger_matrix (np.ndarray | None): A matrix representing the current passengers at each station (origin, destination).
        """
        self.G = graph
        self.current_time: datetime = start_time
        self.previous_time = None
        self.strategy = strategy
        self.passenger_matrix = passenger_matrix or np.zeros(
            (self.G.number_of_nodes(), self.G.number_of_nodes()), dtype=int)
        self.demand = demand
        self.delivered_passengers = 0  # Counter for delivered passengers
        self.total_waiting_time = timedelta(0)  # Total waiting time counter
        self.waiting_passengers = 0
        self.demand.reset()
        coords_present = all(
            'station_coordinates' in self.G.nodes[node] for node in self.G.nodes)
        if not coords_present:
            self.pos = nx.spring_layout(self.G)  # Position for all nodes
        else:
            self.pos = {
                node: self.G.nodes[node]['station_coordinates'] for node in self.G.nodes}
        self.trains = trains  # Dictionary of trains
        self.report_interval = report_interval
        self.last_report_time: datetime = None
        self.report_plot = report_plot
        self.report_json = report_json

        self.savedir = savedir
        if report_interval and savedir:
            if not os.path.exists(savedir):
                os.makedirs(savedir)
            # Clear the directory if it already exists
            for filename in os.listdir(savedir):
                file_path = os.path.join(savedir, filename)
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)

        self.passenger_buffer = {}

        self.event_queue = {}
        self.initialize_event_queue(max_simulation_time=max_simulation_time)

        self.style = style or {}

    def add_to_event_queue(self, timestamp: datetime, event_data: dict):
        if timestamp not in self.event_queue:
            self.event_queue[timestamp] = [event_data]
        else:
            self.event_queue[timestamp].append(event_data)
        self.event_queue = dict(sorted(self.event_queue.items()))

    def initialize_event_queue(self, max_simulation_time: timedelta = timedelta(hours=1)):
        for train_id, train in self.trains.items():
            q = train.get_event_queue()
            for timestamp in q:
                self.add_to_event_queue(
                    timestamp, {'type': 'train', 'train_id': train_id})

        q = self.demand.get_event_queue()
        for timestamp in q:
            self.add_to_event_queue(timestamp, {'type': 'demand'})

        for timestamp in self.passenger_buffer.keys():
            self.add_to_event_queue(timestamp, {'type': 'buffer'})

        if self.report_interval:
            q = pd.date_range(
                start=self.current_time, end=self.current_time+max_simulation_time, freq=self.report_interval)
            for timestamp in q:
                self.add_to_event_queue(timestamp, {'type': 'report'})

        self.event_queue = dict(sorted(self.event_queue.items()))

    def remove_delivered_passengers(self):
        # Calculate delivered passengers (those who have reached their destination)
        delivered = np.diag(self.passenger_matrix).sum()
        print(f"Delivered passengers at {self.current_time}: {delivered}")
        self.delivered_passengers += delivered

        # Remove delivered passengers by zeroing out the diagonal
        np.fill_diagonal(self.passenger_matrix, 0)

    def tick(self):
        """
        Advances the simulation time by the specified timestep, updates the passenger matrix
        with new demand, removes delivered passengers, and accumulates total waiting time.

        Parameters:
        timestep (timedelta): The amount of time to advance in the simulation.
        """
        self.previous_time = self.current_time
        self.current_time = list(self.event_queue.keys())[
            0]  # dict is already sorted by time
        self.current_events = self.event_queue[self.current_time]
        for event in self.current_events:
            if event['type'] == 'demand':
                new_demand = self.demand.get_demand_until(self.current_time)
                self.passenger_matrix += new_demand
            elif event['type'] == 'buffer':
                timestamps_to_remove = []
                for timestamp, passengers in self.passenger_buffer.items():
                    if timestamp <= self.current_time:
                        timestamps_to_remove.append(timestamp)
                        self.passenger_matrix += passengers
                for timestamp in timestamps_to_remove:
                    self.passenger_buffer.pop(timestamp)
            elif event['type'] == 'train':
                self.strategy.update_strategy(self.current_time)

                train_id = event['train_id']
                train = self.trains[train_id]
                train.tick(self)
                # print(f"{train_id} at {self.current_time} report:")
                # print(train)
            elif event['type'] == 'report':
                self.last_report_time = self.current_time
                if self.report_plot:
                    self.draw()
                print(
                    f"Total delivered passengers: {self.delivered_passengers}")
                print(f"Total waiting time: {self.total_waiting_time}")
                if self.report_json:
                    with open(os.path.join(self.savedir, f"railroad_{self.current_time.isoformat()}.json"), 'w') as f:
                        json.dump(self.to_json(), f,
                                  cls=CustomJSONEncoder, indent=4)

        self.remove_delivered_passengers()

        # Accumulate total waiting time (sum of all passengers * timestep duration)
        self.waiting_passengers = self.passenger_matrix.sum() + sum(train.number_of_passengers
                                                                    for train in self.trains.values()) + sum(m.sum() for m in self.passenger_buffer.values())
        self.total_waiting_time += self.waiting_passengers * \
            (self.current_time - self.previous_time)

        # Remove the current time from the event queue
        self.event_queue.pop(self.current_time)

        for train_id, train in self.trains.items():
            print(f"{train_id} at {self.current_time} report:")
            print(train)

        return len(self.event_queue) > 0

    def passengers_in_buffer_matrix(self):
        """passengers ib buffer for each station"""
        result = np.zeros_like(self.passenger_matrix)
        for passengers in self.passenger_buffer.values():
            result += passengers
        return result

    @property
    def total_passengers(self):
        """
        Returns the total number of passengers in the system (on trains and at stations).
        """
        return self.passenger_matrix.sum() + sum(train.number_of_passengers for train in self.trains.values()) + sum(m.sum() for m in self.passenger_buffer.values())

    def add_passengers_to_buffer(self, passengers, station):
        passengers_matrix = np.zeros_like(self.passenger_matrix)
        passengers_matrix[station, :] = passengers
        timestamp = self.current_time + \
            self.G.nodes[station].get('transfer_time', timedelta(minutes=1))
        if timestamp in self.passenger_buffer:
            self.passenger_buffer[timestamp] += passengers_matrix
        else:
            self.passenger_buffer[timestamp] = passengers_matrix
            self.add_to_event_queue(timestamp, {'type': 'buffer'})

    def board_passengers_from_station(self, passengers, station):
        """Boards passengers from the station to the train."""
        self.passenger_matrix[station, :] -= passengers

    def get_passengers_to_station(self, station):
        """Returns the column for passengers wanting to get to the specified station."""
        return self.passenger_matrix[:, station]

    def get_passengers_at_station(self, station):
        """Returns the row for passengers waiting at the specified station."""
        return self.passenger_matrix[station, :]

    def draw(self):
        """Draws the railroad network graph with passenger counts at each station and train positions."""

        # Draw the graph nodes and edges
        nx.draw(self.G, self.pos, with_labels=True, node_size=500,
                node_color="lightblue", font_size=10, font_weight="bold")

        # Draw edge labels for travel times
        edge_labels = {(u, v): f"{int(d['travel_time'].total_seconds() // 60)} min" for u,
                       v, d in self.G.edges(data=True)}
        nx.draw_networkx_edge_labels(
            self.G, self.pos, edge_labels=edge_labels, font_size=9, label_pos=0.5)

        # Display passenger counts near each node
        for node in self.G.nodes:
            if len(self.G.nodes) < 10:
                passengers_waiting = self.get_passengers_at_station(node)
                passenger_info = f"{list(passengers_waiting)}"
            else:
                total_passangers_waiting = self.get_passengers_at_station(
                    node).sum()
                passenger_info = f"{total_passangers_waiting}"
            label_pos = (self.pos[node][0], self.pos[node]
                         [1] + 2 * self.style.get('offset', 0.05))
            plt.text(label_pos[0], label_pos[1],
                     passenger_info, fontsize=9, ha='center')
            if 'station_name' in self.G.nodes[node]:
                station_name = self.G.nodes[node]['station_name']
                label_pos = (self.pos[node][0], self.pos[node]
                             [1] - 2 * self.style.get('offset', 0.05))
                plt.text(label_pos[0], label_pos[1],
                         station_name, fontsize=9, ha='center')

                # Add train positions and passenger counts
        for train_id, train in self.trains.items():
            if train.at_station or (train.previous_station is not None and train.next_station is not None):
                if train.at_station:
                    train_pos = self.pos[train.current_station]
                else:
                    # Interpolate train position based on current time
                    prev_station = train.previous_station
                    next_station = train.next_station
                    prev_time = train.previous_departure_time
                    next_time = train.next_arrival_time

                    assert next_time > prev_time, f'Time glitch, next time: {next_time}, prev time: {prev_time}'
                    elapsed_time = (self.current_time -
                                    prev_time).total_seconds()
                    travel_time = (next_time - prev_time).total_seconds()

                    interpolation_factor = elapsed_time / travel_time if travel_time > 0 else 0
                    train_pos = (
                        self.pos[prev_station][0] * (1 - interpolation_factor) +
                        self.pos[next_station][0] * interpolation_factor,
                        self.pos[prev_station][1] * (1 - interpolation_factor) +
                        self.pos[next_station][1] * interpolation_factor,
                    )

                # Draw the train on top of the graph
                plt.scatter(*train_pos, c=train.style.get('color', 'red'), s=100,
                            label=train_id, zorder=900)

                # Display total passengers above the train
                passenger_label_pos = (
                    train_pos[0], train_pos[1] + self.style.get('offset', 0.05))
                plt.text(passenger_label_pos[0], passenger_label_pos[1],
                         f"{train.number_of_passengers} on board", fontsize=9, color=train.style.get('color', 'red'), ha='center', zorder=901)

        buffer_matrix = self.passengers_in_buffer_matrix()
        # Display passengers in buffer above each station
        for node in self.G.nodes:
            passengers_in_buffer = buffer_matrix[node, :]
            if len(self.G.nodes) < 10:
                buffer_info = f"{list(passengers_in_buffer)}"
            else:
                buffer_info = f"{passengers_in_buffer.sum()}"
            buffer_label_pos = (
                self.pos[node][0], self.pos[node][1] + 3 * self.style.get('offset', 0.05))
            plt.text(buffer_label_pos[0], buffer_label_pos[1],
                     buffer_info, fontsize=9, color='green', ha='center')

        # Add current time window at the bottom right corner
        current_time_str = self.current_time.strftime("%Y-%m-%d %H:%M:%S")
        plt.text(0.95, 0.01, f"Current Time: {current_time_str}",
                 horizontalalignment='right', verticalalignment='bottom',
                 transform=plt.gca().transAxes, fontsize=10,
                 bbox=dict(facecolor='white', alpha=0.5))
        # Show plot with legend
        plt.legend(loc="upper right")
        if self.savedir:
            plt.savefig(os.path.join(
                self.savedir, f"railroad_{self.current_time.isoformat()}.png"))
        plt.show()

    @staticmethod
    def full_schedule_from_trains(trains: dict[str, Train]) -> pd.DataFrame:
        """
        Creates a full schedule DataFrame from a dictionary of Train objects.

        Parameters:
        - trains (dict[str, Train]): A dictionary of Train objects keyed by train IDs.

        Returns:
        - pd.DataFrame: A DataFrame with columns for train ID, departure station, arrival station, departure time, and arrival time.
        """
        df = pd.DataFrame()
        for train_id, train in trains.items():
            train_df = train.get_pandas_schedule()
            train_df['train_id'] = train_id
            df = pd.concat([df, train_df])
        return df

    def get_full_schedule(self):
        df = pd.DataFrame()
        for train_id, train in self.trains.items():
            train_df = train.get_pandas_schedule()
            train_df['train_id'] = train_id
            df = pd.concat([df, train_df])
        return df

    def to_json(self) -> dict:
        """
        Serializes the Railroad object to a JSON-compatible dictionary.

        Returns:
            dict: A dictionary representation of the Railroad object.
        """
        return {
            "graph": nx.node_link_data(self.G),
            "current_time": self.current_time,
            "strategy": self.strategy.to_json(),
            "passenger_matrix": self.passenger_matrix,
            "demand": self.demand.to_json(),
            "delivered_passengers": self.delivered_passengers,
            "total_waiting_time": self.total_waiting_time,
            "waiting_passengers": self.waiting_passengers,
            "trains": {train_id: train.to_json() for train_id, train in self.trains.items()},
            "report_interval": self.report_interval if self.report_interval else None,
            "savedir": self.savedir,
            "passenger_buffer": {str(k): v.tolist() for k, v in self.passenger_buffer.items()}
        }
