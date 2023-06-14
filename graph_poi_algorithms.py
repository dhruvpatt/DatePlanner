"""CSC111 Winter 2023 Project

File for Graph, POI, Algorithm for the CS project

This file contains the code for guilding the graph and any computations to create the path
Copyright by Harley Cai,
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import csv
import math
import random
import python_ta

# global variable for the types
TYPES = set()


# @check_contracts
@dataclass
class PointOfInterest:
    """
    A custom data type that represents data for given POIs

    Instance Attributes:
        - name: name of the POI
        - category: type of POI, ie. cinema, restaurants, etc.
        - phone: phone number of the place
        - website: website of the place
        - address: address of the place
        - postal_code: postal code of the place
        - city: the city the POI is at
        - ward: the big intersection the POI is ate
        - description: the description of the POI
        - coords: coordinates of the POI (long, lat)
    """
    name: str
    category: str
    phone: str
    website: str
    address: str
    postal_code: str
    city: str
    ward: str
    description: str
    coords: tuple[float, float]


# function to calcuate between 2 POIs
def calculate_distance(poi1: PointOfInterest | tuple[float, float], poi2: PointOfInterest) -> float:
    """
    Calculates the distance between 2 POIs or a random coords to a POI

    Preconditions:
        - poi1 != poi2

    >>> poi_start = PointOfInterest("BMO Field","Entertainment","416-815-5982","https://www.bmofield.com/"\
    ,"170 Princes' Blvd","M6K 3C3","Toronto","Spadina-Fort York", \
     "BMO Field is home to the Toronto FC (Major League Soccer),Canada's national soccer team.", \
    (-79.4184156138918, 43.6346633567159))
    >>> poi_end = PointOfInterest("Scotiabank Arena (formerly Air Canada Centre)", \
    "Entertainment","416-815-5500","https://www.scotiabankarena.com/","40 Bay St","M5J 2X2","Toronto",\
    "Spadina-Fort York","The Scotiabank Arena is a multi-purpose indoor sporting arena and concert venue.",\
    (-79.3788009534123, 43.6454365794855))
    >>> calculate_distance(poi_start,poi_end)
    247.4533617022602
    """
    # get coordinates for both POIs
    if isinstance(poi1, tuple):
        coords1 = poi1
    else:
        coords1 = poi1.coords
    coords2 = poi2.coords
    # store latitude and longitude for both POIs
    lat1 = coords1[1]
    long1 = coords1[0]
    lat2 = coords2[1]
    long2 = coords2[0]
    # Calculate the differences between the latitudes and longitudes
    dlat = lat2 - lat1
    dlon = long2 - long1

    # Apply the Haversine formula
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    radius = 6371  # Earth's radius in kilometers
    distance = radius * c

    return distance


class _Vertex:
    """A vertex in a graph.
    Instance Attributes:
        - name: name of POI
        - item: The data stored in this vertex.
        - neighbours: The vertices that are adjacent to this vertex with the weight.

    Representation Invariants:
        - self not in self.neighbours
        - all(self in u.neighbours for u in self.neighbours)
    """
    item: PointOfInterest
    neighbours: dict[_Vertex, float]

    def __init__(self, item: Any) -> None:
        """Initialize a new vertex with the given item and neighbours."""
        self.item = item
        self.neighbours = {}


class Graph:
    """A graph representing the map of all POIs
    Private Instance Attributes:
    - _vertices: A collection of the vertices contained in this graph. Maps POI name to _Vertex object.

    Representation Invariants:
    - all(item == self._vertices[item].item.name for item in self._vertices)

    """

    _vertices: dict[str, _Vertex]

    def __init__(self) -> None:
        """Initialize an empty graph (no vertices or edges)."""
        self._vertices = {}

    def add_vertex(self, poi: PointOfInterest) -> None:
        """Add a vertex with the given item to this graph.

        The new vertex is not adjacent to any other vertices.

        Preconditions:
            - item not in self._vertices
        """
        # 1. Create a new _Vertex
        new_vertex = _Vertex(poi)

        # 2. Add the _Vertex to self._vertices
        self._vertices[poi.name] = new_vertex

    def add_edge(self, poi1: PointOfInterest, poi2: PointOfInterest) -> None:
        """Add an edge between the two vertices with the given items in this graph.

        Raise a ValueError if item1 or item2 do not appear as vertices in this graph.

        Preconditions:
            - item1 != item2
        """
        if poi1.name in self._vertices and poi2.name in self._vertices:
            # Add the edge between the two vertices
            v1 = self._vertices[poi1.name]
            v2 = self._vertices[poi2.name]
            # call the distance between them to create the weighted edges
            distance = calculate_distance(v1.item, v2.item)
            v1.neighbours[v2] = distance
            v2.neighbours[v1] = distance
        else:
            raise ValueError

    def smart_distance(self, start: _Vertex, end: _Vertex, visited: list, preference: set[str]) -> list[_Vertex]:
        """
        creates a smart path from point A to point B. The path is the most opitmal ie, it is the shortest path
        considering the user preference and the length of the path so a direct path is usally not wanted

        Preconditions:
        - start != end
        - start in self._vertices
        - end in self._vertices

        >>> graph = build_graph("places-of-interest-and-attractions_CS_project.csv")
        >>> start1 = graph._vertices["BMO Field"]
        >>> end1 = graph._vertices["Berkeley Street Theatre"]
        >>> path1 = graph.smart_distance(start1,end1,[],set())
        >>> len(path1) == 3
        True
        >>> [x.item.name for x in path1] == ['BMO Field', 'Meridian Hall (formerly Sony Centre for the Performing Arts)', 'Berkeley Street Theatre'] #ignore error for test
        True
        """

        # start the distances to infinity and the start node to 0, so it first assumes that all the path
        # distance is infinity
        distances = {points_interest: math.inf for points_interest in self._vertices.values()}
        distances[start] = 0

        # this list of previous nodes so we can look back and see the shortest path
        best_path_data = {points_interest: None for points_interest in self._vertices.values()}
        # if it is not at end
        counter = 0
        while end not in visited:
            # we want to guarentee that the length of the path is not a direct path for start to end
            # so we will restrict the last POI until it meets this requirement
            if 0 == counter:
                extra_restriction = True
            else:
                extra_restriction = False
            # Find the smallest distance
            # start at inf and then it will accumlate if there is a smaller distance
            current_distance = math.inf
            # this is just an accumlator so start at None
            current_node = None
            # go through all the nodes
            for node in distances:
                if node not in visited and distances[node] < current_distance:
                    current_distance = distances[node]
                    current_node = node
            # copy neighbour of the vertex so we can mutate it
            neighbours = current_node.neighbours.copy()
            if extra_restriction:
                neighbours.pop(end)
            # Update the distances for the current node's neighbors
            for neighbor, weight in neighbours.items():
                # if the POI is in the preference then we need to adjust the weight
                if neighbor.item.category in preference:
                    # just a random factor I came up with
                    weight = weight / 1.129
                # calculate the distance from start to this node
                distance = distances[current_node] + weight
                # we find a shorter distance to this node
                if distance < distances[neighbor]:
                    # replace the distance
                    distances[neighbor] = distance
                    # and change the node that gets us to here
                    best_path_data[neighbor] = current_node
            # Mark the current node as visited
            visited.append(current_node)
            counter += 1
        # Build the shortest path as a list of vertices backwards
        path = [end]
        current_node = end
        # it will go through all the best_path_data node from the end to create the path
        while current_node != start:
            previous_node = best_path_data[current_node]
            path.append(previous_node)
            current_node = previous_node
        # flip it so it is correct order
        path.reverse()
        # Return the shortest path
        return path

    def get_poi_of_type(self, same_types: str) -> list[_Vertex]:
        """
        Return a list of POIs that have the same category as same_types

        Preconditions:
            - same_types in types

        >>> graph = build_graph("places-of-interest-and-attractions_CS_project.csv")
        >>> a = graph.get_poi_of_type("Cinema")
        >>> len(a)
        7
        """
        # return the list of POI that have same type
        return [vertex for vertex in self._vertices.values() if vertex.item.category == same_types]


def build_graph(file: str) -> Graph:
    """Build a weighted graph storing the POI data from the given file.
    1. Creates the POI objecst
    2. Creates the weigthed graph where vertex is POI, edges are distance

    Preconditions:
        - file is the path to a csv file in the format of given example POI file
    """

    graph_so_far = Graph()  # The start of a graph
    # create an accumlator list to store all the POI objects
    list_poi = []
    # load the data
    with open(file) as csv_file:
        reader = csv.reader(csv_file)
        next(reader)
        for row in reader:
            name = row[0].strip()
            category = row[1].strip()
            # add new category in to types
            if category not in TYPES:
                TYPES.add(category)
            phone = row[2].strip()
            website = row[3].strip()
            address = row[4].strip()
            postal_code = row[5].strip()
            city = row[6].strip()
            ward = row[7].strip()
            description = row[8].strip()
            coords = row[9]
            # change coords to tuple
            coords = coords.replace("(", "")  # remove brackets
            coords = coords.replace(")", "")
            # split it to long and lat
            coords = coords.split(",")
            # remove whitespace from beginning and end
            coords = [coord.strip() for coord in coords]
            coords = (float(coords[0]), float(coords[1]))  # change to tuple[float,float]
            new_poi = PointOfInterest(name=name, category=category, phone=phone, website=website, address=address,
                                      postal_code=postal_code, city=city, ward=ward, description=description,
                                      coords=coords)
            list_poi.append(new_poi)

        # Add all vertices first
        for poi in list_poi:
            # Add new vertex for i
            graph_so_far.add_vertex(poi)

        # Add all edges
        for i in range(0, len(list_poi)):
            # Add edges to all previous vertices (0 <= j < i)
            for j in range(0, i):
                graph_so_far.add_edge(list_poi[i], list_poi[j])

        return graph_so_far


# @check_contracts
def build_path_length_helper(n: int, map_poi: Graph, start_poi: _Vertex, end_poi: _Vertex, preference: set[str]) -> \
        list[_Vertex]:
    """
    calls the method smart path but since smart path only gives a path of length 3, this will call the
    start with the middle POI to find the fastest path from start to middle and then middle to end and repeats
    the call to n paths

    Preconditions:
        - start_poi != end_poi
        - start_poi in map_poi._verteices
        - end_poi in map_poi._verteices
        - len(map_poi._verteices) >= n >= 3

    >>> graph = build_graph("places-of-interest-and-attractions_CS_project.csv")
    >>> start1 = graph._vertices["BMO Field"] # just for testing so ignore error
    >>> end1 = graph._vertices["Berkeley Street Theatre"]
    >>> a = build_path_length_helper(5, graph, start1,end1,set())
    >>> len(a)
    5
    >>> a[0] == start1
    True
    >>> a[len(a)-1] == end1
    True


    """
    # path accumlator
    path = []
    # copy the PREFERENCE since we need to mutate it
    preference_copy = preference.copy()
    i = 0
    # creates a path of length n
    while len(path) != n:
        if i == 0:
            # if it is start get the path from start to end which will return length of 3 path
            path = map_poi.smart_distance(start_poi, end_poi, [], preference_copy)
        else:
            # check if it is skewed and fix it ie if there is a big gap between A to B find a cloest midpoint
            if calculate_distance(path[len(path) - 2].item, path[len(path) - 3].item) >= 20:
                visited = path.copy()
                visited.remove(path[len(path) - 3])
                visited.remove(path[len(path) - 2])
                index_to_insert = len(path) - 2
                new_poi = map_poi.smart_distance(path[len(path) - 3], path[len(path) - 2], visited, preference_copy)[1]
                path.insert(index_to_insert, new_poi)
            # adds a new POI to the path, between the internal value and the end
            else:
                visited = path.copy()
                visited.remove(end_poi)
                visited.remove(path[len(path) - 2])
                index_to_insert = len(path) - 2
                new_poi = map_poi.smart_distance(path[len(path) - 2], end_poi, visited, preference_copy)[1]
                path.insert(index_to_insert + 1, new_poi)

        # remove the type in preference if it is in the path
        for poi in path:
            if poi.item.category in preference_copy:
                preference_copy.remove(poi.item.category)
        i += 1
    return path


def get_start_poi(map_poi: Graph, user_coord: tuple[float, float], first_poi_type: str) -> _Vertex:
    """
    Gets the close POI of the given type. This is a helper function for build_path_length

    Preconditions:
        - first_poi_type in TYPES

    >>> graph = build_graph("places-of-interest-and-attractions_CS_project.csv")
    >>> a = get_start_poi(graph, (40.0,60.0), "Cinema")
    >>> a.item.name == "TIFF Bell Lightbox"
    True

    """

    # call graph method get_poi_of_type to get all the POI in the graph of that type
    list_first_poi = map_poi.get_poi_of_type(first_poi_type)
    # get the poi of min distance from users current Coords
    closest = list_first_poi[0]
    min_dist = calculate_distance(user_coord, list_first_poi[0].item)
    # get shortest poi form user place
    for poi in list_first_poi:
        distance = calculate_distance(user_coord, poi.item)
        if distance < min_dist:
            min_dist = distance
            closest = poi
    return closest


def get_end_poi(map_poi: Graph, n: int, end_poi_type: str, start_poi: _Vertex) -> _Vertex:
    """
    Gets the end POI of the given type. This is a helper function for build_path_length

    Preconditions:
        - end_poi_type in TYPES
        - len(map_poi._verteices) >= n >= 3

    >>> graph = build_graph("places-of-interest-and-attractions_CS_project.csv")
    >>> a = get_end_poi(graph, 5, "Restaurants", get_start_poi(graph, (40.0,60.0), "Cinema"))
    >>> a.item.name =='Byblos Downtown'
    True

    """

    # call graph method get_poi_of_type to get all the POI in the graph of that type
    list_end_poi = map_poi.get_poi_of_type(end_poi_type)
    # create a list of distance from users current Coords and the POI for that distance
    list_dist_poi = [(calculate_distance(start_poi.item, poi.item), poi) for poi in list_end_poi if
                     poi is not start_poi]
    # sort it by distance
    list_dist_poi.sort(key=lambda x: x[0])
    # get index to seperate to 1/3
    index_thirds = len(list_end_poi) // 3
    if len(list_dist_poi) < 6 or n < 10:
        return list_dist_poi[0][1]
    # for the given path length choose the best end poi ie, long path end should be far, short path end should be close
    if n < 20:
        return random.choice(list_dist_poi[1:index_thirds])[1]
    if n < 40:
        return random.choice(list_dist_poi[index_thirds:index_thirds * 2])[1]
    if n < 80:
        return random.choice(list_dist_poi[index_thirds * 2:])[1]
    else:
        return list_dist_poi[-1][1]


def build_path_length(n: int, map_poi: Graph, start_type: str, end_type: str, preference: set[str],
                      user_coord: tuple[float, float]) -> list[PointOfInterest]:
    """
    calls the build_path_length_helper to build a path of length n, it gets the specific start, end poi by
    calling other helper functions: get_start_poi, get_end_poi

    Preconditions:
        - len(map_poi._verteices) >= n >= 3
        - start_type in TYPES
        - end_type in TYPES
        - user_coords must be in GTA

    >>> graph = build_graph("places-of-interest-and-attractions_CS_project.csv")
    >>> a = build_path_length(5, graph, "Cinema", "Restaurants", set(), (40.0,60.0))
    >>> a[0].name == "TIFF Bell Lightbox"
    True
    >>> len(a)
    5
    """
    # get start
    start_poi = get_start_poi(map_poi, user_coord, start_type)
    # get end
    end_poi = get_end_poi(map_poi, n, end_type, start_poi)
    # return path
    path = build_path_length_helper(n, map_poi, start_poi, end_poi, preference)
    return [vertex.item for vertex in path]


# for testing
if __name__ == "__main__":
    graph = build_graph("places-of-interest-and-attractions_CS_project.csv")
    # start1 = graph._vertices["BMO Field"]
    # end1 = graph._vertices["Berkeley Street Theatre"]

    import doctest

    doctest.testmod(verbose=True)

    python_ta.check_all(config={
        'extra-imports': ["__future__", "dataclasses", "typing", "csv", "math", "random"],
        # the names (strs) of imported modules
        'allowed-io': ["build_graph"],  # the names (strs) of functions that call print/open/input
        'max-line-length': 120,
        'disable': ['too-many-instance-attributes', 'too-many-locals', "too-many-arguments"]
    })
