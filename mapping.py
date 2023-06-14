"""PLotly Maps"""
import plotly.graph_objs as go
from make_api_call import make_request
from graph_poi_algorithms import PointOfInterest

TOKEN = "Add Map Box Token"
SECRET_TOKEN = "Add Map Box Secret Token"


def points_of_interest_coords(points: list[PointOfInterest]) -> tuple[list[float], list[float]]:
    """Helper function that returns a tuple of lists that contain the latitude and longitude coordinates"""
    lats = []
    lngs = []
    for point in points:
        coord = point.coords
        lngs.append(coord[0])
        lats.append(coord[1])

    return (lngs, lats)


def generate_colors(num_points: int) -> list[str]:
    """
    Generates a list of visually unique colors in a gradient, based on the number of points.
    """
    # Define the start and end colors for the gradient
    color1 = (255, 0, 0)

    # Calculate the step size for the color gradient
    step_size = 255 / (num_points - 1)
    # Create a list of colors in the gradient
    rgb_colors = []
    for i in range(num_points):
        # Calculate the color for the current point
        r = int(color1[0] - i * step_size)
        g = int(color1[1] + i * step_size)
        b = 0
        clr = (r, g, b)

        # Add the color to the list
        rgb_colors.append(clr)

    colors = [f'rgb({color[0]}, {color[1]}, {color[2]})' for color in rgb_colors]

    return colors


def create_hover_text(points: list[PointOfInterest]) -> list[str]:
    """Creates a string that will be used as the hover text for the plotly map object"""

    names = []
    categories = []
    addresses = []
    descriptions = []
    phone_numbers = []

    for point in points:
        names.append(point.name)
        categories.append(point.category)
        addresses.append(point.address)
        descriptions.append(point.description)
        phone_numbers.append(point.phone)

    hovertext = []
    for i in range(len(names)):
        hovertext.append(f'<b>Destination #</b>: {i+1}<br >'
                         f'<b>Name:</b> {names[i]}<br>'
                         f'<b>Address:</b> {addresses[i]}<br>'
                         f'<b>Phone Number:</b> {phone_numbers[i]}<br>'
                         f'<b>Description:</b> {descriptions[i]}<br>'
                         f'<b>Category:</b> {categories[i]}<br>')

    return hovertext


def get_start_route(start_point: tuple[float, float], end_point: PointOfInterest) -> tuple[list[float], list[float]]:
    """Returns the route between the start point and the first point of interest"""
    lats = []
    longs = []
    start_string = f"{start_point[1]}, {start_point[0]}"
    end_string = f"{end_point.coords[1]}, {end_point.coords[0]}"
    coords = make_request(start=start_string, end=end_string)
    for pair in coords:
        for mapping in pair:
            lat = mapping['lat']
            lng = mapping['lng']
            lats.append(lat)
            longs.append(lng)

    return (lats, longs)


def plot_map(points: list[PointOfInterest], start_point: tuple[float, float]) -> str:
    """A function to plot a map containing the path that minimizes the distance
     between all points of interest"""

    # list of latitude and longitudes to pass into trace 1
    lats = []
    longs = []
    coord_list = []
    start_route = get_start_route(start_point=start_point, end_point=points[0])

    points_of_interest = points_of_interest_coords(points)
    colors = generate_colors(num_points=len(points_of_interest[0]))

    # creates lists of latitudes and longitudes for points of interests
    poi_lngs = points_of_interest[0]
    poi_lats = points_of_interest[1]
    hover_text = create_hover_text(points)

    x1 = poi_lngs[0]
    y1 = poi_lats[0]

    # for loop that makes a request to Google Maps API and appends the coordinates of each
    # turn into coord_list
    for i in range(len(points) - 1):
        # converting the coordinates into a string.
        start_string = f"{points[i].coords[1]}, {points[i].coords[0]}"
        end_string = f"{points[i + 1].coords[1]}, {points[i + 1].coords[0]}"
        # sending a request to google maps api to retrieve directions
        coords = make_request(start=start_string, end=end_string)
        # appending to a list of coordinates
        coord_list.append(coords)

    # looping through the coordinates in coord_list and appending them the lat
    # and lng lists which will get passed into the ScatterMapBox traces.
    for coords in coord_list:
        for pair in coords:
            lat = pair[0]['lat']
            lng = pair[0]['lng']
            lats.append(lat)
            longs.append(lng)

    # adding the path to the list of lats and longs.
    last_location = coord_list[-1][-1]
    lats.append(last_location[1]['lat'])
    longs.append(last_location[1]['lng'])

    # trace1 plots the roads/directions that the user would have to take.
    trace1 = go.Scattermapbox(
        lat=lats,
        lon=longs,
        mode="markers+text+lines",
        marker=dict(size=5.0, color='blue'),
        showlegend=False
    )

    # trace2 plots the points of interests that the user would visit
    trace2 = go.Scattermapbox(
        lat=poi_lats,
        lon=poi_lngs,
        mode='markers+text',
        marker=dict(
            size=15.0,
            color=colors
        ),
        hovertext=hover_text,
        hoverinfo='text',
        textfont=dict(
            family='sans-serif',
            size=10,
            color='black'
        ),
        textposition='top center',
        showlegend=False
    )

    # plots the route between the start location and the first point of interest
    start_route = go.Scattermapbox(
        lat=start_route[0],
        lon=start_route[1],
        mode='markers+lines',
        line=dict(
            color='black'
        ),
        name="Start Route"
    )

    # plots the start location as a marker
    start_location = go.Scattermapbox(
        lat=[start_point[1]],
        lon=[start_point[0]],
        mode='markers',
        marker=dict(
            size=15.0,
            color='black'
        ),
        name='Start Location'
    )

    # adding all the traces to the figure
    fig = go.Figure(data=[trace1, trace2, start_route, start_location])
    fig.update_layout(

        mapbox=dict(
            accesstoken=TOKEN,
            center=go.layout.mapbox.Center(
                lon=x1,
                lat=y1
            ),
            zoom=17.0,
            style="open-street-map",
        ),
        height=700
    )

    div = fig.to_html(full_html=False)
    return div


if __name__ == "__main__":
    import python_ta
    python_ta.check_all(config={
        'extra-imports': ['plotly.graph_objs', 'make_api_call', 'graph_poi_algorithms'],
        'allowed-io': [],
        'disable': ['too-many-locals'],
        'max-line-length': 120
    })
