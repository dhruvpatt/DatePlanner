from flask import Flask, render_template, request
from mapping import plot_map
from graph_poi_algorithms import build_graph, build_path_length
import geocoder

app = Flask(__name__)


@app.route('/')
def home():
    """
    Starting point of the application. This function is called when the localhost website is opened. It uses the user's
    IP address to get a rough estimate of a starting location (it is actually the location of the ISP but for the
    pupose of testing it is accurate enough). It then renders the page that contains the form that needs to be filled
    that has the required by our algorithms to create a path that routes the user from their location to an end point.
    """
    g = ["-79.3987", "43.6629"]  # backup
    try:
        g = geocoder.ip('me').latlng
    except Exception as ex:
        print(ex)
    return render_template('home.html', place_long=g[1], place_lat=g[0])


@app.route('/submit', methods=['GET', 'POST'])
def submit():
    """
    This is the function responsible for routing data from the form to the algorithms responsible for generating a path.
    Once the route has been generated it is passed to the mapping function which then creates an HTML render of the map
    which is then rendered on a redirect page.
    """
    form = request.form
    starting_point = form['starting_point']
    current_location = (float(form['Longitude']), float(form['Latitude']))
    number_destinations = int(form['Number Destinations'])
    pois = form.getlist('POIs')
    end_point = form['end_point']

    plot_graph = build_graph("places-of-interest-and-attractions_CS_project.csv")
    path = build_path_length(n=number_destinations, map_poi=plot_graph, start_type=starting_point, end_type=end_point,
                             preference=set(pois), user_coord=tuple(current_location))
    div = plot_map(path, tuple(current_location))
    return render_template('submit.html', plotly_map=div)


if __name__ == '__main__':
    app.run(port=8001)
