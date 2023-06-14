"""Make API Request"""
import requests

KEY = 'Add Google Maps Key Here'


def make_request(start: str, end: str) -> list[list[dict[str, float]]]:
    """Makes API Request to Google Maps API to get a list
    of dictionaries of directions."""

    endpoint = f'origin={start}&destination={end}&key={KEY}'
    headers = {}
    payload = {}
    url = "https://maps.googleapis.com/maps/api/directions/json?" + endpoint

    response = requests.request(method="GET", url=url, headers=headers, data=payload)

    response_json = response.json()
    directions_coords = []
    steps = response_json['routes'][0]['legs'][0]['steps']

    for step in steps:
        sublist = [step['start_location'], step['end_location']]
        directions_coords.append(sublist)

    return directions_coords


if __name__ == "__main__":

    import python_ta
    python_ta.check_all(config={
        'extra-imports': ['requests'],
        'allowed-io': [],
        'max-line-length': 120
    })
