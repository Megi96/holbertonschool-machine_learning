#!/usr/bin/env python3
"""
Fetch the first upcoming SpaceX launch using the unofficial SpaceX API
and print:

<name> (<date_local>) <rocket name> - <launchpad name> (<locality>)
"""

import requests


def get_first_upcoming_launch():
    """
    Retrieves and prints the first upcoming SpaceX launch.
    """
    url = "https://api.spacexdata.com/v4/launches"
    launches = requests.get(url).json()

    # Filter only upcoming launches
    upcoming = [
        launch for launch in launches if launch.get("upcoming")
    ]

    # Sort by date_unix
    upcoming.sort(key=lambda x: x.get("date_unix", float("inf")))

    first = upcoming[0]

    lname = first.get("name")
    ldate = first.get("date_local")
    rocket_id = first.get("rocket")
    pad_id = first.get("launchpad")

    rocket_url = "https://api.spacexdata.com/v4/rockets/" + rocket_id
    rocket = requests.get(rocket_url).json()

    pad_url = "https://api.spacexdata.com/v4/launchpads/" + pad_id
    pad = requests.get(pad_url).json()

    rocket_name = rocket.get("name")
    pad_name = pad.get("name")
    locality = pad.get("locality")

    output = (
        f"{lname} ({ldate}) {rocket_name} - "
        f"{pad_name} ({locality})"
    )
    print(output)


if __name__ == "__main__":
    get_first_upcoming_launch()
