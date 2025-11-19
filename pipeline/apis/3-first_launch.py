#!/usr/bin/python3
"""
Gets the earliest SpaceX launch from the API and prints its information.
"""

import requests


def get_first_launch():
    """Fetches the earliest launch and prints formatted info."""
    url = "https://api.spacexdata.com/v4/launches"
    launches = requests.get(url).json()

    launches.sort(key=lambda x: x.get("date_unix", float("inf")))
    first = launches[0]

    launch_name = first.get("name")
    date_local = first.get("date_local")
    rocket_id = first.get("rocket")
    pad_id = first.get("launchpad")

    rocket_url = (
        "https://api.spacexdata.com/v4/rockets/" + rocket_id
    )
    rocket = requests.get(rocket_url).json()

    pad_url = (
        "https://api.spacexdata.com/v4/launchpads/" + pad_id
    )
    pad = requests.get(pad_url).json()

    rocket_name = rocket.get("name")
    pad_name = pad.get("name")
    locality = pad.get("locality")

    output = (
        f"{launch_name} ({date_local}) {rocket_name} - "
        f"{pad_name} ({locality})"
    )
    print(output)


if __name__ == "__main__":
    get_first_launch()
