#!/usr/bin/env python3
"""
Gets the earliest successful SpaceX launch and prints its information.
"""

import requests


def get_first_launch():
    """
    Fetches the earliest successful SpaceX launch and prints formatted information.
    """
    url = "https://api.spacexdata.com/v4/launches"
    launches = requests.get(url).json()

    valid_launches = [
        l for l in launches
        if l.get("success") is True
        and l.get("upcoming") is False
        and l.get("date_unix")
        and l.get("rocket")
        and l.get("launchpad")
    ]

    valid_launches.sort(key=lambda x: x["date_unix"])
    first = valid_launches[0]

    launch_name = first.get("name")
    date_local = first.get("date_local")
    rocket_id = first.get("rocket")
    pad_id = first.get("launchpad")

    rocket = requests.get(f"https://api.spacexdata.com/v4/rockets/{rocket_id}").json()
    pad = requests.get(f"https://api.spacexdata.com/v4/launchpads/{pad_id}").json()

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
