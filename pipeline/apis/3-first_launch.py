#!/usr/bin/env python3
"""
Script that retrieves and displays the first SpaceX launch.
It prints the launch name, local date, rocket name, and launchpad
information in the following format:

<launch name> (<date>) <rocket name> - <launchpad name> (<locality>)
"""

import requests
from datetime import datetime
import pytz


def fetch_json(url):
    """
    Fetch JSON data from a given URL.

    Args:
        url (str): API endpoint.

    Returns:
        dict or list: Parsed JSON response.
    """
    return requests.get(url).json()


def get_first_launch():
    """
    Retrieve the earliest SpaceX launch based on date_unix.

    Returns:
        dict: Launch data of the earliest SpaceX launch.
    """
    launches = fetch_json("https://api.spacexdata.com/v4/launches")
    launches_sorted = sorted(launches, key=lambda x: x["date_unix"])
    return launches_sorted[0]


def format_date(unix_timestamp):
    """
    Convert a UNIX timestamp to the local timezone ISO string.

    Args:
        unix_timestamp (int): UNIX timestamp.

    Returns:
        str: Local timezone ISO formatted date string.
    """
    utc_dt = datetime.fromtimestamp(unix_timestamp, pytz.utc)
    local_dt = utc_dt.astimezone()
    return local_dt.isoformat()


def main():
    """
    Main function that gathers launch, rocket, and launchpad data,
    formats them, and prints the requested output.
    """
    launch = get_first_launch()

    rocket = fetch_json(
        f"https://api.spacexdata.com/v4/rockets/{launch['rocket']}"
    )
    launchpad = fetch_json(
        f"https://api.spacexdata.com/v4/launchpads/{launch['launchpad']}"
    )

    date_local = format_date(launch["date_unix"])

    output = (
        f"{launch['name']} ({date_local}) "
        f"{rocket['name']} - {launchpad['name']} "
        f"({launchpad['locality']})"
    )

    print(output)


if __name__ == "__main__":
    main()
