#!/usr/bin/env python3
"""
Script to display the first SpaceX launch using the unofficial SpaceX API.

Output format:
<launch name> (<date>) <rocket name> - <launchpad name> (<launchpad locality>)
"""

import requests
from datetime import datetime
import pytz


def get_first_launch():
    """
    Fetches and displays the first SpaceX launch with formatted details:
    - Launch name
    - Local date and time
    - Rocket name
    - Launchpad name and locality
    """
    # Fetch all launches
    launches_url = "https://api.spacexdata.com/v4/launches"
    launches = requests.get(launches_url).json()

    # Sort launches by date_unix
    launches.sort(key=lambda x: x['date_unix'])

    # Select the first launch
    first = launches[0]

    # Fetch rocket details
    rocket_id = first['rocket']
    rocket_url = f"https://api.spacexdata.com/v4/rockets/{rocket_id}"
    rocket = requests.get(rocket_url).json()
    rocket_name = rocket['name']

    # Fetch launchpad details
    launchpad_id = first['launchpad']
    launchpad_url = f"https://api.spacexdata.com/v4/launchpads/{launchpad_id}"
    launchpad = requests.get(launchpad_url).json()
    launchpad_name = launchpad['name']
    launchpad_locality = launchpad['locality']

    # Convert UTC date to local time
    utc_time = datetime.fromisoformat(first['date_utc'].replace("Z", "+00:00"))
    local_time = utc_time.astimezone().isoformat()

    # Print formatted output
    print(f"{first['name']} ({local_time}) {rocket_name} - "
          f"{launchpad_name} ({launchpad_locality})")


if __name__ == '__main__':
    get_first_launch()
