#!/usr/bin/env python3
"""
Fetch and display the first SpaceX launch with detailed information.

Requirements:
- Use the unofficial SpaceX API
- Display:
    <launch name> (<date>) <rocket name> - <launchpad name> (<launchpad locality>)
- Date must be converted to local timezone
- Must sort launches by 'date_unix'
- Must not run on import
"""

import requests
from datetime import datetime
import pytz


def get_first_launch():
    """
    Retrieves and prints the first SpaceX launch ever recorded.

    Fetches:
    - Launch name
    - Local date
    - Rocket name
    - Launchpad name and locality
    """
    # Fetch all launches
    launches = requests.get(
        "https://api.spacexdata.com/v4/launches"
    ).json()

    # Sort by Unix timestamp (ascending → earliest first)
    launches.sort(key=lambda x: x.get("date_unix", float('inf')))

    first = launches[0]

    # Extract fields
    launch_name = first.get("name")
    rocket_id = first.get("rocket")
    launchpad_id = first.get("launchpad")
    date_utc = first.get("date_utc")

    # Convert UTC to local timezone
    utc_time = datetime.fromisoformat(date_utc.replace("Z", "+00:00"))
    local_tz = pytz.timezone("Europe/Tirane")
    local_date = utc_time.astimezone(local_tz).isoformat()

    # Fetch rocket details
    rocket = requests.get(
        f"https://api.spacexdata.com/v4/rockets/{rocket_id}"
    ).json()
    rocket_name = rocket.get("name")

    # Fetch launchpad details
    launchpad = requests.get(
        f"https://api.spacexdata.com/v4/launchpads/{launchpad_id}"
    ).json()
    launchpad_name = launchpad.get("name")
    locality = launchpad.get("locality")

    # Final formatted output
    print(
        f"{launch_name} ({local_date}) {rocket_name} - "
        f"{launchpad_name} ({locality})"
    )


if __name__ == "__main__":
    get_first_launch()
