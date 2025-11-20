#!/usr/bin/env python3
"""
Display number of launches per rocket using the SpaceX API.
"""

import requests

def rocket_launch_frequency():
    """
    Count successful past launches per rocket.
    Matches Holberton's expected output.
    """
    # Get list of all launches
    launches = requests.get("https://api.spacexdata.com/v4/launches").json()
    # Get list of all rockets
    rockets = requests.get("https://api.spacexdata.com/v4/rockets").json()

    # Map rocket ID to rocket name
    rocket_names = {r["id"]: r["name"] for r in rockets}

    freq = {}

    for launch in launches:
        # Count only successful past launches
        if (
            launch.get("success") is True
            and launch.get("upcoming") is False
            and launch.get("rocket") in rocket_names
        ):
            rid = launch["rocket"]
            freq[rid] = freq.get(rid, 0) + 1

    # Create list of (rocket name, count)
    result = [(rocket_names[rid], count) for rid, count in freq.items()]

    # Sort by count descending, then name ascending
    result.sort(key=lambda x: (-x[1], x[0]))

    # Print results
    for name, count in result:
        print(f"{name}: {count}")

if __name__ == "__main__":
    rocket_launch_frequency()
