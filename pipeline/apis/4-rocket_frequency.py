#!/usr/bin/env python3
"""
Script that displays the number of launches per rocket.
"""

import requests


if __name__ == "__main__":
    # 1. Get all launches
    launches = requests.get(
        "https://api.spacexdata.com/v4/launches"
    ).json()

    # Count rocket occurrences
    rocket_counts = {}

    for launch in launches:
        rocket_id = launch.get("rocket")
        if rocket_id:
            rocket_counts[rocket_id] = rocket_counts.get(rocket_id, 0) + 1

    # Replace rocket_id with rocket name
    rocket_names = {}
    for rocket_id in rocket_counts:
        rocket_data = requests.get(
            f"https://api.spacexdata.com/v4/rockets/{rocket_id}"
        ).json()
        name = rocket_data.get("name")
        rocket_names[rocket_id] = name

    # Build final list of (name, count)
    final = []
    for rocket_id, count in rocket_counts.items():
        final.append((rocket_names[rocket_id], count))

    # Sort:
    # 1) by count DESC
    # 2) by name ASC for ties
    final.sort(key=lambda x: (-x[1], x[0]))

    # Print
    for name, count in final:
        print(f"{name}: {count}")
