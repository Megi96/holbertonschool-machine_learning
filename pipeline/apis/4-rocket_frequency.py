#!/usr/bin/env python3
"""
Number of launches per rocket using SpaceX API
"""
import requests


if __name__ == '__main__':
    # Fetch all launches
    url = "https://api.spacexdata.com/v4/launches"
    response = requests.get(url)
    launches = response.json()

    # Dictionary to store rocket ID and count
    rocket_counts = {}

    # Count launches per rocket ID
    for launch in launches:
        rocket_id = launch.get('rocket')
        if rocket_id:
            rocket_counts[rocket_id] = rocket_counts.get(rocket_id, 0) + 1

    # Fetch rocket names
    rocket_names = {}
    rockets_url = "https://api.spacexdata.com/v4/rockets"
    rockets_response = requests.get(rockets_url)
    rockets = rockets_response.json()

    for rocket in rockets:
        rocket_names[rocket['id']] = rocket['name']

    # Create list of (name, count) tuples
    results = []
    for rocket_id, count in rocket_counts.items():
        name = rocket_names.get(rocket_id, "Unknown")
        results.append((name, count))

    # Sort by count (descending), then by name (ascending)
    results.sort(key=lambda x: (-x[1], x[0]))

    # Print results
    for name, count in results:
        print("{}: {}".format(name, count))
