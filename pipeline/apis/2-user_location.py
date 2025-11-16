#!/usr/bin/env python3
"""
Script that prints the location of a GitHub user using the GitHub API.

Usage:
    ./2-user_location.py https://api.github.com/users/<username>

If the user does not exist, print "Not found".
If rate limited (403), print: "Reset in X min".
"""

import sys
import requests
from datetime import datetime


def get_location(url):
    """Retrieve and return the user's location or status message."""
    try:
        response = requests.get(url)
    except Exception:
        return "Not found"

    # Rate limit reached
    if response.status_code == 403:
        reset_timestamp = response.headers.get("X-RateLimit-Reset")
        if reset_timestamp:
            reset_time = datetime.fromtimestamp(int(reset_timestamp))
            now = datetime.now()
            minutes = int((reset_time - now).total_seconds() // 60)
            return f"Reset in {minutes} min"
        return "Reset in 0 min"

    # User not found
    if response.status_code == 404:
        return "Not found"

    # Normal response
    if response.status_code == 200:
        data = response.json()
        location = data.get("location")

        if not location:
            return "Not found"

        # Clean location: only first part before comma
        clean_location = location.split(",")[0].strip()
        return clean_location

    # For any other unexpected status
    return "Not found"


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(1)

    url = sys.argv[1]
    result = get_location(url)
    print(result)
