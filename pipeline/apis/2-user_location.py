#!/usr/bin/env python3
"""
Script that prints the location of a GitHub user using the GitHub API.

Usage:
    ./2-user_location.py https://api.github.com/users/<username>

Behaviors:
- If the user exists: print their "location" or "None"
- If status code is 404: print Not found
- If status code is 403: print "Reset in X min"
    where X is time until rate limit reset
"""

import sys
import requests
import time


def get_user_location(url):
    """Return the location of a GitHub user or error messages."""
    response = requests.get(url)

    # Handle rate-limit
    if response.status_code == 403:
        reset_ts = response.headers.get("X-RateLimit-Reset")

        if reset_ts:
            reset_ts = int(reset_ts)
            remaining = reset_ts - int(time.time())
            minutes = remaining // 60
            return f"Reset in {minutes} min"

        return "Reset in unknown time"

    # Handle 404 not found
    if response.status_code == 404:
        return "Not found"

    # Handle success
    if response.status_code == 200:
        data = response.json()
        return data.get("location")

    # Other unexpected statuses
    return "Error"


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: ./2-user_location.py <GitHub API user URL>")
        sys.exit(1)

    url = sys.argv[1]
    result = get_user_location(url)
    print(result)
