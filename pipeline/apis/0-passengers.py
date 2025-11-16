#!/usr/bin/env python3
"""
Module for retrieving Star Wars starships that can hold
a minimum number of passengers using the SWAPI API.

This module defines a single function:
    availableShips(passengerCount)

The function queries the paginated starships endpoint of SWAPI,
filters ships by passenger capacity, and returns a list of names.
"""

import requests


def availableShips(passengerCount):
    """
    Retrieve a list of starships that can carry at least `passengerCount`
    passengers, using the SWAPI starships API with proper pagination.

    The function:
    - Iterates through all pages of the starships endpoint.
    - Cleans passenger fields (removes commas).
    - Considers only numeric passenger values.
    - Compares capacity to the given passengerCount.
    - Returns a list of matching starship names.
    - Returns an empty list if no match is found or API fails.

    Args:
        passengerCount (int): Minimum required passenger capacity.

    Returns:
        list: A list of starship names (strings) that satisfy the capacity
              requirement. Returns an empty list if none qualify.
    """
    url = "https://swapi-api.hbtn.io/api/starships/"
    result = []

    while url:
        response = requests.get(url)
        if response.status_code != 200:
            return []  # Fail-safe: invalid API response

        data = response.json()
        ships = data.get("results", [])

        for ship in ships:
            passengers = ship.get("passengers", "0")

            # Clean passengers (e.g., "843,342" → "843342")
            passengers_clean = passengers.replace(",", "")

            if passengers_clean.isdigit():
                if int(passengers_clean) >= passengerCount:
                    result.append(ship.get("name"))

        # Pagination: move to next page
        url = data.get("next")

    return result
