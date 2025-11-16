#!/usr/bin/env python3
import requests


def availableShips(passengerCount):
    """
    Returns a list of ship names that can carry at least `passengerCount`.
    Uses SWAPI starships endpoint with proper pagination.
    """
    url = "https://swapi-api.hbtn.io/api/starships/"
    result = []

    while url:
        response = requests.get(url)
        if response.status_code != 200:
            return []  # If API fails, return empty list

        data = response.json()
        ships = data.get("results", [])

        for ship in ships:
            passengers = ship.get("passengers", "0")

            # Clean passengers: remove commas, check numeric
            passengers_clean = passengers.replace(",", "")
            if passengers_clean.isdigit():
                if int(passengers_clean) >= passengerCount:
                    result.append(ship.get("name"))

        # Move to next page
        url = data.get("next")

    return result
