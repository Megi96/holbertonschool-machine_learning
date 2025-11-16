#!/usr/bin/env python3
"""
Module for retrieving the names of planets that are homeworlds
of all sentient species using the SWAPI API.

A species is considered sentient if the word "sentient" appears
in either its 'classification' or 'designation' fields.

This module provides one function:
    sentientPlanets()
"""

import requests


def sentientPlanets():
    """
    Retrieve a list of homeworld names for all sentient species
    from the SWAPI species endpoint.

    A species is considered sentient if:
    - "sentient" appears in its `classification`, OR
    - "sentient" appears in its `designation`.

    The function:
    - Iterates through all pages of the species endpoint.
    - Checks for sentient species based on classification/designation.
    - Resolves each species' homeworld URL into a planet name.
    - If a species has no homeworld or the request fails, uses "unknown".
    - Ensures pagination until all pages are processed.

    Returns:
        list: A list of planet names (strings) corresponding to
              the homeworlds of all sentient species.
    """
    url = "https://swapi-api.hbtn.io/api/species/"
    planets = []

    while url:
        response = requests.get(url)
        if response.status_code != 200:
            return planets  # Fail-safe: return what we have

        data = response.json()
        species_list = data.get("results", [])

        for species in species_list:
            # Extract classification and designation
            classification = species.get("classification", "").lower()
            designation = species.get("designation", "").lower()

            # Check if species is sentient
            if "sentient" in classification or "sentient" in designation:
                homeworld_url = species.get("homeworld")

                if homeworld_url:
                    # Fetch homeworld name
                    planet_response = requests.get(homeworld_url)
                    if planet_response.status_code == 200:
                        planet_name = planet_response.json().get(
                            "name", "unknown"
                        )
                        planets.append(planet_name)
                    else:
                        planets.append("unknown")
                else:
                    planets.append("unknown")

        # Move to next page
        url = data.get("next")

    return planets
