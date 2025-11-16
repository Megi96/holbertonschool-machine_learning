#!/usr/bin/env python3
"""
Retrieve the names of planets that are homeworlds of sentient species
from the SWAPI API.

A species is considered sentient if the word "sentient" appears in either
its 'classification' or 'designation' fields.
"""

import requests


def sentientPlanets():
    """
    Return a list of homeworld names for all sentient species.

    Sentience is detected when "sentient" appears in either the
    species' classification or designation. Species without a valid
    homeworld (or where the homeworld cannot be resolved) are skipped.

    Returns:
        list: planet names (strings) for sentient species' homeworlds.
    """
    url = "https://swapi-api.hbtn.io/api/species/"
    planets = []

    while url:
        resp = requests.get(url)
        if resp.status_code != 200:
            return planets

        data = resp.json()
        for species in data.get("results", []):
            classification = species.get("classification", "").lower()
            designation = species.get("designation", "").lower()

            is_sentient_class = "sentient" in classification
            is_sentient_design = "sentient" in designation

            if not is_sentient_class and not is_sentient_design:
                continue

            homeworld_url = species.get("homeworld")
            if not homeworld_url:
                # Skip species without a homeworld instead of adding "unknown"
                continue

            # Correct indentation starts here
            planet_resp = requests.get(homeworld_url)
            if planet_resp.status_code != 200:
                continue

            planet_name = planet_resp.json().get("name")
            if planet_name:
                planets.append(planet_name)

        url = data.get("next")

    return planets
