#!/usr/bin/env python3
"""
6-bars.py

This module plots a stacked bar chart representing the number of fruits
each person (Farrah, Fred, Felicia) possesses. The bars are stacked
in the order: apples, bananas, oranges, peaches.
Colors and labels follow the specifications.
"""

import numpy as np
import matplotlib.pyplot as plt


def bars():
    """
    Plot a stacked bar chart of fruits per person.

    Uses a fixed dataset to match the reference plot exactly.

    Stacking order:
        apples -> red
        bananas -> yellow
        oranges -> #ff8000
        peaches -> #ffe5b4

    X-axis labels: Farrah, Fred, Felicia
    Y-axis: 0 to 80, ticks every 10
    Bar width: 0.5
    Legend included
    """
    # Fixed dataset matching the reference
    fruit = np.array([[12, 7, 5],   # apples
                      [5, 10, 15],  # bananas
                      [8, 5, 7],    # oranges
                      [3, 2, 5]])   # peaches

    names = ['Farrah', 'Fred', 'Felicia']
    colors = ['red', 'yellow', '#ff8000', '#ffe5b4']
    labels = ['Apples', 'Bananas', 'Oranges', 'Peaches']

    bottom = np.zeros(3)
    for i in range(fruit.shape[0]):
        plt.bar(names, fruit[i], bottom=bottom,
                color=colors[i], label=labels[i], width=0.5)
        bottom += fruit[i]

    plt.title("Number of Fruit per Person")
    plt.ylabel("Quantity of Fruit")
    plt.ylim(0, 80)
    plt.yticks(np.arange(0, 81, 10))
    plt.legend()
    plt.show()


if __name__ == '__main__':
    bars()
