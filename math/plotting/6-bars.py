#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt


def bars():
    """Plot a stacked bar chart of fruits per person."""
    fruit = np.array([[12, 7, 5],   # apples
                      [5, 10, 15],  # bananas
                      [8, 5, 7],    # oranges
                      [3, 2, 5]])   # peaches

    names = ['Farrah', 'Fred', 'Felicia']
    fs = ['Apples', 'Bananas', 'Oranges', 'Peaches']
    colors = ['red', 'yellow', '#ff8000', '#ffe5b4']

    bottom = np.zeros(3)
    for i in range(len(fruit)):
        plt.bar(
            names,
            fruit[i],
            bottom=bottom,
            color=colors[i],
            label=fs[i],
            width=0.5
        )
        bottom += fruit[i]

    plt.title("Number of Fruit per Person")
    plt.ylabel("Quantity of Fruit")
    plt.yticks(np.arange(0, 81, 10))
    plt.ylim(0, 80)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    bars()
