#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt


def bars():
    """Plot a stacked bar chart of fruits per person with specifications."""
    fruit = np.array([[12, 7, 5],   # apples
                      [5, 10, 15],  # bananas
                      [8, 5, 7],    # oranges
                      [3, 2, 5]])   # peaches
    plt.figure(figsize=(6.4, 4.8))

    people = ['Farrah', 'Fred', 'Felicia']
    colors = ['red', 'yellow', '#ff8000', '#ffe5b4']
    fruit_labels = ['Apples', 'Bananas', 'Oranges', 'Peaches']

    bottom = np.zeros(3)

    for i in range(fruit.shape[0]):
        plt.bar(
            people,
            fruit[i],
            bottom=bottom,
            color=colors[i],
            width=0.5,
            label=fruit_labels[i]
        )
        bottom += fruit[i]

    plt.ylabel('Quantity of Fruit')
    plt.ylim(0, 80)
    plt.yticks(np.arange(0, 81, 10))
    plt.title('Number of Fruit per Person')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    bars()
