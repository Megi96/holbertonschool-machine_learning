import numpy as np
import matplotlib.pyplot as plt


def bars():
    """Plot a stacked bar chart of fruits per person."""
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
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


bars()
