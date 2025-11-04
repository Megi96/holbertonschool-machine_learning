#!/usr/bin/env python3
"""Plot a histogram of student scores for Project A."""

import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """Plot a histogram showing the distribution of student grades."""
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))

    # Create histogram with bins every 10 units (0–100 inclusive)
    plt.hist(student_grades, bins=range(0, 110, 10), edgecolor='black')

    # Add labels and title
    plt.title('Project A')
    plt.xlabel('Grades')
    plt.ylabel('Number of Students')
    plt.xlim(0, 100)
    plt.ylim(0, 30)
    plt.xticks(np.arange(0, 101, step=10))
    # Display plot
    plt.show()
