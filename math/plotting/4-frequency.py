#!/usr/bin/env python3
"""Plot a histogram of student scores for Project A."""

import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """Plot a histogram showing the distribution of student grades."""
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))

    # Create histogram
    plt.hist(student_grades, bins=range(0, 101, 10), edgecolor='black')

    # Add labels and title
    plt.xlabel('Grades')
    plt.ylabel('Number of Students')
    plt.title('Project A')

    # Display plot
    plt.show()
