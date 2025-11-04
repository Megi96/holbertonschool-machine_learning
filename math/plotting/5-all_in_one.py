#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt


def all_in_one():
    # ----- Task 1: line -----
    y0 = np.arange(0, 11) ** 3

    # ----- Task 2: scatter -----
    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    np.random.seed(5)
    x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
    y1 += 180

    # ----- Task 3: change_scale -----
    x2 = np.arange(0, 28651, 5730)
    r2 = np.log(0.5)
    t2 = 5730
    y2 = np.exp((r2 / t2) * x2)

    # ----- Task 4: two -----
    x3 = np.arange(0, 21000, 1000)
    r3 = np.log(0.5)
    t31 = 5730
    t32 = 1600
    y31 = np.exp((r3 / t31) * x3)
    y32 = np.exp((r3 / t32) * x3)

    # ----- Task 5: frequency -----
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)

    # ----- Combine all -----
    fig = plt.figure(figsize=(8, 6))
    fig.suptitle('All in One')

    # Plot 1 - line
    plt.subplot(3, 2, 1)
    plt.plot(y0, color='red')
    plt.xlim(0, 10)
    plt.title('Cube Numbers', fontsize='x-small')

    # Plot 2 - scatter (Height vs Weight)
    plt.subplot(3, 2, 2)
    plt.scatter(x1, y1, color='magenta')
    plt.title("Men's Height vs Weight", fontsize='x-small')
    plt.xlabel("Height (in)", fontsize='x-small')
    plt.ylabel("Weight (lbs)", fontsize='x-small')
    plt.xticks(np.arange(60, 81, 10))  # ✅ Bins every 10 units on x-axis
    plt.yticks(np.arange(170, 191, 10))

    # Plot 3 - change scale
    plt.subplot(3, 2, 3)
    plt.plot(x2, y2)
    plt.yscale('log')
    plt.xlim(x2[0], x2[-1])
    plt.title('Exponential Decay of C-14', fontsize='x-small')
    plt.xlabel('Time (years)', fontsize='x-small')
    plt.ylabel('Fraction Remaining', fontsize='x-small')
    plt.xticks([0, 10000, 20000])

    # Plot 4 - two decays
    plt.subplot(3, 2, 4)
    plt.plot(x3, y31, color='red', linestyle='dashed', label='C-14')
    plt.plot(x3, y32, color='green', label='Ra-226')
    plt.xlim(0, 20000)
    plt.ylim(0, 1)
    plt.title('Exponential Decay of Radioactive Elements', fontsize='x-small')
    plt.xlabel('Time (years)', fontsize='x-small')
    plt.ylabel('Fraction Remaining', fontsize='x-small')
    plt.legend(fontsize='x-small')

    # Plot 5 - histogram (Project A)
    plt.subplot(3, 1, 3)
    plt.hist(student_grades, bins=10, range=(0, 100), edgecolor='black')
    plt.title('Project A', fontsize='x-small')
    plt.xlabel('Grades', fontsize='x-small')
    plt.ylabel('Number of Students', fontsize='x-small')
    plt.xlim(0, 100)
    plt.ylim(0, 30)
    plt.xticks(np.arange(0, 101, 10))

    plt.tight_layout()
    plt.show()


all_in_one()
