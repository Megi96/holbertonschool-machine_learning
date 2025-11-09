import numpy as np
import matplotlib.pyplot as plt

def gradient():
    """Displays a scatter plot of sampled elevations on a mountain."""
    # Set random seed for reproducibility
    np.random.seed(5)

    # Generate random x and y coordinates (representing positions on a map)
    x = np.random.randn(2000) * 10
    y = np.random.randn(2000) * 10

    # Compute elevation (z) based on distance from the origin + random noise
    z = np.random.rand(2000) + 40 - np.sqrt(np.square(x) + np.square(y))

    # Create a new figure window with a standard size
    plt.figure(figsize=(6.4, 4.8))

    # Create a scatter plot where color represents elevation
    scatter = plt.scatter(x, y, c=z, cmap='viridis')

    # Add title and axis labels
    plt.title('Mountain Elevation')
    plt.xlabel('x coordinate (m)')
    plt.ylabel('y coordinate (m)')

    # Add colorbar to show what colors mean in terms of elevation
    cbar = plt.colorbar(scatter)
    cbar.set_label('elevation (m)')

    # Display the plot
    plt.show()
