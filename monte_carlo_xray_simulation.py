import numpy as np
import matplotlib.pyplot as plt

# Constants
NUM_PHOTONS = 10000        # Number of photons to simulate
ABSORPTION_PROBABILITY = 0.05  # 5% chance of photon absorption
SCATTER_ANGLE_RANGE = (0, 180)  # Scatter angles in degrees

def monte_carlo_xray_simulation():
    """
    Simulates the scattering and absorption of X-ray photons using a Monte Carlo method.
    Returns the photon positions, scatter angles, and whether the photon was absorbed.
    """
    # Photon positions (3D space)
    photon_positions = np.random.rand(NUM_PHOTONS, 3) * 10  # Assume a 10x10x10 cubic volume

    # Scattering angles (random angles between 0 and 180 degrees)
    scatter_angles = np.random.uniform(SCATTER_ANGLE_RANGE[0], SCATTER_ANGLE_RANGE[1], NUM_PHOTONS)

    # Absorption (random decision for each photon)
    absorbed_photons = np.random.rand(NUM_PHOTONS) < ABSORPTION_PROBABILITY  # 5% absorption

    # Return the results
    return photon_positions, scatter_angles, absorbed_photons

def visualize_photon_scattering(photon_positions, scatter_angles):
    """
    Visualizes the photon positions and scatter angles.
    """
    # Plot photon positions in 3D space
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(photon_positions[:, 0], photon_positions[:, 1], photon_positions[:, 2], c='blue', marker='o', alpha=0.5)

    # Plot scatter angles (for simplicity, plot as a histogram)
    plt.figure()
    plt.hist(scatter_angles, bins=20, edgecolor='black')
    plt.title('Scatter Angle Distribution')
    plt.xlabel('Scatter Angle (Degrees)')
    plt.ylabel('Frequency')
    plt.show()

if __name__ == "__main__":
    # Run the simulation
    photon_positions, scatter_angles, absorbed_photons = monte_carlo_xray_simulation()

    # Print statistics about the simulation
    print(f"Total number of photons: {NUM_PHOTONS}")
    print(f"Number of absorbed photons: {np.sum(absorbed_photons)}")
    print(f"Percentage of absorbed photons: {np.sum(absorbed_photons) / NUM_PHOTONS * 100:.2f}%")

    # Visualize the photon scattering
    visualize_photon_scattering(photon_positions, scatter_angles)

