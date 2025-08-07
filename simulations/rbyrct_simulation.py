import numpy as np
import matplotlib.pyplot as plt

def rbyct_simulation(num_rays=1000):
    """
    Simulates the interaction of X-ray beams passing through a material using Ray-by-Ray Computed Tomography (RBYCT).
    Returns the final X-ray image (after passing through tissue layers).
    """
    # Simulate X-ray rays passing through a 2D grid representing tissue
    tissue_density = np.random.rand(100, 100)  # Simulating tissue density (0: low density, 1: high density)
    rays = np.random.rand(num_rays, 2) * 100  # Simulating 1000 random rays in a 100x100 grid

    # For each ray, calculate the density it passes through (simplified model)
    xray_image = np.zeros((100, 100))
    for ray in rays:
        x, y = int(ray[0]), int(ray[1])
        xray_image[x, y] += tissue_density[x, y]  # Accumulate the tissue density along the ray path
    
    return xray_image

def visualize_xray_image(xray_image):
    """
    Visualizes the X-ray image simulated by RBYCT.
    """
    plt.imshow(xray_image, cmap='gray')
    plt.colorbar(label='X-ray Intensity')
    plt.title("Simulated X-ray Image (RBYCT)")
    plt.show()

if __name__ == "__main__":
    # Simulate the RBYCT process
    xray_image = rbyct_simulation()

    # Visualize the simulated X-ray image
    visualize_xray_image(xray_image)

