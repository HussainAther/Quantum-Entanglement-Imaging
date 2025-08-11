import numpy as np
import matplotlib.pyplot as plt

def tissue_interaction_simulation(tissue_types=["Bone", "Tissue", "Air"], num_layers=50):
    """
    Simulate the interaction of X-rays with different types of tissues.
    Each tissue has a different attenuation coefficient.
    """
    # Tissue attenuation coefficients (arbitrary values for this example)
    tissue_attenuation = {
        "Bone": 0.3,  # High attenuation
        "Tissue": 0.1,  # Moderate attenuation
        "Air": 0.01   # Low attenuation
    }

    # Initialize an empty array to store the attenuation values for each layer
    attenuation_values = []

    for tissue in tissue_types:
        attenuation = tissue_attenuation[tissue]
        intensities = np.zeros(num_layers)
        intensities[0] = 1000  # Initial intensity

        for i in range(1, num_layers):
            # Apply attenuation for each tissue type using an exponential decay model
            intensities[i] = intensities[i-1] * np.exp(-attenuation)
        attenuation_values.append(intensities)

    return attenuation_values

def visualize_interaction(attenuation_values, tissue_types):
    """
    Visualizes the X-ray intensity through different tissues.
    """
    plt.figure(figsize=(10, 6))

    for i, tissue in enumerate(tissue_types):
        plt.plot(attenuation_values[i], label=f"{tissue} Attenuation")

    plt.title("X-ray Intensity through Different Tissues")
    plt.xlabel("Layer Number")
    plt.ylabel("Intensity")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Simulate X-ray interaction with different tissues
    attenuation_values = tissue_interaction_simulation()

    # Visualize the results
    visualize_interaction(attenuation_values, tissue_types=["Bone", "Tissue", "Air"])

