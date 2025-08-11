import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

def build_cnn_model(input_shape=(64, 64, 1)):
    """
    Builds a simple CNN model for image reconstruction (denoising or enhancement).
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(np.prod(input_shape), activation='sigmoid')  # Output image shape (64x64)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def simulate_xray_data():
    """
    Simulate noisy X-ray data as input for the CNN.
    """
    noisy_image = np.random.rand(64, 64)  # Simulating noisy data
    return noisy_image

def visualize_reconstruction(original_image, reconstructed_image):
    """
    Visualizes the original and reconstructed images.
    """
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    ax[0].imshow(original_image, cmap='gray')
    ax[0].set_title("Original Noisy Image")
    
    ax[1].imshow(reconstructed_image, cmap='gray')
    ax[1].set_title("Reconstructed Image")

    plt.show()

if __name__ == "__main__":
    # Simulate a noisy X-ray image
    noisy_image = simulate_xray_data()

    # Build and train the CNN model (in a real scenario, you would train on actual data)
    cnn_model = build_cnn_model(input_shape=(64, 64, 1))
    cnn_model.fit(np.expand_dims(noisy_image, axis=-1)[np.newaxis, :], np.expand_dims(noisy_image, axis=-1)[np.newaxis, :], epochs=10)

    # Use the trained CNN to reconstruct the image
    reconstructed_image = cnn_model.predict(np.expand_dims(noisy_image, axis=-1)[np.newaxis, :])

    # Visualize the reconstruction
    visualize_reconstruction(noisy_image, reconstructed_image[0, :, :, 0])

