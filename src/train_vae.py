import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
import numpy as np
import json

import json
from sklearn.model_selection import train_test_split

# Load the dataset
with open("/Users/roschkach/Projekte/BCA/Dataset/creatures_data.json", "r") as infile:
    dataset = json.load(infile)

# Extract features and labels
X = [item["hox_genes"] for item in dataset]  # Features: Hox gene sequences
Y = [item["model_data"] for item in dataset]  # Labels: 3D model data

# Split into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

X_train = np.array(X_train)
Y_train = np.array(Y_train)
#print(Y_train[0])

print("First 20 elements of X_train:", X_train[:20])
print("Total length of X_train:", len(X_train))

# Convert each string to a list of integers
X_train_numeric = np.array([[int(char) for char in gene_seq] for gene_seq in X_train])

# Check the shape
print("Converted X_train shape:", X_train_numeric.shape)  # Should be (800, 13)

# Define possible segment types
segment_types = ['cylinder', 'sphere', 'cube']

# Assuming `X_train` is an array where each row is a Hox gene sequence
max_segments = X_train_numeric.shape[1]  # Max segments in the dataset
max_connectors = max_segments - 1  # One connector less than segments


def convert_model_to_vector(model_data, max_segments, max_connectors):
    """Convert a single model's data into a flattened vector with padding to ensure consistent length."""
    segments = model_data['segments']
    connectors = model_data['connectors']
    
    # Each segment: length (1), radius (1), position (3), type (3 one-hot encoded) = 11 features
    features_segments = 11
    # Each connector: start position (3), end position (3), rotation (3), length (1) = 10 features
    features_connectors = 10

    # Initialize the vector with zeros for padding
    vector = np.zeros((max_segments * 11) + (max_connectors * 10))

    # Process segments
    # Process segments
    for i, segment in enumerate(segments[:max_segments]):
        # Segment length, radius, position, and rotation
        vector[i * 11] = segment['length']
        vector[i * 11 + 1] = segment['radius']
        vector[i * 11 + 2:i * 11 + 5] = segment['position']
        vector[i * 11 + 5:i * 11 + 8] = segment['rotation']
        
        # One-hot encode the type
        type_index = segment_types.index(segment['type'])
        vector[i * 11 + 8 + type_index] = 1  # Set one-hot encoding for segment type

    # Process connectors as before
    for j, connector in enumerate(connectors[:max_connectors]):
        vector[max_segments * 11 + j * 10] = connector['length']
        vector[max_segments * 11 + j * 10 + 1: max_segments * 11 + j * 10 + 4] = connector['start']
        vector[max_segments * 11 + j * 10 + 4: max_segments * 11 + j * 10 + 7] = connector['end']
        vector[max_segments * 11 + j * 10 + 7: max_segments * 11 + j * 10 + 10] = connector['rotation']


    return vector

# Convert all Y_train data
Y_train_numeric = np.array([convert_model_to_vector(model, max_segments, max_connectors) for model in Y_train])

# Check the shape of Y_train_numeric for debugging
print(f"Y_train_numeric shape: {Y_train_numeric.shape}")

# Sampling layer for the reparameterization trick
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Latent space dimensions
latent_dim = 16  # Adjust latent dimensions

# Encoder model
gene_input = layers.Input(shape=(X_train_numeric.shape[1],))    # Input for Hox genes
x = layers.Dense(64, activation='relu')(gene_input)
x = layers.Dense(32, activation='relu')(x)

# Latent space mean and variance
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])  # Sample latent vector

# Build encoder
encoder = models.Model(gene_input, [z_mean, z_log_var, z], name="encoder")
encoder.summary()

# Calculate the total output size
num_samples = X_train_numeric.shape[1]
#max_segments = len(Y_train[0]['segments'])  # Number of segments (equal to the number of genes)
#max_connectors = max(0, max_segments - 1)   # Connectors are one less than segments
total_output_size = (max_segments * 11) + (max_connectors * 10)

print("Expected total_output_size:", total_output_size)
print("First vector length in Y_train_numeric:", len(Y_train_numeric[0]))
print("Actual length of Y_train_numeric vectors:", Y_train_numeric.shape[1])
print(Y_train_numeric[1])

# Decoder network
latent_inputs = layers.Input(shape=(latent_dim,))
x = layers.Dense(64, activation="relu")(latent_inputs)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dense(256, activation="relu")(x)

# Output layer with the size matching Y_train_numeric
decoder_outputs = layers.Dense(total_output_size, activation="sigmoid")(x)

# Build decoder
decoder = models.Model(latent_inputs, decoder_outputs, name="decoder")
decoder.summary()

# VAE model
class VAE(models.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        self.add_loss(kl_loss)  # Add KL loss to the total loss
        return reconstructed

# Instantiate and compile VAE
vae = VAE(encoder, decoder)
vae.compile(optimizer='adam', loss='mse')

# Train the VAE
vae.fit(X_train_numeric, Y_train_numeric, epochs=50, batch_size=16)


# Save the encoder and decoder models
encoder.save('/Users/roschkach/Projekte/BCA/vae/vae_encoder_1.h5')
decoder.save('/Users/roschkach/Projekte/BCA/vae/vae_decoder_2.h5')

print("VAE training completed and models saved!")
