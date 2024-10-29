import tensorflow as tf
import numpy as np
import json
import math

from tensorflow.keras.layers import Layer
import keras

# Define the Sampling layer
class Sampling(Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Load the model using custom objects
encoder = keras.models.load_model('/Users/roschkach/Projekte/BCA/vae/vae_encoder_1.h5', 
                                  custom_objects={'Sampling': Sampling}, compile=False)
decoder = keras.models.load_model('/Users/roschkach/Projekte/BCA/vae/vae_decoder_1.h5', 
                                  custom_objects={'Sampling': Sampling}, compile=False)


# Function to decode segment type from one-hot vector
def decode_type(one_hot_vector):
    segment_types = ['cylinder', 'sphere', 'cube']
    type_index = np.argmax(one_hot_vector)
    return segment_types[type_index]

def generate_3d_model_from_gene(gene_sequence):
    # Encode the gene sequence to obtain the latent vector
    _, _, z = encoder.predict(gene_sequence)

    # Decode the latent vector to predict 3D model properties
    predicted_3d_model = decoder.predict(z)

    # Initialize output JSON structure
    output_model = {
        "hox_genes": ''.join(str(int(g)) for g in gene_sequence[0]),  # Convert gene sequence to string
        "model_data": {
            "segments": [],
            "connectors": []
        }
    }

    # Process segments for active genes
    active_segment_index = 0  # Keep track of active segments in predicted model data
    for i, gene in enumerate(gene_sequence[0]):  # Iterate over gene sequence
        if gene == 1:  # Only add a segment if gene is active (1)
            # Decode segment properties based on 11 features per segment
            length = predicted_3d_model[0][active_segment_index * 11]
            radius = predicted_3d_model[0][active_segment_index * 11 + 1]
            position = [
                predicted_3d_model[0][active_segment_index * 11 + 2],
                predicted_3d_model[0][active_segment_index * 11 + 3],
                predicted_3d_model[0][active_segment_index * 11 + 4]
            ]
            rotation = [
                predicted_3d_model[0][active_segment_index * 11 + 5],
                predicted_3d_model[0][active_segment_index * 11 + 6],
                predicted_3d_model[0][active_segment_index * 11 + 7]
            ]
            # Decode type from one-hot vector
            type_one_hot = predicted_3d_model[0][active_segment_index * 11 + 8: active_segment_index * 11 + 11]
            segment_type = decode_type(type_one_hot)

            # Add segment information to output
            segment = {
                "type": segment_type,
                "length": float(length),
                "radius": float(radius),
                "position": [float(p) for p in position],
                "rotation": [float(r) for r in rotation]
            }
            output_model["model_data"]["segments"].append(segment)

            # Move to the next active segment
            active_segment_index += 1

    # Process connectors based on remaining data in predicted_3d_model
    num_connectors = active_segment_index - 1
    for j in range(num_connectors):
        # Each connector contributes 10 features in the output vector
        offset = len(gene_sequence) * 11 + j * 10  # Offset to start of connector features

        length = predicted_3d_model[0][offset]
        start = [
            predicted_3d_model[0][offset + 1],
            predicted_3d_model[0][offset + 2],
            predicted_3d_model[0][offset + 3]
        ]
        end = [
            predicted_3d_model[0][offset + 4],
            predicted_3d_model[0][offset + 5],
            predicted_3d_model[0][offset + 6]
        ]
        rotation = [
            predicted_3d_model[0][offset + 7],
            predicted_3d_model[0][offset + 8],
            predicted_3d_model[0][offset + 9]
        ]

        connector = {
            "start": start,
            "end": end,
            "rotation": rotation,
            "length": float(length)
        }
        output_model["model_data"]["connectors"].append(connector)
    
    return output_model


# Convert float32 values for JSON compatibility
def convert_float32(obj):
    if isinstance(obj, np.float32):
        return float(obj)
    elif isinstance(obj, (list, tuple)):
        return [convert_float32(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: convert_float32(v) for k, v in obj.items()}
    elif isinstance(obj, np.ndarray):
        return convert_float32(obj.tolist())
    else:
        return obj

# Example Hox gene sequence for 3D model generation
new_gene_sequence = np.array([[1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1]])  # Example input

# Generate the model data and convert to JSON format
output_model = generate_3d_model_from_gene(new_gene_sequence)
output_model_converted = convert_float32(output_model)


# Save to JSON file
with open('/Users/roschkach/Projekte/BCA/generated_model_from_gene_7.json', 'w') as f:
# Write the opening bracket
    f.write('[')
    json.dump(output_model_converted, f)
    f.write(']')

print("Generated 3D model saved to 'generated_model_from_gene_7.json'")
