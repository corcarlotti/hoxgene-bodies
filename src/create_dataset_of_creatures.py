import bpy
import random
import math
import json
import os
from datetime import date

# Parameters
num_creatures = 1000  # Number of creatures to generate
num_genes = 2  # Number of Hox genes
segment_base_length = 1.0
segment_base_radius = 0.5
randomness_factor = 0.4  # Factor for random adjustments
connector_radius = 0.1  # Radius for the connecting cylinder

# Clear existing objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# Prepare to save creatures data
creatures_data = []

def add_segment(position, rotation, segment_length, segment_radius, shape_type):
    """Create a segment at the specified position based on shape type."""
    if shape_type == 'cylinder':
        bpy.ops.mesh.primitive_cylinder_add(radius=segment_radius, depth=segment_length, 
                                            location=position, rotation=rotation)
    elif shape_type == 'cube':
        bpy.ops.mesh.primitive_cube_add(size=segment_radius * 2, location=position)
    else:  # sphere
        bpy.ops.mesh.primitive_uv_sphere_add(radius=segment_radius, location=position)

    segment = bpy.context.object
    segment.name = f"Segment_{shape_type.capitalize()}"

    # Optional: Add material
    material = bpy.data.materials.new(name=f"Material_{segment.name}")
    material.diffuse_color = (random.random(), random.random(), random.random(), 1.0)
    segment.data.materials.append(material)

    return {
        "type": shape_type,
        "length": segment_length,
        "radius": segment_radius,
        "position": position,
        "rotation": rotation
    }

def add_connector(start, end, rotation):
    """Add a thin cylinder to bridge a gap between two points."""
    connector_length = math.dist(start, end)
    midpoint = [(start[i] + end[i]) / 2 for i in range(3)]
    
    # Add the connector with appropriate rotation
    bpy.ops.mesh.primitive_cylinder_add(radius=connector_radius, depth=connector_length, 
                                        location=midpoint, rotation=rotation)
    connector = bpy.context.object
    connector.name = "Connector"

    return {
        "start": start,
        "end": end,
        "rotation": rotation,
        "length": connector_length
    }

# Loop through to create multiple creatures
for creature_index in range(num_creatures):
    hox_genes = [random.choice([0, 1]) for _ in range(num_genes)]  # Random Hox gene activation
    x_position, y_position, z_position = 0.0, 0.0, 0.0
    last_position = (x_position, y_position, z_position)
    last_radius = segment_base_radius

    segments = []
    connectors = []

    # Loop through the array and add segments for each active (1) gene
    for i, gene_active in enumerate(hox_genes):
        if gene_active:
            # Randomize segment properties
            segment_length = segment_base_length * (1 + randomness_factor * random.uniform(-1, 1))
            segment_radius = segment_base_radius * (1 + randomness_factor * random.uniform(-0.5, 0.5))
            
            # Random orientation or branching based on gene position
            orientation_choice = random.choice(["x", "y", "z"])
            shape_type = random.choice(['cylinder', 'cube', 'sphere'])  # Randomly choose shape

            # Adjust position based on orientation
            if orientation_choice == "x":
                rotation = (0, math.radians(90), 0)
                new_position = (x_position + segment_length / 2, y_position, z_position)
                x_position += segment_length
            elif orientation_choice == "y":
                rotation = (math.radians(90), 0, 0)
                new_position = (x_position, y_position + segment_length / 2, z_position)
                y_position += segment_length
            else:
                rotation = (0, 0, 0)
                new_position = (x_position, y_position, z_position + segment_length / 2)
                z_position += segment_length

            # Create the segment and store its data
            segment_data = add_segment(new_position, rotation, segment_length, segment_radius, shape_type)
            segments.append(segment_data)

            # Calculate the exact distance between segment edges
            edge_gap_distance = math.dist(last_position, new_position) - (last_radius + segment_radius)
            
            # Add connector only if there's a significant gap based on surface distance
            if i > 0 and edge_gap_distance > 0.01:
                # Determine rotation for connector based on position change
                if orientation_choice == "x":
                    connector_rotation = (0, math.radians(90), 0)
                elif orientation_choice == "y":
                    connector_rotation = (math.radians(90), 0, 0)
                else:
                    connector_rotation = (0, 0, 0)

                connector_data = add_connector(last_position, new_position, connector_rotation)
                connectors.append(connector_data)

            # Update the last position and radius for gap calculations
            last_position = new_position
            last_radius = segment_radius

    # Store creature data
    creatures_data.append({
        "hox_genes": ''.join(map(str, hox_genes)),
        "model_data": {
            "segments": segments,
            "connectors": connectors
        }
    })

    # Delete the objects to save memory
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    # Optional: Save the data to a JSON file after each creature
    now = datetime.now()
    name = 'hox' + now
    output_json_path = "/Users/roschkach/Projekte/BCA/{name}/Dataset/creatures_data_2_hoxgenes.json"  # Update this path as necessary
    with open(output_json_path, "w") as outfile:
        json.dump(creatures_data, outfile, indent=4)

    config_file_path = "/Users/roschkach/Projekte/BCA/{name}/Dataset/creatures_data_2_hoxgenes.json" 


print(f"{num_creatures} creatures generated and saved to {output_json_path}!")
