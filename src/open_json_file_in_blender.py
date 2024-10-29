import bpy
import json
import math

print("Starting script...")
# Path to your JSON file
json_file_path = "/Users/roschkach/Projekte/BCA/Dataset/hox_dataset_1.json"  # Change this to your JSON file path

# Clear existing objects in the scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

try:
    # Check if Blender can access the file
    print("Attempting to open JSON file...")
    
    # Load the JSON data
    with open(json_file_path, "r") as infile:
        data = json.load(infile)
    print("JSON file loaded successfully.")

    # Debugging: print the loaded JSON data
    import json  # Make sure to import json at the beginning if not already done
    print(json.dumps(data, indent=4))  # Pretty-print the JSON data

    # Access model_data from the first element of the list
    model_data = data[0]["model_data"]

    # Function to create a segment
    def create_segment(position, rotation, segment_length, segment_radius, shape_type):
        if shape_type == 'cylinder':
            bpy.ops.mesh.primitive_cylinder_add(radius=segment_radius, depth=segment_length,
                                                location=position, rotation=rotation)
        elif shape_type == 'cube':
            bpy.ops.mesh.primitive_cube_add(size=segment_radius * 2, location=position)
        elif shape_type == 'sphere':
            bpy.ops.mesh.primitive_uv_sphere_add(radius=segment_radius, location=position)

    # Function to create a connector
    def create_connector(start, end, rotation, length):
        midpoint = [(start[i] + end[i]) / 2 for i in range(3)]
        bpy.ops.mesh.primitive_cylinder_add(radius=0.1, depth=length, 
                                            location=midpoint, rotation=rotation)

    # Create segments
    for segment in model_data["segments"]:
        position = segment["position"]
        rotation = segment.get("rotation", [0, 0, 0])  # Default rotation if not present
        length = segment["length"]
        radius = segment["radius"]
        shape_type = segment["type"]
        create_segment(position, rotation, length, radius, shape_type)

    # Create connectors if any
    if "connectors" in model_data:
        for connector in model_data["connectors"]:
            start = connector["start"]
            end = connector["end"]
            rotation = connector.get("rotation", [0, 0, 0])  # Default rotation if not present
            length = connector["length"]
            create_connector(start, end, rotation, length)

    print("3D model generated from JSON data!")

except Exception as e:
    print(f"An error occurred: {e}")
