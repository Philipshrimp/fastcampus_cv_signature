import trimesh
import os


glb_file_path = "sample.glb"
obj_file_name = "output_model"
output_dir = "obj"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print(f"Loading GLB file: {glb_file_path}")

try:
    input_mesh = trimesh.load(glb_file_path, force='mesh', process=False)

    if isinstance(input_mesh, trimesh.Scene):
        print("Loaded object is a Scene. Processing geometries...")

        count = 0
        exported_files = []

        for geometry_name, geometry in input_mesh.geometry.items():
            if isinstance(geometry, trimesh.Trimesh):
                obj_export_path = os.path.join(output_dir, f"{obj_file_name}_{count}.obj")
                geometry.export(obj_export_path)
                exported_files.append(obj_export_path)
                count += 1

        if not exported_files:
            print("Warning: No mesh geometries.")

    elif isinstance(input_mesh, trimesh.Trimesh):
        obj_export_path = os.path.join(output_dir, obj_file_name + ".obj")
        input_mesh.export(obj_export_path)
        exported_files = [obj_export_path]

    else:
        exported_files = []

except Exception as e:
    print(f"An error occured during conversion: {e}")