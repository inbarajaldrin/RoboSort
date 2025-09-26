import omni.usd
import omni.kit.commands
import os
from pxr import UsdGeom, Gf, Sdf, UsdPhysics, PhysxSchema, UsdShade

def create_omni_pbr_and_get_path():
    """Create a new OmniPBR material and return its path"""
    out = []
    omni.kit.commands.execute(
        "CreateAndBindMdlMaterialFromLibrary",
        mdl_name="OmniPBR.mdl",
        mtl_name="OmniPBR",
        mtl_created_list=out
    )
    return out[0] if out else None

def create_cube_from_visuals(aruco_prim_path):
    """Step 1: Create a cube with the same dimensions as the visuals prim"""
    # Dynamic paths based on the aruco prim
    visuals_prim_path = f"{aruco_prim_path}/visuals"
    cube_prim_path = f"{aruco_prim_path}/Cube"
    
    # Get stage and visuals prim
    stage = omni.usd.get_context().get_stage()
    visuals_prim = stage.GetPrimAtPath(visuals_prim_path)
    
    # Create cube directly in the aruco prim using CreateMeshPrimCommand
    omni.kit.commands.execute("CreateMeshPrimCommand", prim_path=cube_prim_path, prim_type="Cube")
    
    # Get the cube prim
    cube_prim = stage.GetPrimAtPath(cube_prim_path)
    
    # Read transform attributes from visuals prim
    translate_attr = visuals_prim.GetAttribute("xformOp:translate")
    orient_attr = visuals_prim.GetAttribute("xformOp:orient")  
    scale_attr = visuals_prim.GetAttribute("xformOp:scale")
    
    # Set cube transform attributes to match visuals
    if translate_attr.IsValid():
        cube_prim.GetAttribute("xformOp:translate").Set(translate_attr.Get())
    if orient_attr.IsValid():
        cube_prim.GetAttribute("xformOp:orient").Set(orient_attr.Get())
    if scale_attr.IsValid():
        cube_prim.GetAttribute("xformOp:scale").Set(scale_attr.Get())
    
    # Delete the old visuals prim
    omni.kit.commands.execute("DeletePrimsCommand", paths=[visuals_prim_path])
    print(f"✓ Deleted visuals prim: {visuals_prim_path}")
    
    return cube_prim_path

def assign_texture_to_material(material_path, texture_filename):
    """Step 3: Assign aruco.png to Albedo map of the Material"""
    stage = omni.usd.get_context().get_stage()
    shader_path = f'{material_path}/Shader'
    prim = stage.GetPrimAtPath(shader_path)
    
    # Convert tilde path to absolute path
    texture_path = os.path.expanduser(f'~/Projects/RoboSort/drop_container_description/drop_container_description/aruco_markers/{texture_filename}')
    print(f"Assigning texture: {texture_path}")
    
    # Check if file exists
    if os.path.exists(texture_path):
        texture_attr = prim.CreateAttribute('inputs:diffuse_texture', Sdf.ValueTypeNames.Asset)
        texture_attr.Set(Sdf.AssetPath(texture_path))
        print(f"Successfully assigned texture to {material_path}")
    else:
        print(f"Warning: Texture file not found: {texture_path}")

def assign_material_to_cube(cube_path, material_path):
    """Step 4: Assign material to newly created cube prim"""
    omni.kit.commands.execute("BindMaterial", 
        prim_path=cube_path, 
        material_path=material_path
    )
    print(f"Material {material_path} assigned to {cube_path}")

def process_all_aruco_markers():
    """Process all 15 ArUco markers"""
    # Define all aruco marker paths and corresponding texture files
    aruco_markers = []
    for i in range(1, 16):  # 01 to 15
        aruco_prim_path = f"/drop_container/aruco_{i:02d}_1"
        texture_filename = f"aruco_marker_{i:02d}.png"
        aruco_markers.append((aruco_prim_path, texture_filename))
    
    print(f"Processing {len(aruco_markers)} ArUco markers...")
    
    for i, (aruco_prim_path, texture_filename) in enumerate(aruco_markers):
        print(f"\n--- Processing marker {i+1}/15: {aruco_prim_path} ---")
        
        try:
            # Step 1: Create cube from visuals and delete visuals prim
            cube_path = create_cube_from_visuals(aruco_prim_path)
            print(f"✓ Created cube: {cube_path}")
            
            # Step 2: Create new material
            material_path = create_omni_pbr_and_get_path()
            if material_path:
                print(f"✓ Created material: {material_path}")
                
                # Step 3: Assign texture to material
                assign_texture_to_material(material_path, texture_filename)
                
                # Step 4: Assign material to cube
                assign_material_to_cube(cube_path, material_path)
                
            else:
                print(f"✗ Failed to create material for {aruco_prim_path}")
                
        except Exception as e:
            print(f"✗ Error processing {aruco_prim_path}: {str(e)}")
    
    print("\n=== Processing Complete ===")

# Execute the processing
if __name__ == "__main__":
    process_all_aruco_markers()