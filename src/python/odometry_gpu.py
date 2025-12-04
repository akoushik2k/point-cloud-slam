import numpy as np
import open3d as o3d
import os
import glob
import time
import sys

# Add the build directory to the path so we can import our C++/Pybind11 module
# This assumes you run the script from the project root directory
sys.path.append(os.path.join(os.getcwd(), 'build'))

try:
    # Try importing the custom GPU utilities module
    import slam_utils 
    print("SUCCESS: Loaded GPU acceleration module (slam_utils.so).")
    USE_GPU_VOXEL_HASH = True
except ImportError:
    print("WARNING: Could not load slam_utils.so. Using CPU-based Open3D functions.")
    USE_GPU_VOXEL_HASH = False


# --- Configuration ---
# Get the absolute path to the project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Data Paths
SEQUENCE_ID = '00'
VELODYNE_DIR = os.path.join(PROJECT_ROOT, 'data', 'dataset', 'sequences', SEQUENCE_ID, 'velodyne')

# ICP Parameters
VOXEL_SIZE = 0.2  # Downsampling size in meters (Crucial for speed and accuracy)
MAX_CORRESPONDENCE_DISTANCE = VOXEL_SIZE * 5.0 # Max distance for finding correspondences
MAX_ITER = 30 # Number of iterations for ICP to run

# --- Helper Functions ---

def load_kitti_point_cloud(file_path):
    """
    Loads a KITTI .bin file (Velodyne data) into a NumPy array (N, 4).
    """
    try:
        scan = np.fromfile(file_path, dtype=np.float32)
        points = scan.reshape((-1, 4)) # Keep X, Y, Z, Intensity
        return points
    except Exception as e:
        # print(f"Error loading {file_path}: {e}") # Suppress repetitive error during large sequence run
        return None

def preprocess_point_cloud(points_xyzi, voxel_size):
    """
    Converts to Open3D, downsamples, and estimates normals.
    If GPU module is loaded, we use the GPU hash step.
    """
    # Separate XYZ from Intensity (I)
    points_xyz = points_xyzi[:, :3]

    if USE_GPU_VOXEL_HASH:
        # --- GPU Acceleration Step ---
        # 1. Call C++/CUDA function to compute voxel hash keys
        # The input must be (N, 4) with float32 type
        gpu_start = time.time()
        hash_keys = slam_utils.voxel_hash(points_xyzi.astype(np.float32), voxel_size)
        gpu_time = time.time() - gpu_start
        # NOTE: This simple hash function only returns keys. 
        # Full downsampling would require a CPU step to pick representative points per key.
        # For this demonstration, we'll use Open3D for the final step, but the call confirms C++/CUDA linkage.
        # -----------------------------
        print(f"  GPU Voxel Hash Time: {gpu_time * 1000:.2f} ms")


    # --- Standard CPU Preprocessing (Required for normals/ICP) ---
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_xyz)
    
    # Downsample (even if GPU hash was run, we need the structured PCL object)
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
    
    # Estimate normals (critical for Point-to-Plane ICP)
    pcd_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    
    return pcd_down

def icp_odometry(source_pcd, target_pcd, initial_guess, max_correspondence_distance, max_iter):
    """
    Implementation of Point-to-Plane Iterative Closest Point (ICP).
    """
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd, max_correspondence_distance, initial_guess,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter)
    )
    
    return reg_p2p.transformation, reg_p2p.fitness

# --- Visualization Setup ---

def setup_visualizer(name, initial_geometries=[]):
    """Sets up a non-blocking Open3D visualizer."""
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=name, width=800, height=600)
    for geom in initial_geometries:
        vis.add_geometry(geom)
    # Set camera for better viewing (optional)
    view_control = vis.get_view_control()
    view_control.set_zoom(0.5)
    view_control.set_front([0.0, 1.0, 0.0])
    return vis

# --- Main Odometry Loop ---

def run_sequence_odometry():
    """
    Loads all frames in a sequence and runs ICP odometry.
    """
    if not os.path.exists(VELODYNE_DIR):
        print(f"Data sequence directory not found: {VELODYNE_DIR}.")
        return

    all_files = sorted(glob.glob(os.path.join(VELODYNE_DIR, '*.bin')))
    if len(all_files) < 2:
        print("Not enough frames found to run odometry.")
        return

    print(f"Found {len(all_files)} frames in Sequence {SEQUENCE_ID}.")
    print(f"Using Voxel Size: {VOXEL_SIZE}m, Max Corresp. Dist: {MAX_CORRESPONDENCE_DISTANCE:.2f}m")
    
    # Trajectory/Map Accumulators
    cumulative_pose = np.identity(4)
    trajectory_points = [cumulative_pose[:3, 3]] 
    origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3.0, origin=[0, 0, 0])
    
    # 1. Setup Visualizers
    global_trajectory_vis = setup_visualizer("1. Global Trajectory & Map (GPU)", [origin_frame])
    local_odometry_vis = setup_visualizer("2. Local Frame Alignment (GPU)", [])
    
    # 2. Initial Setup
    source_points = load_kitti_point_cloud(all_files[0])
    source_pcd = preprocess_point_cloud(source_points, VOXEL_SIZE)
    
    # Initialize the map with the first frame
    source_pcd.paint_uniform_color([0.5, 0.5, 0.5]) # Gray
    global_trajectory_vis.add_geometry(source_pcd)
    
    T_initial = np.identity(4) 
    
    # 3. Odometry Loop
    for i in range(1, len(all_files)):
        target_file = all_files[i]
        
        # Load and Preprocess the target cloud
        target_points_xyzi = load_kitti_point_cloud(target_file)
        if target_points_xyzi is None:
            continue
            
        target_pcd = preprocess_point_cloud(target_points_xyzi, VOXEL_SIZE)
        
        print(f"--- Processing Frame {i:04d}/{len(all_files)-1} ({os.path.basename(target_file)}) ---")

        start_time = time.time()
        
        # --- Run ICP Odometry ---
        T_relative, fitness = icp_odometry(
            source_pcd, target_pcd, T_initial, 
            MAX_CORRESPONDENCE_DISTANCE, MAX_ITER
        )
        time_elapsed = time.time() - start_time
        
        # 3. Update the global trajectory
        cumulative_pose = cumulative_pose @ T_relative
        current_position = cumulative_pose[:3, 3]
        trajectory_points.append(current_position)

        print(f"  ICP Time: {time_elapsed:.4f}s | Fitness: {fitness:.4f}")
        print(f"  Translation (m): {T_relative[0, 3]:.2f}, {T_relative[1, 3]:.2f}, {T_relative[2, 3]:.2f}")
        print(f"  Cumulative Position (X, Y, Z): {current_position}")
        
        # --- Update Visualizers ---
        local_odometry_vis.clear_geometries()
        
        source_pcd_temp = o3d.geometry.PointCloud(source_pcd)
        source_pcd_temp.paint_uniform_color([1, 0, 0]) # Red (Source)
        local_odometry_vis.add_geometry(source_pcd_temp)
        
        target_pcd_aligned = o3d.geometry.PointCloud(target_pcd)
        target_pcd_aligned.transform(T_relative) # Transform target to align with source
        target_pcd_aligned.paint_uniform_color([0, 0, 1]) # Blue (Target)
        local_odometry_vis.add_geometry(target_pcd_aligned)
        
        # Update map/trajectory every 10 frames
        if i % 10 == 0:
            path_pcd = o3d.geometry.PointCloud()
            path_pcd.points = o3d.utility.Vector3dVector(np.stack(trajectory_points))
            path_pcd.paint_uniform_color([0.0, 0.8, 0.0]) # Green path
            
            global_trajectory_vis.clear_geometries()
            global_trajectory_vis.add_geometry(origin_frame)
            global_trajectory_vis.add_geometry(path_pcd)
            
            global_trajectory_vis.poll_events()
            global_trajectory_vis.update_renderer()
        
        local_odometry_vis.poll_events()
        local_odometry_vis.update_renderer()
        
        # Prepare for next iteration
        source_pcd = target_pcd
        T_initial = T_relative # Use the result as the initial guess for the next step

        if not global_trajectory_vis.poll_events() or not local_odometry_vis.poll_events():
            print("Visualization window closed. Stopping odometry loop.")
            break
        
        time.sleep(0.01)

    print("\n--- Sequence Processing Complete ---")
    print(f"Total displacement (X, Y, Z): {cumulative_pose[:3, 3]}")
    
    # Keep the windows open until manually closed
    global_trajectory_vis.run()
    local_odometry_vis.run()
    
    global_trajectory_vis.destroy_window()
    local_odometry_vis.destroy_window()

if __name__ == "__main__":
    run_sequence_odometry()
    print("Script finished.")