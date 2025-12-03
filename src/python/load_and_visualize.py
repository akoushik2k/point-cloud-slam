import numpy as np
import open3d as o3d
import os
import glob
import time

# --- Configuration ---
# Get the absolute path to the project root
# Go up three levels to reach the project root directory (from src/python/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Data Paths
SEQUENCE_ID = '00'
VELODYNE_DIR = os.path.join(PROJECT_ROOT, 'data', 'dataset', 'sequences', SEQUENCE_ID, 'velodyne')

# ICP Parameters
VOXEL_SIZE = 0.2  # Downsampling size in meters (Crucial for speed and accuracy)
MAX_CORRESPONDENCE_DISTANCE = VOXEL_SIZE * 5.0 # Max distance for finding correspondences
ICP_THRESHOLD = 0.05 # Smaller threshold for fine registration (optional in this basic demo)
MAX_ITER = 30 # Number of iterations for ICP to run

# --- Helper Functions ---

def load_kitti_point_cloud(file_path):
    """
    Loads a KITTI .bin file (Velodyne data) into a NumPy array (N, 3).
    """
    try:
        scan = np.fromfile(file_path, dtype=np.float32)
        points = scan.reshape((-1, 4))[:, :3] # Only X, Y, Z
        return points
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def preprocess_point_cloud(points_xyz, voxel_size):
    """
    Converts to Open3D, downsamples, and estimates normals (needed for Point-to-Plane ICP).
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_xyz)
    
    # Downsample
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
    
    # Estimate normals (critical for Point-to-Plane ICP)
    # Normals are estimated using nearest neighbor search
    pcd_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    
    return pcd_down

def icp_odometry(source_pcd, target_pcd, initial_guess, max_correspondence_distance, max_iter):
    """
    Implementation of Point-to-Plane Iterative Closest Point (ICP).
    
    We use Point-to-Plane as it often converges faster and more accurately 
    than Point-to-Point for structured environments like KITTI.
    """
    
    # Define the registration method
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd, max_correspondence_distance, initial_guess,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter)
    )
    
    # reg_p2p contains: transformation, fitness, and inlier_rmse
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
    Loads all frames in a sequence and runs ICP odometry, visualizing 
    the result in two separate windows.
    """
    if not os.path.exists(VELODYNE_DIR):
        print(f"Data sequence directory not found: {VELODYNE_DIR}.")
        print("Please verify the full dataset download and extraction.")
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
    global_trajectory_vis = setup_visualizer("1. Global Trajectory & Map", [origin_frame])
    local_odometry_vis = setup_visualizer("2. Local Frame Alignment (Video)", [])
    
    # 2. Initial Setup
    print(f"Loading initial frame: {os.path.basename(all_files[0])}")
    source_points = load_kitti_point_cloud(all_files[0])
    source_pcd = preprocess_point_cloud(source_points, VOXEL_SIZE)
    
    # Initialize the map with the first frame
    source_pcd.paint_uniform_color([0.5, 0.5, 0.5]) # Gray
    global_trajectory_vis.add_geometry(source_pcd)
    
    # Initial guess for the first step is always identity
    T_initial = np.identity(4) 
    
    # 3. Odometry Loop
    for i in range(1, len(all_files)):
        target_file = all_files[i]
        
        # Load and Preprocess the target cloud
        target_points = load_kitti_point_cloud(target_file)
        if target_points is None:
            continue
            
        target_pcd = preprocess_point_cloud(target_points, VOXEL_SIZE)
        
        print(f"--- Processing Frame {i:04d}/{len(all_files)-1} ({os.path.basename(target_file)}) ---")

        start_time = time.time()
        
        # --- Run ICP Odometry ---
        T_relative, fitness = icp_odometry(
            source_pcd, target_pcd, T_initial, 
            MAX_CORRESPONDENCE_DISTANCE, MAX_ITER
        )
        time_elapsed = time.time() - start_time
        
        # 3. Update the global trajectory
        # T_new = T_old @ T_relative
        cumulative_pose = cumulative_pose @ T_relative
        current_position = cumulative_pose[:3, 3]
        trajectory_points.append(current_position)

        print(f"  ICP Time: {time_elapsed:.4f}s | Fitness: {fitness:.4f}")
        print(f"  Translation (m): {T_relative[0, 3]:.2f}, {T_relative[1, 3]:.2f}, {T_relative[2, 3]:.2f}")
        print(f"  Cumulative Position (X, Y, Z): {current_position}")
        
        # --- Update Visualizers ---

        # A. Local Odometry (The "Video" Feed)
        # Clear local visualizer
        local_odometry_vis.clear_geometries()
        
        # 1. Visualize Source (Previous Frame)
        source_pcd_temp = o3d.geometry.PointCloud(source_pcd)
        source_pcd_temp.paint_uniform_color([1, 0, 0]) # Red (Source)
        local_odometry_vis.add_geometry(source_pcd_temp)
        
        # 2. Visualize Aligned Target (Current Frame)
        target_pcd_aligned = o3d.geometry.PointCloud(target_pcd)
        target_pcd_aligned.transform(T_relative) # Transform target to align with source
        target_pcd_aligned.paint_uniform_color([0, 0, 1]) # Blue (Target)
        local_odometry_vis.add_geometry(target_pcd_aligned)
        
        local_odometry_vis.update_geometry(source_pcd_temp)
        local_odometry_vis.update_geometry(target_pcd_aligned)
        local_odometry_vis.poll_events()
        local_odometry_vis.update_renderer()
        
        # B. Global Trajectory and Map (Trajectory Window)
        
        # Update map/trajectory every 10 frames to avoid clutter
        if i % 10 == 0:
            # Update trajectory
            path_pcd = o3d.geometry.PointCloud()
            path_pcd.points = o3d.utility.Vector3dVector(np.stack(trajectory_points))
            path_pcd.paint_uniform_color([0.0, 0.8, 0.0]) # Green path
            
            global_trajectory_vis.clear_geometries()
            global_trajectory_vis.add_geometry(origin_frame)
            global_trajectory_vis.add_geometry(path_pcd)
            
            global_trajectory_vis.poll_events()
            global_trajectory_vis.update_renderer()
        
        # Prepare for next iteration
        source_pcd = target_pcd
        T_initial = T_relative # Use the result as the initial guess for the next step

        # Must check if visualization windows are still open
        if not global_trajectory_vis.poll_events() or not local_odometry_vis.poll_events():
            print("Visualization window closed. Stopping odometry loop.")
            break
        
        # Small delay for smooth visual updates
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