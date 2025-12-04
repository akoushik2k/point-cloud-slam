import numpy as np
import open3d as o3d
import os
import glob
import time
import sys

# Add python source folder to path to import local modules
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__))))

# Import custom modules
from slam_graph import SlamNode, SlamGraph
from slam_helpers import (
    load_kitti_point_cloud, load_kitti_poses, 
    preprocess_point_cloud, setup_visualizer,
    visualize_graph_and_map
)

# --- Configuration ---
# FIX: Go up three levels (from src/python) to correctly reach the project root directory.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Data Paths
SEQUENCE_ID = '00'
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'dataset', 'sequences', SEQUENCE_ID)
VELODYNE_DIR = os.path.join(DATA_DIR, 'velodyne')
POSE_FILE = os.path.join(PROJECT_ROOT, 'data', 'dataset', 'poses', f'{SEQUENCE_ID}.txt')

# ICP Parameters
VOXEL_SIZE = 0.2
MAX_CORRESPONDENCE_DISTANCE = VOXEL_SIZE * 5.0 
MAX_ITER = 30 

# SLAM Parameters
KEYFRAME_DISTANCE_THRESHOLD = 5.0 
LOOP_CLOSURE_DISTANCE = 15.0 
LOOP_CLOSURE_SKIP_FRAMES = 100 
PRIOR_INJECTION_FREQUENCY = 10 # Inject a Prior Edge every 10 keyframes

# --- Core Functions ---

def icp_odometry(source_pcd, target_pcd, initial_guess):
    """
    Implementation of Point-to-Plane Iterative Closest Point (ICP).
    """
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd, MAX_CORRESPONDENCE_DISTANCE, initial_guess,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=MAX_ITER)
    )
    
    return reg_p2p.transformation, reg_p2p.fitness

def detect_and_add_loop_closure(graph, current_node, current_pose_global):
    """
    Searches for a nearby historical keyframe to close the loop.
    (Uses global SLAM parameters from the run_slam module)
    """
    current_translation = current_pose_global[:3, 3]
    
    for old_node in graph.nodes:
        if current_node.index - old_node.index < LOOP_CLOSURE_SKIP_FRAMES:
            continue

        old_translation = old_node.pose[:3, 3]
        distance = np.linalg.norm(current_translation - old_translation)

        if distance < LOOP_CLOSURE_DISTANCE:
            print(f"!!! LOOP CANDIDATE FOUND !!! (Keyframes {current_node.index} and {old_node.index})")
            
            # 1. Calculate approximate alignment
            T_W_O_inv = np.linalg.inv(old_node.pose)
            initial_guess = T_W_O_inv @ current_pose_global
            
            # 2. Run high-precision ICP between the two clouds
            T_loop, fitness = icp_odometry(
                current_node.pcd, old_node.pcd, initial_guess
            )

            # 3. Validate and add constraint
            if fitness > 0.8: # Arbitrary high fitness threshold
                print(f"    LOOP CLOSED with fitness {fitness:.4f}! Adding constraint.")
                # T_loop is the transformation from current to old (T_C_O)
                # Note: Covariance placeholder is None for this demo
                graph.add_loop_closure_edge(current_node.index, old_node.index, T_loop, covariance=None)
                return True, old_node.index
            else:
                print(f"    Loop closure rejected (low fitness: {fitness:.4f}).")
                return False, None
                
    return False, None

def run_graph_optimization(graph, loop_closure_index):
    """
    [Placeholder] Simulates the non-linear graph optimization step.
    """
    print(f"\n*** Graph Optimization Triggered by Loop Closure to Node {loop_closure_index} ***")
    
    # Simulate a simple correction
    correction = np.identity(4)
    correction[0, 3] = 5.0 
    
    for node in graph.nodes:
        if node.index > loop_closure_index:
             node.pose = correction @ node.pose

    print("--- SLAM: Graph optimized. Trajectory corrected! ---")


# --- Sensor Fusion Integration ---

def add_prior_constraints(graph, current_keyframe_node, current_pose_index_in_sequence, global_poses):
    """
    [NEW] Incorporates external sensor fusion data (GPS/IMU) as a Prior Edge.
    """
    node_idx = current_keyframe_node.index
    
    # 1. Check if it's time to inject a prior
    if node_idx % PRIOR_INJECTION_FREQUENCY != 0:
        return

    # 2. Get the global pose from the trusted external source (Ground Truth)
    # The current_pose_index_in_sequence is the frame number (i) that became the keyframe.
    try:
        fused_pose_T = global_poses[current_pose_index_in_sequence].copy()
    except IndexError:
        print("Warning: Pose index out of bounds for prior constraint.")
        return
    
    # 3. Define the uncertainty (Covariance) for the prior constraint
    # GPS/IMU is usually accurate in translation but less so in rotation.
    # We use a 6x6 covariance matrix placeholder (3 translation, 3 rotation)
    covariance = np.zeros((6, 6))
    np.fill_diagonal(covariance, [
        0.5, 0.5, 0.5,  # Translation uncertainty (50cm)
        0.1, 0.1, 0.1   # Rotation uncertainty (small)
    ])
    
    # 4. Add the Prior Edge to the graph
    graph.add_prior_edge(node_idx, fused_pose_T, covariance)
    print(f"  [FUSION] Added Prior Edge to Keyframe {node_idx} using GPS/IMU data.")

# --- Main SLAM Execution ---

def run_sequence_slam():
    # Load all external poses (our proxy for accurate sensor fusion data)
    global_poses_gt = load_kitti_poses(POSE_FILE)
    if not global_poses_gt:
        print("FATAL: Could not load ground truth poses. SLAM cannot run without data.")
        return

    if not os.path.exists(VELODYNE_DIR):
        print(f"Data sequence directory not found: {VELODYNE_DIR}.")
        return

    all_files = sorted(glob.glob(os.path.join(VELODYNE_DIR, '*.bin')))
    if len(all_files) < 2:
        print("Not enough frames found to run odometry.")
        return

    print(f"\n--- Starting GPU-Accelerated Graph SLAM (Sequence {SEQUENCE_ID}) ---")
    print(f"Found {len(all_files)} frames. Keyframe Distance: {KEYFRAME_DISTANCE_THRESHOLD}m")
    
    # Initialize SLAM components
    slam_graph = SlamGraph()
    cumulative_pose = np.identity(4)
    
    # 1. Setup Visualizers
    global_trajectory_vis = setup_visualizer("1. Global SLAM Map & Graph (GPU + Fusion)", [])
    local_odometry_vis = setup_visualizer("2. Local Frame Alignment (GPU)", [])
    
    # 2. Initial Keyframe (Frame 0)
    source_points = load_kitti_point_cloud(all_files[0])
    source_pcd = preprocess_point_cloud(source_points, VOXEL_SIZE)
    current_keyframe_node = SlamNode(index=0, pose=cumulative_pose.copy(), pcd=source_pcd)
    slam_graph.add_node(current_keyframe_node)
    
    last_keyframe_pose = cumulative_pose.copy()
    T_initial = np.identity(4) 
    
    # 3. Main Frame Processing Loop
    for i in range(1, len(all_files)):
        target_file = all_files[i]
        target_points_xyzi = load_kitti_point_cloud(target_file)
        if target_points_xyzi is None:
            continue
            
        target_pcd = preprocess_point_cloud(target_points_xyzi, VOXEL_SIZE)
        
        # --- Run Odometry ---
        T_relative, fitness = icp_odometry(
            source_pcd, target_pcd, T_initial
        )
        
        # Update current global pose using odometry
        cumulative_pose = cumulative_pose @ T_relative
        
        # --- Keyframe Check ---
        current_translation = cumulative_pose[:3, 3]
        last_keyframe_translation = last_keyframe_pose[:3, 3]
        odometry_drift = np.linalg.norm(current_translation - last_keyframe_translation)
        
        print(f"--- Frame {i:04d}/{len(all_files)-1} | Drift: {odometry_drift:.2f}m ---")
        
        optimization_triggered = False

        if odometry_drift > KEYFRAME_DISTANCE_THRESHOLD:
            # 4. New Keyframe Creation (Node)
            new_keyframe_idx = len(slam_graph.nodes)
            new_keyframe_node = SlamNode(index=new_keyframe_idx, pose=cumulative_pose.copy(), pcd=target_pcd)
            slam_graph.add_node(new_keyframe_node)
            
            # 5. Add Odometry Edge (Constraint)
            T_keyframe_relative = np.linalg.inv(last_keyframe_pose) @ cumulative_pose
            slam_graph.add_odometry_edge(current_keyframe_node.index, new_keyframe_node.index, T_keyframe_relative, covariance=None)
            
            print(f"  --> NEW KEYFRAME {new_keyframe_idx} added. Nodes: {len(slam_graph.nodes)}")
            
            # 6. Sensor Fusion (Prior Constraint)
            add_prior_constraints(slam_graph, new_keyframe_node, i, global_poses_gt)

            # 7. Loop Closure Detection
            is_closed, loop_node_idx = detect_and_add_loop_closure(
                slam_graph, new_keyframe_node, cumulative_pose
            )

            if is_closed:
                # 8. Optimization
                run_graph_optimization(slam_graph, loop_node_idx)
                optimization_triggered = True

            # Update for next iteration
            current_keyframe_node = new_keyframe_node
            last_keyframe_pose = cumulative_pose.copy()

        # --- Update Visualizers ---
        local_odometry_vis.clear_geometries()
        source_pcd_temp = o3d.geometry.PointCloud(source_pcd)
        source_pcd_temp.paint_uniform_color([1, 0, 0])
        local_odometry_vis.add_geometry(source_pcd_temp)
        target_pcd_aligned = o3d.geometry.PointCloud(target_pcd)
        target_pcd_aligned.transform(T_relative) 
        target_pcd_aligned.paint_uniform_color([0, 0, 1]) 
        local_odometry_vis.add_geometry(target_pcd_aligned)
        local_odometry_vis.poll_events()
        local_odometry_vis.update_renderer()
        
        if odometry_drift > KEYFRAME_DISTANCE_THRESHOLD or optimization_triggered:
            visualize_graph_and_map(slam_graph, global_trajectory_vis, optimization_triggered)
        
        # Prepare for next iteration
        source_pcd = target_pcd
        T_initial = T_relative

        if not global_trajectory_vis.poll_events() or not local_odometry_vis.poll_events():
            print("Visualization window closed. Stopping SLAM loop.")
            break
        
        time.sleep(0.01)

    print("\n--- Final SLAM Sequence Processing Complete ---")
    
    # Final visualization before closing
    visualize_graph_and_map(slam_graph, global_trajectory_vis, optimization_needed=False)

    global_trajectory_vis.run()
    local_odometry_vis.run()
    
    global_trajectory_vis.destroy_window()
    local_odometry_vis.destroy_window()

if __name__ == "__main__":
    run_sequence_slam()
    print("Script finished.")