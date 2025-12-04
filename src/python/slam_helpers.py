import numpy as np 
import open3d as o3d
import os
import glob
import time

# Attempt to import the GPU module built by CMAKE/Pybind11
try:
    import sys
    # FIX: Make the path to the 'build' directory robust by starting from this file's location
    # This ensures we always find the C++ module, regardless of where the script is run from.
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.append(os.path.join(PROJECT_ROOT, 'build'))
    import slam_utils as gpu_module
    print("SUCCESS: Loaded GPU acceleration module (slam_utils.so).")
    USE_GPU_VOXEL_HASH = True
except ImportError:
    print("WARNING: Could not load slam_utils.so. Using CPU-based Open3D functions.")
    USE_GPU_VOXEL_HASH = False


# --- Data Handling ---

def load_kitti_point_cloud(file_path):
    """
    Loads a KITTI .bin file (Velodyne data) into a NumPy array (N, 4).
    """
    try:
        scan = np.fromfile(file_path, dtype=np.float32)
        points = scan.reshape((-1, 4)) # Keep X, Y, Z, Intensity
        return points
    except Exception as e:
        return None

def load_kitti_poses(pose_file_path):
    """
    Loads KITTI ground truth poses (4x12 matrix) and converts to 4x4 matrix list.
    """
    try:
        poses_raw = np.loadtxt(pose_file_path).reshape(-1, 12)
        poses = []
        for row in poses_raw:
            T = np.eye(4)
            T[:3, :4] = row.reshape(3, 4)
            poses.append(T)
        return poses
    except FileNotFoundError:
        print(f"ERROR: Pose file not found at {pose_file_path}")
        return []
    except Exception as e:
        print(f"Error loading poses: {e}")
        return []


def preprocess_point_cloud(points_xyzi, voxel_size):
    """
    Converts to Open3D, downsamples, and estimates normals. 
    Calls GPU hash step if available.
    """
    points_xyz = points_xyzi[:, :3]

    if USE_GPU_VOXEL_HASH:
        # --- GPU Acceleration Step (For demonstration and timing) ---
        gpu_start = time.time()
        # This calls the C++/CUDA code defined in gpu_processing.cpp and downsample_kernel.cu
        hash_keys = gpu_module.voxel_hash(points_xyzi.astype(np.float32), voxel_size)
        gpu_time = time.time() - gpu_start
        print(f"  GPU Voxel Hash Time: {gpu_time * 1000:.2f} ms")


    # --- Standard CPU Preprocessing (Required for normals/ICP) ---
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_xyz)
    
    # Downsample
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
    
    # Estimate normals (critical for Point-to-Plane ICP)
    pcd_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    
    return pcd_down


# --- Visualization ---

def setup_visualizer(name, initial_geometries=[]):
    """Sets up a non-blocking Open3D visualizer."""
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=name, width=800, height=600)
    for geom in initial_geometries:
        vis.add_geometry(geom)
    view_control = vis.get_view_control()
    view_control.set_zoom(0.5)
    view_control.set_front([0.0, 1.0, 0.0])
    return vis

def visualize_graph_and_map(graph, visualizer, optimization_needed=False):
    """Updates the global map visualizer with nodes, edges, and map fragments."""
    visualizer.clear_geometries()
    
    # 1. Draw Map Fragments (keyframes)
    for node in graph.nodes:
        # Transform keyframe cloud to its current global pose
        pcd_global = o3d.geometry.PointCloud(node.pcd)
        pcd_global.transform(node.pose)
        
        # Color the map fragment based on whether optimization is needed
        color = [0.5, 0.5, 0.5] if not optimization_needed else [0.0, 0.8, 0.8] # Cyan if optimized
        pcd_global.paint_uniform_color(color)
        visualizer.add_geometry(pcd_global)
    
    # 2. Draw Trajectory (Path connecting nodes)
    path_points = [node.pose[:3, 3] for node in graph.nodes]
    if path_points:
        path_pcd = o3d.geometry.PointCloud()
        path_pcd.points = o3d.utility.Vector3dVector(np.stack(path_points))
        path_pcd.paint_uniform_color([0.0, 0.8, 0.0]) # Green path
        visualizer.add_geometry(path_pcd)

        # Create LineSet for odometry and loop closures
        points_list = np.stack(path_points)
        lineset = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points_list),
            lines=o3d.utility.Vector2iVector([[i, i+1] for i in range(len(points_list)-1)])
        )
        lineset.paint_uniform_color([0.5, 0.5, 0.5]) # Gray Odometry Edges
        visualizer.add_geometry(lineset)
        
        # Add origin frame last
        origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3.0, origin=[0, 0, 0])
        visualizer.add_geometry(origin_frame)
    
    visualizer.poll_events()
    visualizer.update_renderer()