import numpy as np

class SlamNode:
    """Represents a pose in the map graph (a Keyframe)."""
    def __init__(self, index, pose, pcd):
        self.index = index        # Keyframe ID
        self.pose = pose          # 4x4 Global Transformation Matrix (T_W_K)
        self.pcd = pcd            # Downsampled Point Cloud (Map component)

class SlamGraph:
    """Stores the graph structure: Nodes (Keyframes) and Edges (Constraints)."""
    def __init__(self):
        self.nodes = [] # List of SlamNode objects
        self.edges = [] # List of edge dictionaries

    def add_node(self, node):
        self.nodes.append(node)
        
    def add_odometry_edge(self, source_node_idx, target_node_idx, transformation, covariance=None):
        """Adds an Odometry constraint between consecutive keyframes."""
        self.edges.append({
            'type': 'odometry',
            'source': source_node_idx, 
            'target': target_node_idx, 
            'transform': transformation,
            'covariance': covariance # Optional: uncertainty of the measurement
        })
        
    def add_loop_closure_edge(self, source_node_idx, target_node_idx, transformation, covariance=None):
        """Adds a Loop Closure constraint between distant keyframes."""
        self.edges.append({
            'type': 'loop_closure',
            'source': source_node_idx, 
            'target': target_node_idx, 
            'transform': transformation,
            'covariance': covariance
        })

    def add_prior_edge(self, node_idx, pose, covariance):
        """Adds a Prior (Sensor Fusion) constraint to a specific node."""
        # Priors usually come from external sensors (GPS/IMU)
        self.edges.append({
            'type': 'prior',
            'node': node_idx,
            'pose': pose,
            'covariance': covariance
        })

    def get_last_node(self):
        return self.nodes[-1] if self.nodes else None