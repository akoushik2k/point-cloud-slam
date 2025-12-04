/**
 * @file main.cpp
 * @brief Simple PCL application to test C++ compilation and PCL linking.
 * * This program creates a dummy point cloud and applies a simple Voxel Grid filter.
 */
#include <iostream>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>

void test_pcl_functionality() {
    // 1. Create a point cloud object (using XYZ data type)
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);

    // 2. Populate the cloud with 10 dummy points
    cloud->width    = 10;
    cloud->height   = 1;
    cloud->is_dense = false;
    cloud->points.resize (cloud->width * cloud->height);

    for (size_t i = 0; i < cloud->points.size (); ++i) {
        cloud->points[i].x = 1024 * rand () / (RAND_MAX + 1.0f);
        cloud->points[i].y = 1024 * rand () / (RAND_MAX + 1.0f);
        cloud->points[i].z = 1024 * rand () / (RAND_MAX + 1.0f);
    }

    std::cout << "Loaded " << cloud->width * cloud->height << " data points (before filtering)." << std::endl;

    // 3. Create the Voxel Grid filtering object
    pcl::VoxelGrid<pcl::PointXYZ> sor;
    sor.setInputCloud (cloud);
    sor.setLeafSize (10.0f, 10.0f, 10.0f);

    // 4. Apply the filter
    sor.filter (*cloud_filtered);

    std::cout << "Filtered down to " << cloud_filtered->width * cloud_filtered->height << " data points." << std::endl;
    std::cout << "PCL linking successful!" << std::endl;
}

int main(int argc, char** argv) {
    std::cout << "--- PCL Test Program ---" << std::endl;
    test_pcl_functionality();
    return (0);
}