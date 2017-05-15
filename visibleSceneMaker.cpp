#pragma warning(disable: 4819)

#include <thread>
#include <chrono>
#include <algorithm>
#include <chrono>

#include <boost/filesystem.hpp>

#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/visualization/pcl_visualizer.h>

#include "setting.hpp"
#include "visible_check.h"

// sort pointcloud based on the distance from the camera
void cloud_sort_by_distance_from_camera(
	const Vec3& cam_pos,
	const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
{
	const auto sqrd_dist_from_cam = [&](const pcl::PointXYZ& p) {
		return (cam_pos[0] - p.x) * (cam_pos[0] - p.x) + (cam_pos[1] - p.y) * (cam_pos[1] - p.y) + (cam_pos[2] - p.z) * (cam_pos[2] - p.z);
	};

	std::sort(cloud->begin(), cloud->end(),
		[&](const pcl::PointXYZ& lhs, const pcl::PointXYZ& rhs) {
		return sqrd_dist_from_cam(lhs) < sqrd_dist_from_cam(rhs);
	});
}

// pcl::PointCloud<pcl::PointXYZ>::Ptr -> std::vector<float>
std::vector<float> cloud2vector(const pcl::PointCloud<pcl::PointXYZ>::Ptr& c)
{
	std::vector<float> v(c->size() * 3);
	for (int i = 0; i < c->size(); ++i)
	{
		v[i * 3 + 0] = (*c)[i].x;
		v[i * 3 + 1] = (*c)[i].y;
		v[i * 3 + 2] = (*c)[i].z;
	}

	return v;
}

// std::vector<float> -> pcl::PointCloud<pcl::PointXYZ>::Ptr
pcl::PointCloud<pcl::PointXYZ>::Ptr vector2cloud(const std::vector<float>& v)
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr c(new pcl::PointCloud<pcl::PointXYZ>);
	for (int i = 0; i < v.size() / 3; ++i)
	{
		c->push_back(pcl::PointXYZ(v[i * 3 + 0], v[i * 3 + 1], v[i * 3 + 2]));
	}

	return c;
}

int main(int argc, char** argv)
{
	VisibleSceneMakerSetting setting(argc, argv);
	const float lambda_sqrd = setting.lambda * setting.lambda;

	//
	// load
	//

	const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_load(new pcl::PointCloud<pcl::PointXYZ>);

	if (boost::filesystem::path(setting.load_path).extension() == ".pcd")
	{
		if (pcl::io::loadPCDFile(setting.load_path, *cloud_load) == -1)
		{
			error_exit("Load error.");
		}
	}
	else if (boost::filesystem::path(setting.load_path).extension() == ".ply")
	{
		if (pcl::io::loadPLYFile(setting.load_path, *cloud_load) == -1)
		{
			error_exit("Load error.");
		}
	}
	else
	{
		error_exit("PLY or PCD file are only available to load.");
	}

	std::cout << "Input cloud: " << cloud_load->size() << std::endl;

	//
	// visiblity check
	//

	const auto start = std::chrono::system_clock::now();

	// check form camera1
	const Vec3 cam_pos_1 {
		static_cast<float>(setting.camera_1_x),
		static_cast<float>(setting.camera_1_y),
		static_cast<float>(setting.camera_1_z)
	};
	cloud_sort_by_distance_from_camera(cam_pos_1, cloud_load);
	auto cloud_visible1_v = visible_check(cam_pos_1, cloud2vector(cloud_load), lambda_sqrd);
	auto cloud_visible1 = vector2cloud(cloud_visible1_v);

	// check form camera2
	const Vec3 cam_pos_2 {
		static_cast<float>(setting.camera_2_x),
		static_cast<float>(setting.camera_2_y),
		static_cast<float>(setting.camera_2_z)
	};
	cloud_sort_by_distance_from_camera(cam_pos_2, cloud_visible1);
	auto cloud_save_v = visible_check(cam_pos_1, cloud2vector(cloud_visible1), lambda_sqrd);
	auto cloud_save = vector2cloud(cloud_save_v);

	const auto end = std::chrono::system_clock::now();
	std::cout << "time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " [msec]" << std::endl;

	//
	// save
	//
	
	if (cloud_save->size() > 0)
	{
		if (boost::filesystem::path(setting.save_path).extension() == ".pcd")
		{
			pcl::io::savePCDFileBinary(setting.save_path, *cloud_save);
		}
		else if (boost::filesystem::path(setting.save_path).extension() == ".ply")
		{
			pcl::io::savePLYFileBinary(setting.save_path, *cloud_save);
		}
		else
		{
			error_exit("PLY or PCD file are only available to save.");
		}
	}
	else
	{
		error_exit("No point is exist in visible scene. Please chenge lambda and/or camera positions.");
	}
	
	//
	// visualization
	//

	if (setting.visualization)
	{
		pcl::visualization::PCLVisualizer::Ptr viewer_input(new pcl::visualization::PCLVisualizer("Input"));
		if (cloud_load->size() > 0)
		{
			viewer_input->addPointCloud(cloud_load);
		}
		
		pcl::visualization::PCLVisualizer::Ptr viewer_output(new pcl::visualization::PCLVisualizer("Output"));
		if (cloud_save->size() > 0)
		{
			viewer_output->addPointCloud(cloud_save);
		}
		
		viewer_input->spinOnce();
		viewer_output->spinOnce();

		while (!viewer_input->wasStopped() && !viewer_output->wasStopped())
		{
			viewer_input->spinOnce();
			viewer_output->spinOnce();
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
		}
	}
	return 0;
}