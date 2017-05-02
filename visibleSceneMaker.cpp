#pragma warning(disable: 4819)

#include <thread>
#include <chrono>
#include <algorithm>

#include <boost/filesystem.hpp>

#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/visualization/pcl_visualizer.h>

#include "setting.hpp"

std::vector<bool> visible_check(
	const Eigen::Vector3f& cam_pos,
	const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
	const double lambda_sqrd)
{
	// sort by distance from the camera
	std::sort(cloud->begin(), cloud->end(),
		[&](const pcl::PointXYZ& lhs, const pcl::PointXYZ& rhs) {
		return (cam_pos - lhs.getVector3fMap()).squaredNorm() < (cam_pos - rhs.getVector3fMap()).squaredNorm();
	});

	std::vector<bool> is_visible(cloud->size(), true);

	for (int i = 0; i < cloud->size(); ++i)
	{
		if (i % 1024 == 0) std::cout << i << " / " << cloud->size() << std::endl;
		if (is_visible[i])
		{
#pragma omp parallel for
			for (int j = i + 1; j < cloud->size(); ++j)
			{
				if (is_visible[j])
				{
					const auto& c = cam_pos;
					const auto& p = (*cloud)[i].getVector3fMap();
					const auto& q = (*cloud)[j].getVector3fMap();
					const auto& v = p - c;
					const auto& v_sqrd = v.squaredNorm();
					const auto& k = v.dot(q - c) / v_sqrd;
					const auto& r_sqrd = (q - c - k * v).squaredNorm();
					const auto& s_sqrd = k * k * v_sqrd;
					if (r_sqrd < lambda_sqrd * s_sqrd)
					{
						is_visible[j] = false;
					}
				}
			}
		}
	}

	return is_visible;
}

int main(int argc, char** argv)
{
	VisibleSceneMakerSetting setting(argc, argv);
	const double lambda_sqrd = setting.lambda * setting.lambda;

	//
	// load
	//

	const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_load(new pcl::PointCloud<pcl::PointXYZ>);

	if (pcl::io::loadPCDFile(setting.load_path, *cloud_load) == -1)
	{
		if (pcl::io::loadPLYFile(setting.load_path, *cloud_load) == -1)
		{
			error_exit("PLY or PCD file are only available to load.");
		}
	}

	std::cout << "Input cloud: " << cloud_load->size() << std::endl;

	//
	// visiblity check
	//

	// check form camera1
	Eigen::Vector3f cam_pos_1(setting.camera_1_x, setting.camera_1_y, setting.camera_1_z);
	auto is_visible1 = visible_check(cam_pos_1, cloud_load, lambda_sqrd);

	const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_visible1(new pcl::PointCloud<pcl::PointXYZ>);
	cloud_visible1->reserve(cloud_load->size());
	for (int i = 0; i < cloud_load->size(); ++i)
	{
		if (is_visible1[i]) cloud_visible1->push_back((*cloud_load)[i]);
	}

	// check form camera2
	Eigen::Vector3f cam_pos_2(setting.camera_2_x, setting.camera_2_y, setting.camera_2_z);
	auto is_visible2 = visible_check(cam_pos_2, cloud_visible1, lambda_sqrd);

	const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_save(new pcl::PointCloud<pcl::PointXYZ>());
	cloud_save->reserve(cloud_visible1->size());
	for (int i = 0; i < cloud_visible1->size(); ++i)
	{
		if (is_visible2[i]) cloud_save->push_back((*cloud_visible1)[i]);
	}
	
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
		
		while (!viewer_input->wasStopped() && !viewer_output->wasStopped())
		{
			viewer_input->spinOnce();
			viewer_output->spinOnce();
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
		}
	}
	return 0;
}