#pragma once

#include <iostream>
#include <stdexcept>
#include <random>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include <boost/filesystem.hpp>

std::string setting_file = "./setting.ini";

void error_exit(const std::string& message)
{
	std::cerr << message << std::endl;
	throw std::runtime_error(message);
}

class SettingBase
{
public:
	template<typename T>
	T load_value(const std::string& key)
	{
		auto val_opt = pt.get_optional<T>(key);
		if (val_opt)
		{
			return *val_opt;
		}
		else
		{
			error_exit("Cannot load " + key);
		}
	}

	boost::property_tree::ptree pt;
};

class BinSceneMakerSetting : public SettingBase
{
public:
	double normalize_length;
	int random_seed;
	int model_num;
	int falling_interval;
	int postsimulation_wait;
	int downsample_target_points_num;
	double downsample_initial_leaf_size;
	double downsample_factor;
	double cup_r;
	double cup_h;
	double cup_restitution;
	double cup_friction;
	double model_restitution;
	double model_friction;
	double precapture_wait;
	bool capture_screenshots;
	bool visualization;
	std::string load_model_path;
	std::string save_pointcloud_path;
	std::string save_transforms_path;
	std::string save_centering_model_path;
	std::string save_screenshots_dir;

	BinSceneMakerSetting(int argc, char** argv)
	{
		if (argc > 1)
		{
			setting_file = argv[1];
		}

		if (!boost::filesystem::exists(setting_file))
		{
			error_exit("setting.ini does not exist.");
		}

		boost::property_tree::read_ini(setting_file, pt);

		random_seed = load_value<int>("binSceneMaker.random_seed");
		if (random_seed < 0)
		{
			random_seed = std::random_device()();
		}

		normalize_length = load_value<double>("binSceneMaker.normalize_length");
		if (normalize_length < 0.0)
		{
			error_exit("normalize_length should be larger than 0.0");
		}

		cup_r = load_value<double>("binSceneMaker.cup_r");
		if (cup_r < 0.0)
		{
			error_exit("cup_r should be larger than 0.0");
		}

		cup_h = load_value<double>("binSceneMaker.cup_h");
		if (cup_h < 0.0)
		{
			error_exit("cup_h should be larger than 0.0");
		}

		model_num = load_value<int>("binSceneMaker.model_num");
		if (model_num < 1)
		{
			error_exit("model_num should be larger than 0");
		}

		downsample_initial_leaf_size = load_value<double>("binSceneMaker.downsample_initial_leaf_size");
		if (downsample_initial_leaf_size < 0.0)
		{
			error_exit("downsample_initial_leaf_size should be larger than 0.0");
		}

		downsample_factor = load_value<double>("binSceneMaker.downsample_factor");
		if (downsample_factor < 1.0)
		{
			error_exit("downsample_factor should be larger than 1.0");
		}

		cup_restitution = load_value<double>("binSceneMaker.cup_restitution");
		if (cup_restitution < 0.0)
		{
			error_exit("cup_restitution should be larger than 0.0");
		}

		cup_friction = load_value<double>("binSceneMaker.cup_friction");
		if (cup_friction < 0.0)
		{
			error_exit("cup_friction should be larger than 1.0");
		}

		model_restitution = load_value<double>("binSceneMaker.model_restitution");
		if (model_restitution < 0.0)
		{
			error_exit("model_restitution should be larger than 1.0");
		}

		model_friction = load_value<double>("binSceneMaker.model_friction");
		if (model_friction < 0.0)
		{
			error_exit("model_friction should be larger than 1.0");
		}

		downsample_target_points_num = load_value<int>("binSceneMaker.downsample_target_points_num");
		if (downsample_target_points_num < 1)
		{
			error_exit("downsample_target_points_num should be larger than 0");
		}

		falling_interval = load_value<int>("binSceneMaker.falling_interval");
		if (falling_interval < 1)
		{
			error_exit("falling_interval should be larger than 0");
		}

		postsimulation_wait = load_value<int>("binSceneMaker.postsimulation_wait");
		if (postsimulation_wait < 0)
		{
			error_exit("postsimulation_wait should be larger than or equal to 0");
		}

		load_model_path = load_value<std::string>("binSceneMaker.load_model_path");
		if (!boost::filesystem::exists(load_model_path))
		{
			error_exit("load_model does not exist.");
		}

		save_pointcloud_path = load_value<std::string>("binSceneMaker.save_pointcloud_path");
		boost::filesystem::create_directories(boost::filesystem::path(save_pointcloud_path).parent_path());

		save_centering_model_path = load_value<std::string>("binSceneMaker.save_centering_model_path");
		boost::filesystem::create_directories(boost::filesystem::path(save_centering_model_path).parent_path());

		save_transforms_path = load_value<std::string>("binSceneMaker.save_transforms_path");
		boost::filesystem::create_directories(boost::filesystem::path(save_transforms_path).parent_path());

		visualization = load_value<bool>("binSceneMaker.visualization");

		capture_screenshots = load_value<bool>("binSceneMaker.capture_screenshots");
		if (capture_screenshots)
		{
			if (!visualization)
			{
				std::cerr << "Cannot enable capture_screenshots without visualization. capture_screenshots is set disabled." << std::endl;
				capture_screenshots = false;
			}
			else
			{
				precapture_wait = load_value<double>("binSceneMaker.precapture_wait");
				if (precapture_wait < 0.0)
				{
					error_exit("precapture_wait should be larger than 0.0");
				}

				save_screenshots_dir = load_value<std::string>("binSceneMaker.save_screenshots_dir");
				boost::filesystem::create_directories(save_screenshots_dir);
			}
		}
	}
};

class VisibleSceneMakerSetting : public SettingBase
{
public:
	bool visualization;
	std::string load_path;
	std::string save_path;
	double lambda;
	double camera_1_x;
	double camera_1_y;
	double camera_1_z;
	double camera_2_x;
	double camera_2_y;
	double camera_2_z;

	VisibleSceneMakerSetting(int argc, char** argv)
	{
		if (argc > 1)
		{
			setting_file = argv[1];
		}

		if (!boost::filesystem::exists(setting_file))
		{
			error_exit("setting.ini does not exist.");
		}

		boost::property_tree::read_ini(setting_file, pt);

		load_path = load_value<std::string>("visibleSceneMaker.load_path");
		if (!boost::filesystem::exists(load_path))
		{
			error_exit("load_path does not exist.");
		}

		save_path = load_value<std::string>("visibleSceneMaker.save_path");
		boost::filesystem::create_directories(boost::filesystem::path(save_path).parent_path());

		lambda = load_value<double>("visibleSceneMaker.lambda");
		if (lambda < 0.0)
		{
			error_exit("lambda should be larger than 0.0");
		}

		visualization = load_value<bool>("binSceneMaker.visualization");

		camera_1_x = load_value<double>("visibleSceneMaker.camera_1_x");
		camera_1_y = load_value<double>("visibleSceneMaker.camera_1_y");
		camera_1_z = load_value<double>("visibleSceneMaker.camera_1_z");
		camera_2_x = load_value<double>("visibleSceneMaker.camera_2_x");
		camera_2_y = load_value<double>("visibleSceneMaker.camera_2_y");
		camera_2_z = load_value<double>("visibleSceneMaker.camera_2_z");
	}
};