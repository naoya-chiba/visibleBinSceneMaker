#pragma once

#include <iostream>
#include <stdexcept>
#include <random>
#include <cstdlib>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include <boost/filesystem.hpp>

#include <pcl/console/parse.h>

const std::string default_setting_file = "./setting.ini";

#define LOAD_VALUE(key) load_value_impl(key, #key)

void error_exit(const std::string& message)
{
	pcl::console::print_error(message.c_str());
	std::exit(0);
	//throw std::runtime_error(message);
}

class SettingBase
{
public:
	SettingBase(int argc, char** argv, const std::string& section_name)
	{
		std::vector<int> ini_file_indices = pcl::console::parse_file_extension_argument(argc, argv, ".ini");
		if (ini_file_indices.size() < 1)
		{
			setting_file = default_setting_file;
		}
		else
		{
			setting_file = argv[ini_file_indices[0]];
		}

		open_setting_file(section_name);
	}

	template<typename T>
	void load_value_impl(T& retval, const std::string& key) const
	{
		if (!pt)
		{
			error_exit("Setting section does not loaded.");
		}

		const auto& val_opt = pt->get_optional<T>(key);
		if (!val_opt)
		{
			error_exit("Cannot load " + key);
		}
		
		retval = *val_opt;
	}

	void open_setting_file(const std::string& key)
	{
		if (!setting_file)
		{
			error_exit("setting_file is not initialized.");
		}
		else
		{
			if (!boost::filesystem::exists(*setting_file))
			{
				error_exit("setting.ini does not exist.");
			}
		}

		boost::property_tree::ptree pt_root;
		boost::property_tree::read_ini(*setting_file, pt_root);

		pt = pt_root.get_child_optional(key);
		if (!pt)
		{
			error_exit(key + " does not found in setting file.");
		}
	}

	void absolute_path(std::string& path) const
	{
		if (!setting_file)
		{
			error_exit("setting_file is not initialized.");
		}

		const auto& base_path = boost::filesystem::path(*setting_file).parent_path();
		path = boost::filesystem::absolute(path, base_path).string();
	}

	boost::optional<boost::property_tree::ptree> pt;
	boost::optional<std::string> setting_file;
};

class BinSceneMakerSetting : public SettingBase
{
public:
	std::string load_model_path;
	std::string save_centering_model_path;
	std::string save_pointcloud_path;
	std::string save_transforms_path;
	std::string save_screenshots_dir;
	int model_num;
	double precapture_wait;
	int falling_interval;
	int postsimulation_wait;
	int random_seed;
	double cup_r;
	double cup_h;
	double cup_restitution;
	double cup_friction;
	double model_restitution;
	double model_friction;
	double normalize_length;
	int downsample_target_points_num;
	double downsample_initial_leaf_size;
	double downsample_factor;
	bool visualization;
	bool capture_screenshots;

	BinSceneMakerSetting(int argc, char** argv) : SettingBase(argc, argv, "binSceneMaker")
	{
		if (pcl::console::find_switch(argc, argv, "-h"))
		{
			show_help();
			exit(0);
		}

		if (pcl::console::parse_argument(argc, argv, "--load_model_path", load_model_path) == -1)
		{
			LOAD_VALUE(load_model_path);
		}
		absolute_path(load_model_path);
		if (!boost::filesystem::exists(load_model_path))
		{
			error_exit("load_model_path does not exist.");
		}

		if (pcl::console::parse_argument(argc, argv, "--save_centering_model_path", load_model_path) == -1)
		{
			LOAD_VALUE(save_centering_model_path);
		}
		absolute_path(save_centering_model_path);
		boost::filesystem::create_directories(boost::filesystem::path(save_centering_model_path).parent_path());

		if (pcl::console::parse_argument(argc, argv, "--save_pointcloud_path", save_pointcloud_path) == -1)
		{
			LOAD_VALUE(save_pointcloud_path);
		}
		absolute_path(save_pointcloud_path);
		boost::filesystem::create_directories(boost::filesystem::path(save_pointcloud_path).parent_path());

		if (pcl::console::parse_argument(argc, argv, "--save_transforms_path", save_transforms_path) == -1)
		{
			LOAD_VALUE(save_transforms_path);
		}
		absolute_path(save_transforms_path);
		boost::filesystem::create_directories(boost::filesystem::path(save_transforms_path).parent_path());

		if (pcl::console::parse_argument(argc, argv, "--model_num", model_num) == -1)
		{
			LOAD_VALUE(model_num);
		}
		if (model_num < 1)
		{
			error_exit("model_num should be larger than 0");
		}

		if (pcl::console::parse_argument(argc, argv, "--postsimulation_wait", postsimulation_wait) == -1)
		{
			LOAD_VALUE(postsimulation_wait);
		}
		if (postsimulation_wait < 0)
		{
			error_exit("postsimulation_wait should be larger than or equal to 0");
		}
		
		if (pcl::console::parse_argument(argc, argv, "--falling_interval", falling_interval) == -1)
		{
			LOAD_VALUE(falling_interval);
		}
		if (falling_interval < 1)
		{
			error_exit("falling_interval should be larger than 0");
		}
		
		if (pcl::console::parse_argument(argc, argv, "--random_seed", random_seed) == -1)
		{
			LOAD_VALUE(random_seed);
		}
		if (random_seed < 0)
		{
			random_seed = std::random_device()();
		}
		
		if (pcl::console::parse_argument(argc, argv, "--cup_r", cup_r) == -1)
		{
			LOAD_VALUE(cup_r);
		}
		if (cup_r < 0.0)
		{
			error_exit("cup_r should be larger than 0.0");
		}

		if (pcl::console::parse_argument(argc, argv, "--cup_h", cup_h) == -1)
		{
			LOAD_VALUE(cup_h);
		}
		if (cup_h < 0.0)
		{
			error_exit("cup_h should be larger than 0.0");
		}

		if (pcl::console::parse_argument(argc, argv, "--cup_restitution", cup_restitution) == -1)
		{
			LOAD_VALUE(cup_restitution);
		}
		if (cup_restitution < 0.0)
		{
			error_exit("cup_restitution should be larger than 0.0");
		}

		if (pcl::console::parse_argument(argc, argv, "--cup_friction", cup_friction) == -1)
		{
			LOAD_VALUE(cup_friction);
		}
		if (cup_friction < 0.0)
		{
			error_exit("cup_friction should be larger than 1.0");
		}
		
		if (pcl::console::parse_argument(argc, argv, "--model_restitution", model_restitution) == -1)
		{
			LOAD_VALUE(model_restitution);
		}
		if (model_restitution < 0.0)
		{
			error_exit("model_restitution should be larger than 1.0");
		}

		if (pcl::console::parse_argument(argc, argv, "--model_friction", model_friction) == -1)
		{
			LOAD_VALUE(model_friction);
		}
		if (model_friction < 0.0)
		{
			error_exit("model_friction should be larger than 1.0");
		}

		if (pcl::console::parse_argument(argc, argv, "--normalize_length", normalize_length) == -1)
		{
			LOAD_VALUE(normalize_length);
		}
		if (normalize_length < 0.0)
		{
			error_exit("normalize_length should be larger than 0.0");
		}

		if (pcl::console::parse_argument(argc, argv, "--downsample_target_points_num", downsample_target_points_num) == -1)
		{
			LOAD_VALUE(downsample_target_points_num);
		}
		if (downsample_target_points_num < 1)
		{
			error_exit("downsample_target_points_num should be larger than 0");
		}

		if (pcl::console::parse_argument(argc, argv, "--downsample_initial_leaf_size", downsample_initial_leaf_size) == -1)
		{
			LOAD_VALUE(downsample_initial_leaf_size);
		}
		if (downsample_initial_leaf_size < 0.0)
		{
			error_exit("downsample_initial_leaf_size should be larger than 0.0");
		}

		if (pcl::console::parse_argument(argc, argv, "--downsample_factor", downsample_factor) == -1)
		{
			LOAD_VALUE(downsample_factor);
		}
		if (downsample_factor < 1.0)
		{
			error_exit("downsample_factor should be larger than 1.0");
		}

		if (pcl::console::find_switch(argc, argv, "--visualization"))
		{
			visualization = true;
		}
		else
		{
			LOAD_VALUE(visualization);
		}

		if (pcl::console::find_switch(argc, argv, "--capture_screenshots"))
		{
			capture_screenshots = true;
		}
		else
		{
			LOAD_VALUE(capture_screenshots);
		}

		if (capture_screenshots)
		{
			if (!visualization)
			{
				pcl::console::print_error("Cannot enable capture_screenshots without visualization. capture_screenshots is set disabled.");
				capture_screenshots = false;
			}
			else
			{
				if (pcl::console::parse_argument(argc, argv, "--precapture_wait", precapture_wait) == -1)
				{
					LOAD_VALUE(precapture_wait);
				}
				if (precapture_wait < 0.0)
				{
					error_exit("precapture_wait should be larger than 0.0");
				}

				if (pcl::console::parse_argument(argc, argv, "--save_screenshots_dir", save_screenshots_dir) == -1)
				{
					LOAD_VALUE(save_screenshots_dir);
				}
				absolute_path(save_screenshots_dir);
				boost::filesystem::create_directories(save_screenshots_dir);
			}
		}
	}

	void show_help()
	{
		std::cout
			<< "Usage: " << "visibleSceneMaker [setting.ini] [Options] \n\n"
			<< "Options: \n"
			<< "    -h, --help             : Show this help \n"
			<< "    --load_model_path [filepath]            \n"
			<< "    --save_centering_model_path [filepath]  \n"
			<< "    --save_pointcloud_path [filepath]       \n"
			<< "    --save_transforms_path [filepath]       \n"
			<< "    --save_screenshots_dir [dirpath]        \n"
			<< "    --model_num [number]                    \n"
			<< "    --precapture_wait [time]                \n"
			<< "    --falling_interval [number]             \n"
			<< "    --postsimulation_wait [time]            \n"
			<< "    --random_seed [number]                  \n"
			<< "    --cup_r [number]                        \n"
			<< "    --cup_h [number]                        \n"
			<< "    --cup_restitution [number]              \n"
			<< "    --cup_friction [number]                 \n"
			<< "    --model_restitution [number]            \n"
			<< "    --model_friction [number]               \n"
			<< "    --normalize_length [number]             \n"
			<< "    --downsample_target_points_num [number] \n"
			<< "    --downsample_initial_leaf_size [number] \n"
			<< "    --downsample_factor [number]            \n"
			<< "    --visualization                         \n"
			<< "    --capture_screenshots                   \n"
			<< " \n"
			<< "Note: Command line options will override parameters in the setting.ini. \n"
			<< std::endl;
	}
};

class VisibleSceneMakerSetting : public SettingBase
{
public:
	std::string load_path;
	std::string save_path;
	double lambda;
	double camera_1_x;
	double camera_1_y;
	double camera_1_z;
	double camera_2_x;
	double camera_2_y;
	double camera_2_z;
	bool visualization;

	VisibleSceneMakerSetting(int argc, char** argv) : SettingBase(argc, argv, "visibleSceneMaker")
	{
		if (pcl::console::find_switch(argc, argv, "-h"))
		{
			show_help();
			exit(0);
		}

		if (pcl::console::parse_argument(argc, argv, "--load_path", load_path) == -1)
		{
			LOAD_VALUE(load_path);
		}
		absolute_path(load_path);
		if (!boost::filesystem::exists(load_path))
		{
			error_exit("load_path does not exist.");
		}

		if (pcl::console::parse_argument(argc, argv, "--save_path", save_path) == -1)
		{
			LOAD_VALUE(save_path);
		}
		absolute_path(save_path);
		boost::filesystem::create_directories(boost::filesystem::path(save_path).parent_path());
		
		if (pcl::console::parse_argument(argc, argv, "--lambda", lambda) == -1)
		{
			LOAD_VALUE(lambda);
		}
		if (lambda < 0.0)
		{
			error_exit("lambda should be larger than 0.0");
		}

		if (pcl::console::find_switch(argc, argv, "--visualization"))
		{
			visualization = true;
		}
		else
		{
			LOAD_VALUE(visualization);
		}

		if (pcl::console::parse_3x_arguments(argc, argv, "--camera_1", camera_1_x, camera_1_y, camera_1_z) == -1)
		{
			LOAD_VALUE(camera_1_x);
			LOAD_VALUE(camera_1_y);
			LOAD_VALUE(camera_1_z);
		}

		if (pcl::console::parse_3x_arguments(argc, argv, "--camera_2", camera_2_x, camera_2_y, camera_2_z) == -1)
		{
			LOAD_VALUE(camera_2_x);
			LOAD_VALUE(camera_2_y);
			LOAD_VALUE(camera_2_z);
		}
	}

	void show_help()
	{
		std::cout
			<< "Usage: " << "visibleSceneMaker [setting.ini] [Options] \n\n"
			<< "Options: \n"
			<< "    -h, --help             : Show this help \n"
			<< "    --load_path [filepath] : load pointcloud, .ply or .pcd \n"
			<< "    --save_path [filepath] : save pointcloud, .ply or .pcd \n"
			<< "    --lambda [number]      : density of pixel \n"
			<< "    --camera_1 [X],[Y],[Z] : Camera 1 position \n"
			<< "    --camera_2 [X],[Y],[Z] : Camera 2 position \n"
			<< "    --visualization        : Enable visualization \n"
			<< " \n"
			<< "Note: Command line options will override parameters in the setting.ini. \n"
			<< std::endl;
	}
};

#undef LOAD_VALUE