#define _CRT_SECURE_NO_WARNINGS
#pragma warning(disable:4819)

#include <iostream>
#include <thread>
#include <random>
#include <memory>
#include <chrono>

#include <omp.h>

#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/surface/gp3.h>
#include <pcl/common/geometry.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <bullet/btBulletDynamicsCommon.h>
#include <bullet/LinearMath/btIDebugDraw.h>

#include "convexDecomposition.hpp"
#include "setting.hpp"

template<typename T> constexpr const T pi() { return static_cast<T>(std::atan2(0.0, -1.0)); }

int main(int argc, char* argv[])
{
	const BinSceneMakerSetting setting(argc, argv);

	const pcl::PointCloud<pcl::PointXYZ>::Ptr model_load(new pcl::PointCloud<pcl::PointXYZ>());

	//
	// load model point cloud
	//

	// model load
	if (boost::filesystem::path(setting.load_model_path).extension() == ".pcd")
	{
		if (pcl::io::loadPCDFile(setting.load_model_path, *model_load) == -1)
		{
			error_exit("Model load error.");
		}
	}
	else if (boost::filesystem::path(setting.load_model_path).extension() == ".ply")
	{
		if (pcl::io::loadPLYFile(setting.load_model_path, *model_load) == -1)
		{
			error_exit("Model load error.");
		}
	}
	else
	{
		error_exit("PLY or PCD file are only available to load.");
	}

	// centering
	Eigen::Vector3d sum_of_pos = Eigen::Vector3d::Zero();
	for (const auto& p : *(model_load)) sum_of_pos += p.getVector3fMap().cast<double>();

	Eigen::Matrix4d transform_centering = Eigen::Matrix4d::Identity();
	transform_centering.topRightCorner<3, 1>() = -sum_of_pos / model_load->size();

	pcl::transformPointCloud(*model_load, *model_load, transform_centering);

	// save centering model
	if (boost::filesystem::path(setting.save_centering_model_path).extension() == ".pcd")
	{
		pcl::io::savePCDFileBinary(setting.save_centering_model_path, *model_load);
	}
	else if (boost::filesystem::path(setting.save_centering_model_path).extension() == ".ply")
	{
		pcl::io::savePLYFileBinary(setting.save_centering_model_path, *model_load);
	}
	else
	{
		error_exit("PLY or PCD file are only available to save.");
	}
	
	//downsample less than 2000 points
	const pcl::PointCloud<pcl::PointXYZ>::Ptr model_downsampled(new pcl::PointCloud<pcl::PointXYZ>(*model_load));

	double leaf_size = setting.downsample_initial_leaf_size;
	while (model_downsampled->size() > setting.downsample_target_points_num)
	{
		pcl::VoxelGrid<pcl::PointXYZ> sor;
		sor.setInputCloud(model_load);
		sor.setLeafSize(leaf_size, leaf_size, leaf_size);
		sor.filter(*model_downsampled);
		leaf_size *= setting.downsample_factor;
		std::cout << "size = " << model_downsampled->size() << ", leaf_size: " << leaf_size << std::endl;;
	}

	// detect scaling factor
	double max_dist = 0.0f;

	for (int i = 0; i < model_downsampled->size(); ++i)
	{
		for (int j = 0; j < i; ++j)
		{
			const double dist = pcl::geometry::distance((*model_downsampled)[i], (*model_downsampled)[j]);
			if (dist > max_dist)
			{
				max_dist = dist;
			}
		}
	}

	// make scaled model
	const pcl::PointCloud<pcl::PointXYZ>::Ptr model(new pcl::PointCloud<pcl::PointXYZ>);
	const double scale = setting.normalize_length / max_dist;

	for (const auto& point : *model_downsampled)
	{
		model->push_back(pcl::PointXYZ(point.x * scale, point.y * scale, point.z * scale));
	}
	std::cout << "Max distance: " << max_dist << std::endl;
	std::cout << "Scale: " << scale << std::endl;

	//
	// construct polygon model
	//

	// kdtree of model
	const pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree_model(new pcl::search::KdTree<pcl::PointXYZ>);
	kdtree_model->setInputCloud(model);

	// normal estimation
	pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> ne;
	const pcl::PointCloud<pcl::Normal>::Ptr model_normal(new pcl::PointCloud<pcl::Normal>);
	const pcl::PointCloud<pcl::PointNormal>::Ptr model_point_normal(new pcl::PointCloud<pcl::PointNormal>);
	ne.setInputCloud(model);
	ne.setSearchMethod(kdtree_model);
	ne.setKSearch(20);
	ne.compute(*model_normal);
	pcl::concatenateFields(*model, *model_normal, *model_point_normal);
	pcl::search::KdTree<pcl::PointNormal>::Ptr kdtree_model_normal(new pcl::search::KdTree<pcl::PointNormal>);
	kdtree_model_normal->setInputCloud(model_point_normal);
	std::cout << "Normal Estimation: Finished." << std::endl;

	// construct polygon mesh
	pcl::PolygonMesh poly_mesh;
	pcl::GreedyProjectionTriangulation<pcl::PointNormal> gpt;
	gpt.setSearchRadius(100.0);
	gpt.setMu(2.5);
	gpt.setMaximumNearestNeighbors(150);
	gpt.setMaximumSurfaceAngle(pi<double>() / 4.0);
	gpt.setMinimumAngle(pi<double>() / 18.0);
	gpt.setMaximumAngle(2.0 * pi<double>() / 3.0);
	gpt.setNormalConsistency(false);
	gpt.setInputCloud(model_point_normal);
	gpt.setSearchMethod(kdtree_model_normal);
	gpt.reconstruct(poly_mesh);
	std::cout << "PolyMesh Construct: Finished." << std::endl;
	
	//visualize polygon
	boost::optional<pcl::visualization::PCLVisualizer::Ptr> polygon_viewer;

	if (setting.visualization)
	{
		polygon_viewer = pcl::visualization::PCLVisualizer::Ptr(new pcl::visualization::PCLVisualizer("Polygon"));
		(*polygon_viewer)->addPolygonMesh(poly_mesh);

		(*polygon_viewer)->spinOnce();
	}

	//
	// construct model 3D rigid body for Bullet
	//

	// convex decomposition object
	const pcl::PointCloud<pcl::PointNormal>::Ptr mesh_cloud(new pcl::PointCloud<pcl::PointNormal>);
	pcl::fromPCLPointCloud2(poly_mesh.cloud, *mesh_cloud);

	MyConvexDecomposition convexDecomposition;
	gContactAddedCallback = &MyContactCallback;

	// HACD
	// Thanks for https://github.com/kripken/bullet/blob/master/Demos/ConvexDecompositionDemo/ConvexDecompositionDemo.cpp
	std::vector<HACD::Vec3<HACD::Real>> hacd_points;
	std::vector<HACD::Vec3<long>> hacd_triangles;

	for (const auto& point : *mesh_cloud)
	{
		HACD::Vec3<HACD::Real> vertex(point.x, point.y, point.z);
		hacd_points.push_back(vertex);
	}

	for (const auto& triangle : poly_mesh.polygons)
	{
		HACD::Vec3<long> triangle_local(triangle.vertices[0], triangle.vertices[1], triangle.vertices[2]);
		hacd_triangles.push_back(triangle_local);
	}

	HACD::HACD hacd;
	hacd.SetPoints(&hacd_points[0]);
	hacd.SetNPoints(hacd_points.size());
	hacd.SetTriangles(&hacd_triangles[0]);
	hacd.SetNTriangles(hacd_triangles.size());
	hacd.SetCompacityWeight(0.1);
	hacd.SetVolumeWeight(0.0);
	hacd.SetNClusters(2);
	hacd.SetNVerticesPerCH(100);
	hacd.SetConcavity(100);
	hacd.SetAddExtraDistPoints(false);
	hacd.SetAddNeighboursDistPoints(false);
	hacd.SetAddFacesPoints(false);
	hacd.Compute();

	const int nClusters = static_cast<int>(hacd.GetNClusters());

	for (int c = 0; c < nClusters; c++)
	{
		const int nPoints = static_cast<int>(hacd.GetNPointsCH(c));
		const int nTriangles = static_cast<int>(hacd.GetNTrianglesCH(c));

		std::vector<float> vertices(nPoints * 3);
		std::vector<unsigned int> triangles(nTriangles * 3);

		std::vector<HACD::Vec3<HACD::Real>> pointsCH(nPoints);
		std::vector<HACD::Vec3<long>> trianglesCH(nTriangles);
		hacd.GetCH(c, pointsCH.data(), trianglesCH.data());

		// points
		for (int v = 0; v < nPoints; v++)
		{
			vertices[3 * v + 0] = pointsCH[v].X();
			vertices[3 * v + 1] = pointsCH[v].Y();
			vertices[3 * v + 2] = pointsCH[v].Z();
		}
		// triangles
		for (int f = 0; f < nTriangles; f++)
		{
			triangles[3 * f + 0] = trianglesCH[f].X();
			triangles[3 * f + 1] = trianglesCH[f].Y();
			triangles[3 * f + 2] = trianglesCH[f].Z();
		}

		ConvexDecomposition::ConvexResult r(nPoints, vertices.data(), nTriangles, triangles.data());
		convexDecomposition.ConvexDecompResult(r);
	}

	btCompoundShape* const fallbody_compound = new btCompoundShape;
	btTransform trans;
	trans.setIdentity();

	for (int i = 0; i < convexDecomposition.m_convexShapes.size(); i++)
	{
		btVector3 centroid = convexDecomposition.m_convexCentroids[i];
		trans.setOrigin(centroid);
		auto convexShape = convexDecomposition.m_convexShapes[i];
		convexShape->setMargin(0.0);
		fallbody_compound->addChildShape(trans, convexShape);

	}
	fallbody_compound->setMargin(leaf_size);

	std::cout << "Bullet Compound Construction: Finished." << std::endl;

	//
	// initialize bullet
	//

	//bullet world construct
	btDefaultCollisionConfiguration* const config = new btDefaultCollisionConfiguration;
	btCollisionDispatcher* const dispatcher = new btCollisionDispatcher(config);
	btDbvtBroadphase* const broadphase = new btDbvtBroadphase;
	btSequentialImpulseConstraintSolver* const solver = new btSequentialImpulseConstraintSolver;
	btDynamicsWorld* const dynamicsWorld = new btDiscreteDynamicsWorld(dispatcher, broadphase, solver, config);

	dynamicsWorld->setGravity(btVector3(0, -9.8, 0));
	std::cout << "Bullet dynamicsWorld construct: Finished." << std::endl;

	//construct cup
	btTriangleMesh* const mesh_cup = new btTriangleMesh;
	const pcl::PointCloud<pcl::PointXYZ>::Ptr cup_ground(new pcl::PointCloud<pcl::PointXYZ>());

	const double y_step = setting.normalize_length / 10.0;
	const double theta_step = pi<double>() / 60.0;
	for (double theta = 0; theta < pi<double>() * 2.0; theta += theta_step)
	{
		for (double y = - setting.normalize_length / 10.0; y < setting.normalize_length * setting.cup_h; y += y_step)
		{
			const double x1 = setting.normalize_length * setting.cup_r * std::cos(theta);
			const double z1 = setting.normalize_length * setting.cup_r * std::sin(theta);
			const double x2 = setting.normalize_length * setting.cup_r * std::cos(theta + theta_step);
			const double z2 = setting.normalize_length * setting.cup_r * std::sin(theta + theta_step);
			{
				const btVector3 PointA(x1, y, z1);
				const btVector3 PointB(x1, y + y_step, z1);
				const btVector3 PointC(x2, y, z2);
				mesh_cup->addTriangle(PointA, PointB, PointC);
			}
			{
				const btVector3 PointA(x1, y + y_step, z1);
				const btVector3 PointB(x2, y + y_step, z2);
				const btVector3 PointC(x2, y, z2);
				mesh_cup->addTriangle(PointA, PointB, PointC);
			}
			cup_ground->push_back(pcl::PointXYZ(x1, y, z1));
		}

		for (double r = 0.0; r < setting.normalize_length * setting.cup_r; r += y_step)
		{
			const double x1 = r * std::cos(theta);
			const double z1 = r * std::sin(theta);
			const double x2 = r * std::cos(theta + theta_step);
			const double z2 = r * std::sin(theta + theta_step);
			{
				const btVector3 PointA(x1, 0.0, z1);
				const btVector3 PointB(x1, 0.0, z2);
				const btVector3 PointC(x2, 0.0, z1);
				mesh_cup->addTriangle(PointA, PointB, PointC);
			}
			{
				const btVector3 PointA(x1, 0.0, z2);
				const btVector3 PointB(x2, 0.0, z1);
				const btVector3 PointC(x2, 0.0, z2);
				mesh_cup->addTriangle(PointA, PointB, PointC);
			}
			cup_ground->push_back(pcl::PointXYZ(x1, 0.0, z1));
		}
	}
	btBvhTriangleMeshShape* const cup_collision_shape = new btBvhTriangleMeshShape(mesh_cup, true);
	btDefaultMotionState* const cup_motion_state = new btDefaultMotionState(btTransform(btQuaternion(0, 0, 0, 1)));
	const btRigidBody::btRigidBodyConstructionInfo cup_rigidbodyCI(0, cup_motion_state, cup_collision_shape);
	btRigidBody* const cup_body = new btRigidBody(cup_rigidbodyCI);
	cup_body->setRestitution(setting.cup_restitution);
	cup_body->setFriction(setting.cup_friction);

	dynamicsWorld->addRigidBody(cup_body);

	std::cout << "Bullet Ground Construct: Finished." << std::endl;

	// prepare visualizer
	boost::optional<pcl::visualization::PCLVisualizer::Ptr> viewer;
	if (setting.visualization)
	{
		viewer = pcl::visualization::PCLVisualizer::Ptr(new pcl::visualization::PCLVisualizer("Simulator"));
		(*viewer)->addPointCloud(cup_ground);
	}

	// wait for camera position setting
	if (setting.capture_screenshots)
	{
		std::cout << "Waiting for your camera setting for capture." << endl;

		const int wait_time = static_cast<int>(setting.precapture_wait * 1000);
		for (int i = 0; i < wait_time; ++i)
		{
			(*viewer)->spinOnce();
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
			if (i % 1000 == 0) std::cout << i / 1000 << " / " << int(wait_time / 1000) << endl;
		}
	}

	//simulation setting initialize
	const double dt = 1.0 / 60.0;
	int cnt = setting.falling_interval;
	int cnt_save = 0;
	int now_model_num = 0;
	int post_sim_wait = setting.postsimulation_wait;
	std::mt19937 mt(setting.random_seed);
	std::normal_distribution<double> norm_db(0, 1);
	std::uniform_real_distribution<double> unif_db(0, 1);

	std::cout << "Start simulation..." << endl;

	// main simulation loop
	while (now_model_num < setting.model_num || post_sim_wait > 0)
	{
		dynamicsWorld->stepSimulation(dt);
		++cnt;

		// add next rigid body
		if (cnt > setting.falling_interval)
		{
			cnt = 0;

			if (now_model_num < setting.model_num)
			{
				// get the highest body's centroid
				double max_y = 0.0;
				for (int idx = 0; idx < dynamicsWorld->getNumCollisionObjects(); idx++)
				{
					btRigidBody* fallBody = btRigidBody::upcast(dynamicsWorld->getCollisionObjectArray()[idx]);
					if (fallBody && fallBody->getMotionState() && fallBody != cup_body)
					{
						btTransform trans;
						fallBody->getMotionState()->getWorldTransform(trans);
						float y = trans.getOrigin().y();
						if (y > max_y) max_y = y;
					}
				}

				// generate random position and rotation
				const Eigen::Vector3f random_axis(unif_db(mt) * 2.0 - 1.0, unif_db(mt) * 2.0 - 1.0, unif_db(mt) * 2.0 - 1.0);
				const double random_angle = unif_db(mt) * 2.0 * pi<double>();
				const Eigen::Quaternionf random_rotation_eigen = Eigen::Quaternionf(Eigen::AngleAxisf(random_angle, random_axis));
				const btQuaternion random_rotation(random_rotation_eigen.x(), random_rotation_eigen.y(), random_rotation_eigen.z(), random_rotation_eigen.w());
				const btVector3 random_posision = btVector3(0.01 * norm_db(mt), max_y + setting.normalize_length * 2.0, 0.01 * norm_db(mt));
				btScalar mass = btScalar(100.0);
				btVector3 inertia = btVector3(0, 0, 0);
				fallbody_compound->calculateLocalInertia(mass, inertia);

				btDefaultMotionState* motion_state = new btDefaultMotionState(btTransform(random_rotation, random_posision));
				btRigidBody* const body = new btRigidBody(mass, motion_state, fallbody_compound, inertia);
				body->setRestitution(setting.model_restitution);
				body->setFriction(setting.model_friction);
				body->setSleepingThresholds(0.2, 2);
				body->setCollisionFlags(body->getCollisionFlags() | btCollisionObject::CF_CUSTOM_MATERIAL_CALLBACK);
				dynamicsWorld->addRigidBody(body);

				if (setting.visualization)
				{
					const std::string cloud_name = std::to_string(now_model_num) + "model";
					(*viewer)->addPointCloud(model, cloud_name);
				}
				++now_model_num;

			}
		}

		if (now_model_num >= setting.model_num)
		{
			--post_sim_wait;
		}

		// update point cloud's state as bullet world
		int model_body_idx = 0;
		for (int idx = 0; idx < dynamicsWorld->getNumCollisionObjects(); idx++)
		{
			btRigidBody* body = btRigidBody::upcast(dynamicsWorld->getCollisionObjectArray()[idx]);
			if (body && body->getMotionState() && body != cup_body)
			{
				if (setting.visualization)
				{
					const std::string cloud_name = std::to_string(model_body_idx) + "model";
					++model_body_idx;
					btTransform trans;
					body->getMotionState()->getWorldTransform(trans);

					float mat_v[16];
					trans.getOpenGLMatrix(mat_v);

					(*viewer)->updatePointCloudPose(cloud_name, Eigen::Affine3f(Eigen::Map<Eigen::Matrix4f>(mat_v)));
				}
			}
		}

		if (setting.visualization)
		{
			if ((*viewer)->wasStopped() ||
				(*polygon_viewer)->wasStopped() )
			{
				break;
			}

			(*viewer)->spinOnce();
			(*polygon_viewer)->spinOnce();

			if (setting.capture_screenshots)
			{
				(*viewer)->saveScreenshot(setting.save_screenshots_dir + "/" + std::to_string(cnt_save) + ".png");
				++cnt_save;
			}
		}
	}

	std::cout << "Simulation is finished." << std::endl;

	//
	// save bin scene
	//

	// make save_cloud and output transforms
	const pcl::PointCloud<pcl::PointXYZ>::Ptr save_cloud(new pcl::PointCloud<pcl::PointXYZ>());
	std::ofstream ofs(setting.save_transforms_path);

	for (int idx = 0; idx < dynamicsWorld->getNumCollisionObjects(); idx++)
	{
		const btRigidBody* const body = btRigidBody::upcast(dynamicsWorld->getCollisionObjectArray()[idx]);
		if (body && body->getMotionState() && !body->isStaticObject())
		{
			btTransform trans;
			body->getMotionState()->getWorldTransform(trans);

			float mat_v[16];
			trans.getOpenGLMatrix(mat_v);
			mat_v[12] /= scale;
			mat_v[13] /= scale;
			mat_v[14] /= scale;

			for (int i = 0; i < 14; ++i)
			{
				ofs << mat_v[i] << ", ";
			}
			ofs << mat_v[15] << std::endl;

			const pcl::PointCloud<pcl::PointXYZ>::Ptr model_transformed(new pcl::PointCloud<pcl::PointXYZ>);
			pcl::transformPointCloud(*model_load, *model_transformed, Eigen::Affine3f(Eigen::Map<Eigen::Matrix4f>(mat_v)));
			(*save_cloud) += (*model_transformed);
		}
	}

	// save
	if (boost::filesystem::path(setting.save_pointcloud_path).extension() == ".pcd")
	{
		pcl::io::savePCDFileBinary(setting.save_pointcloud_path, *save_cloud);
	}
	else if (boost::filesystem::path(setting.save_pointcloud_path).extension() == ".ply")
	{
		pcl::io::savePLYFileBinary(setting.save_pointcloud_path, *save_cloud);
	}
	else
	{
		error_exit("PLY or PCD file are only available to save.");
	}
	
	// clean up bullet vars
	for (int idx = 0; idx < dynamicsWorld->getNumCollisionObjects(); idx++)
	{
		btRigidBody* const body = btRigidBody::upcast(dynamicsWorld->getCollisionObjectArray()[idx]);
		if (body)
		{
			dynamicsWorld->removeRigidBody(body);
			if (body->getMotionState())
			{
				delete body->getMotionState();
			}
			delete body;
		}
	}

	delete dynamicsWorld;
	delete cup_collision_shape;
	delete mesh_cup;
	delete fallbody_compound;
	delete solver;
	delete broadphase;
	delete dispatcher;
	delete config;

	// visualize
	if (setting.visualization)
	{
		pcl::visualization::PCLVisualizer::Ptr viewer_scene(new pcl::visualization::PCLVisualizer("Bin Scene"));
		viewer_scene->addPointCloud(save_cloud);

		while (true)
		{
			if ((*viewer)->wasStopped() ||
				(*polygon_viewer)->wasStopped() ||
				viewer_scene->wasStopped() )
			{
				break;
			}

			(*viewer)->spinOnce();
			(*polygon_viewer)->spinOnce();
			viewer_scene->spinOnce();
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
		}
	}

	return 0;
}