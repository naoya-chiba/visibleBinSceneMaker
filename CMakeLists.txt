cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(binSceneMaker)

#set(CMAKE_BUILD_TYPE Release)

find_package(PCL 1.8 REQUIRED)
find_package(BULLET)
find_package(OpenMP)
if (OPENMP_FOUND) 
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()
find_package(CUDA REQUIRED)

include_directories(${PCL_INCLUDE_DIRS} ${BULLET_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS} "C:/lib/Bullet/install/lib")
add_definitions(${PCL_DEFINITIONS})

add_executable (binSceneMaker binSceneMaker.cpp convexDecomposition.hpp setting.hpp)
target_link_libraries (binSceneMaker ${PCL_LIBRARIES} ${BULLET_LIBRARIES} 
	optimized LinearMath debug LinearMath_Debug
	optimized BulletInverseDynamics debug BulletInverseDynamics_Debug
	optimized BulletCollision debug BulletCollision_Debug
	optimized BulletDynamics debug BulletDynamics_Debug
	optimized BulletSoftBody debug BulletSoftBody_Debug
	optimized HACD debug HACD_Debug)

cuda_add_executable (visibleSceneMaker visibleSceneMaker.cpp setting.hpp visible_check.cu visible_check.cpp visible_check.h visible_check.cuh)
target_link_libraries (visibleSceneMaker ${PCL_LIBRARIES})
