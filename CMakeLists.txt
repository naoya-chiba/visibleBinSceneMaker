cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(binSceneMaker)

#set(CMAKE_BUILD_TYPE Release)

find_package(PCL 1.8 REQUIRED)
find_package(BULLET REQUIRED)

find_package(OpenMP)
if (OPENMP_FOUND) 
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

find_package(CUDA)
if (OPENMP_FOUND) 
	set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}; -Xcompiler \"/wd 4819\" -arch=sm_61")
	option(USE_CUDA "Enable CUDA" TRUE)
else()
	option(USE_CUDA "Enable CUDA" FALSE)
endif()

if (USE_CUDA)
	add_definitions(-DUSE_GPU)
	message(STATUS "CUDA Enabled.")
endif()


include_directories(${PCL_INCLUDE_DIRS} ${BULLET_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS} "${BULLET_INCLUDE_DIRS}/../lib")
add_definitions(${PCL_DEFINITIONS})

add_executable(binSceneMaker binSceneMaker.cpp convexDecomposition.hpp setting.hpp)
target_link_libraries(binSceneMaker ${PCL_LIBRARIES} ${BULLET_LIBRARIES}
	optimized BulletCollision debug BulletCollision_Debug
	optimized BulletDynamics debug BulletDynamics_Debug
	optimized BulletInverseDynamics debug BulletInverseDynamics_Debug
	optimized HACD debug HACD_Debug)

cuda_add_executable (visibleSceneMaker visibleSceneMaker.cpp setting.hpp visible_check.cu visible_check.cpp visible_check.h visible_check.cuh)
target_link_libraries (visibleSceneMaker ${PCL_LIBRARIES})
