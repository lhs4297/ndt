cmake_minimum_required(VERSION 3.0.2)
project(localization)

# ROS 패키지 요구사항
find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  tf
  pcl_conversions
)

# Eigen3 찾기
find_package(Eigen3 REQUIRED)

# PCL 라이브러리 찾기
find_package(PCL REQUIRED)

# 병렬 처리를 위한 thread 라이브러리 추가
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# catkin_package 설정
catkin_package(
  CATKIN_DEPENDS roscpp std_msgs tf pcl_conversions
)

# include 디렉토리 설정
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR} # Eigen3 헤더 디렉토리 포함
  ${PCL_INCLUDE_DIRS}
)

# PCL 라이브러리 연결 설정
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# localization 노드 추가
add_executable(loc_node src/computeDerivatives.cpp)
target_link_libraries(loc_node
  ${catkin_LIBRARIES}
  Eigen3::Eigen
  ${PCL_LIBRARIES} # PCL 라이브러리 추가
)

# my_loc 노드 추가
add_executable(my_loc src/my_loc.cpp)
target_link_libraries(my_loc
  ${catkin_LIBRARIES}
  Eigen3::Eigen
  ${PCL_LIBRARIES} # PCL 라이브러리 추가
)
