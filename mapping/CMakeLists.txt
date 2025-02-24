cmake_minimum_required(VERSION 3.0.2)
project(mapping)

# ROS 패키지 설정
find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
)

# C++ 표준 설정
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 의존성 패키지 설정
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(OpenMP REQUIRED)
find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED) # GLFW
find_package(Qhull REQUIRED) # Qhull 설정
find_package(fmt REQUIRED) # fmt 라이브러리 설정


# include 디렉토리 설정
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
  ${GLFW_INCLUDE_DIRS}
  ${QHULL_INCLUDE_DIRS}
)

link_directories(${QHULL_LIBRARY_DIRS})


# OpenGL 정책 설정 (GLVND 사용)
cmake_policy(SET CMP0072 NEW)
set(OpenGL_GL_PREFERENCE "GLVND")

# Open3D 경로 설정
include_directories("/LHS/self_lib/Open3D/cpp")
include_directories("/LHS/self_lib/Open3D/cpp/open3d")
link_directories("/LHS/self_lib/Open3D/build/lib")


# JsonCpp 설정
include_directories("/usr/include/jsoncpp")
link_directories("/usr/lib/x86_64-linux-gnu")

# catkin_package 설정
catkin_package()


# map_node 실행 파일 추가
add_executable(map_node src/mapping.cpp)

# map_node에 라이브러리 링크
target_link_libraries(map_node
    PRIVATE
        ${catkin_LIBRARIES}
        ${PCL_LIBRARIES}
        Eigen3::Eigen
        OpenMP::OpenMP_CXX
        OpenGL::GL
        qhullcpp
        qhull
        fmt::fmt
        glfw
        Open3D
)

add_executable(test_node src/test.cpp)

# map_node에 라이브러리 링크
target_link_libraries(test_node
    PRIVATE
        ${catkin_LIBRARIES}
        ${PCL_LIBRARIES}
        Eigen3::Eigen
        OpenMP::OpenMP_CXX
        OpenGL::GL
        qhullcpp
        qhull
        fmt::fmt
        glfw
        Open3D
)