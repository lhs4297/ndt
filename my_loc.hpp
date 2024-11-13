#ifndef MYLOC_HPP
#define MYLOC_HPP

#pragma once

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <Eigen/Dense>


#include <ros/ros.h>

#include <tf/transform_broadcaster.h> 

#include <sensor_msgs/PointCloud2.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/ndt.h>
#include <pcl/filters/voxel_grid.h>  // VoxelGrid 사용을 위한 헤더 추가

//백그라운드 ndt 실행
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>

// 커스텀 ndt
#include "ndt.h"



// 가우시안 셀 구조체
struct GaussianCell {
    Eigen::Vector3d stGauCell_vt3_d_mean;               // 평균
    Eigen::Matrix3d stGauCell_mat3_d_cov;         // 공분산 행렬
    Eigen::Matrix3d stGauCell_mat3_d_inv_cov; // 공분산 행렬의 역행렬
    double stGauCell_d_det;                 // 공분산 행렬의 행렬식
};

class LOCALIZATION
{
public:
    
    LOCALIZATION(ros::NodeHandle& nh);
    ~LOCALIZATION();

    // pcd파일 불러오기 & 맵 일부만 가져오기 & point cloud화 해주기
    pcl::PointCloud<pcl::PointXYZ>::Ptr LoadPcdFile(const std::string& pcd_dir);
    // map 퍼블리쉬 (시각화 할때 사용)
    void PublishPointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, ros::Publisher& pub, const std::string& frame_id);
    // 입력데이터 Callback 함수
    void NDTCallback(const sensor_msgs::PointCloud2ConstPtr& msg);
    void ProcessNDT(const sensor_msgs::PointCloud2ConstPtr& msg);

    // 스레드함수
    void NDTProcessingThread();
    
    void setInputTarget(const std::vector<pcl::PointXYZ>& target_points);
    void setInputTarget(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);
    void setInputSource(const std::vector<pcl::PointXYZ>& source_points);
    void setInputSource(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);


    void buildTargetCells();

    void startNDTThread();

    std::vector<pcl::PointXYZ> convertToPointVector(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);

    // 정규화 상수 계산 함수
    void computeNormalizationConstants();
    // step size adaptive하게
    double lineSearch(const Eigen::Matrix<double, 6, 1>& parameters,
                        const Eigen::Matrix<double, 6, 1>& delta,
                        double initial_score,
                        const Eigen::Matrix<double, 6, 1>& gradient);
    
    
private:

    //라이브러리 선언
    pcl::custom::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> m_ndt;

    // 숫자 추출 함수(맵불러올때)
    int extract_number(const std::string& i_file_path);
    // 포즈 데이터를 로드하는 함수
    std::vector<std::array<std::array<float, 4>, 4>> load_poses(const std::string& pose_file_name,
                                                                const std::array<std::array<float, 4>, 4>& Tr,
                                                                const std::array<std::array<float, 4>, 4>& Tr_inv);
    // 캘리브레이션 데이터를 로드하는 함수
    std::tuple<std::array<std::array<float, 4>, 4>, std::array<std::array<float, 4>, 4>> load_calibration(
                                                                                        const std::string& calib_file_name);


public:
    pcl::PointCloud<pcl::PointXYZ>::Ptr m_pc_map_cloud;
    ros::Publisher m_ros_map_pub;
    //pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> m_ndt_ndt;

    Eigen::Matrix4d m_matrix4d_initial_esti;
    Eigen::Matrix4d m_matrix4d_prev;

    double m_cfg_d_trans_error_allow;
    double m_cfg_d_rot_error_allow;
    float m_cfg_f_step_size_m;
    float m_cfg_f_grid_size_m;
    int m_cfg_int_iterate_max;
    double m_cfg_d_gauss_k2;
    double m_cfg_d_gauss_k1;

    int ndt_iter;
    int m_pose_num;

    pcl::PointCloud<pcl::PointXYZ>::Ptr m_filtered_map;

    uint32_t m_cfg_uint32_message_count = 0;

    std::vector<pcl::PointXYZ> inputed_target_points;
    std::vector<pcl::PointXYZ> inputed_source_points;


    //백그라운드 ndt실행
    std::queue<sensor_msgs::PointCloud2ConstPtr> m_lidar_data_queue;
    std::mutex m_queue_mutex;
    std::condition_variable m_data_condition;
    std::thread m_ndt_thread;
    bool m_is_running;
    bool m_cfg_b_debug_mode;


private:
    pcl::PointCloud<pcl::PointXYZ>::Ptr m_pc_trajectory_cloud;
    pcl::PointCloud<pcl::PointXYZ>::Ptr m_pc_predict_cloud;
    ros::Publisher m_ros_trajectory_pub;
    ros::Publisher m_ros_aligned_pub;
    ros::Publisher m_ros_filtered_map_pub;
    ros::Publisher m_ros_filtered_input_pub;
    ros::Publisher m_ros_predict_pub;
    ros::Subscriber m_ros_point_cloud_sub;
    tf::TransformBroadcaster m_tf_broadcaster_br;

    std::vector<GaussianCell> m_vt_stGauCell_gaussian_cells;


    int nr_iterations_;                 // 반복 횟수
    bool converged_;                    // 수렴 여부
    pcl::VoxelGrid<pcl::PointXYZ> target_cells_;            // 타겟 셀 (VoxelGrid 형태)
    double outlier_ratio_;              // 아웃라이어 비율
    Eigen::Affine3d final_transformation_; // 최종 변환 행렬
    Eigen::Matrix4d previous_transformation_;

    std::vector<std::array<std::array<float, 4>, 4>> m_vt_f_44_poses;



};

#endif