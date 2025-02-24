#ifndef MYLOC_HPP
#define MYLOC_HPP

#pragma once

#include <string>

#include <pcl/point_cloud.h> // LoadPcdFile
#include <pcl/point_types.h> // LoadPcdFile
#include <pcl/io/pcd_io.h> // LoadPcdFile

#include <sensor_msgs/PointCloud2.h> // publish

#include <ros/ros.h> // NodeHandle

//백그라운드 ndt 실행
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>


// 가우시안 셀 구조체
struct GaussianCell {
    Eigen::Vector3d stGauCell_vec3d_mean;               // 평균
    Eigen::Matrix3d stGauCell_mat3d_cov;         // 공분산 행렬
    Eigen::Matrix3d stGauCell_mat3d_inv_cov; // 공분산 행렬의 역행렬
    double stGauCell_d_det;                 // 공분산 행렬의 행렬식

    Eigen::Vector3d getMean() const { return stGauCell_vec3d_mean; }
    Eigen::Matrix3d getInverseCov() const { return stGauCell_mat3d_inv_cov; }
};



class LOCALIZATION
{
public:
 
    LOCALIZATION(ros::NodeHandle& nh);
    ~LOCALIZATION();

    void startNDTThread();

    pcl::PointCloud<pcl::PointXYZ>::Ptr LoadPcdFile(const std::string& pcd_dir);
    void PublishPointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                             ros::Publisher& pub,
                              const std::string& frame_id);
private:
    //기초
    int extract_number(const std::string& i_file_path);
    std::vector<pcl::PointXYZ> convertToPointVector(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);
    
    //백그라운드 쓰레드
    void NDTProcessingThread();
    void ProcessNDT(const sensor_msgs::PointCloud2ConstPtr& msg);
    void NDTCallback(const sensor_msgs::PointCloud2ConstPtr& msg);

    //NDT 옵션
    void setInputTarget(const std::vector<pcl::PointXYZ>& target_points);
    void setInputTarget(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);
    void setInputSource(const std::vector<pcl::PointXYZ>& source_points);
    void setInputSource(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);
    void buildTargetCells();
    

public:
    Eigen::Matrix4d m_matrix4d_initial_esti;
    pcl::PointCloud<pcl::PointXYZ>::Ptr m_pcl_map_cloud;
    ros::Publisher m_ros_map_pub;

    std::vector<GaussianCell> m_vec_stGauCell_gaussian_cells;

private:
    bool m_b_debugmode;
    uint32_t m_cfg_uint32_message_count = 0;

    // 파라미터
    float m_f_radius_m;
    float m_cfg_f_grid_size_m;

    //백그라운드 쓰레드
    std::queue<sensor_msgs::PointCloud2ConstPtr> m_std_lidar_data_queue;
    std::mutex m_std_queue_mutex;
    std::condition_variable m_std_data_condition;
    std::thread m_std_thread;
    bool m_b_is_running;
    bool m_b_debug_mode;

    std::vector<pcl::PointXYZ> m_pcl_inputed_target_points;
    std::vector<pcl::PointXYZ> m_pcl_inputed_source_points;


    pcl::PointCloud<pcl::PointXYZ>::Ptr m_pcl_filtered_map_ptr;
    ros::Publisher m_ros_filtered_map_pub;
    ros::Publisher m_ros_filtered_input_pub;

    ros::Subscriber m_ros_point_cloud_sub;
    
    

};


#endif