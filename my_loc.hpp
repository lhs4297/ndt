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
    //float computeScoreAndGradient(const Eigen::Matrix4f& transformation, Eigen::VectorXf& gradient);
    double trialValueSelectionMT(double a_l, double f_l, double g_l, double a_u, double f_u,
                                    double g_u, double a_t, double f_t, double g_t) const;
    bool updateIntervalMT(double& a_l, double& f_l, double& g_l, double& a_u,
                            double& f_u, double& g_u, double a_t, double f_t, double g_t) const;

    void computeHessian(Eigen::Matrix<double, 6, 6>& hessian, 
                                    const pcl::PointCloud<pcl::PointXYZ>& trans_cloud);
    void computeAngleDerivatives(const Eigen::Matrix<double, 6, 1>& transform, bool compute_hessian = true);
    void updateHessian(Eigen::Matrix<double, 6, 6>& hessian,
                                 const Eigen::Vector3d& x_trans,
                                 const Eigen::Matrix3d& c_inv) const;
    double updateDerivatives(Eigen::Matrix<double, 6, 1>& score_gradient,
                                Eigen::Matrix<double, 6, 6>& hessian,
                                const Eigen::Vector3d& x_trans,
                                const Eigen::Matrix3d& c_inv,
                                bool compute_hessian = true) const ;
    void computePointDerivatives(const Eigen::Vector3d& x, bool compute_hessian);
    double computeDerivatives(Eigen::Matrix<double, 6, 1>& score_gradient,
                                        Eigen::Matrix<double, 6, 6>& hessian,
                                        const pcl::PointCloud<pcl::PointXYZ>& trans_cloud,
                                        const Eigen::Matrix<double, 6, 1>& transform,
                                        bool compute_hessian); 
    double computeStepLengthMT(const Eigen::Matrix<double, 6, 1>& x,
                                        Eigen::Matrix<double, 6, 1>& step_dir,
                                        double step_init,
                                        double step_max,
                                        double step_min,
                                        double& score,
                                        Eigen::Matrix<double, 6, 1>& score_gradient,
                                        Eigen::Matrix<double, 6, 6>& hessian,
                                        pcl::PointCloud<pcl::PointXYZ>& trans_cloud);
    void align(Eigen::Matrix4d& m_matrix4d_initial_esti, pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_input_cloud);
    std::vector<pcl::PointXYZ> convertToPointVector(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);
    void startNDTThread();

    // 정규화 상수 계산 함수
    void computeNormalizationConstants();
    // step size adaptive하게
    double lineSearch(const Eigen::Matrix<double, 6, 1>& parameters,
                        const Eigen::Matrix<double, 6, 1>& delta,
                        double initial_score,
                        const Eigen::Matrix<double, 6, 1>& gradient);
    
    
private:
    // 숫자 추출 함수(맵불러올때)
    int extract_number(const std::string& i_file_path);
    // 포즈 데이터를 로드하는 함수
    std::vector<std::array<std::array<float, 4>, 4>> load_poses(const std::string& pose_file_name,
                                                                const std::array<std::array<float, 4>, 4>& Tr,
                                                                const std::array<std::array<float, 4>, 4>& Tr_inv);
    // 캘리브레이션 데이터를 로드하는 함수
    std::tuple<std::array<std::array<float, 4>, 4>, std::array<std::array<float, 4>, 4>> load_calibration(
                                                                                        const std::string& calib_file_name);

    // void setTransformationEpsilon(const float& m_cfg_f_trans_error_allow);
    // void setStepSize(const float& m_cfg_f_step_size_m); // 5.0
    // void setResolution(const float& m_cfg_f_grid_size_m); // 2.0, 1.8
    // void setMaximumIterations(const float& m_cfg_int_iterate_max); //30



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
    double trans_likelihood_;
    Eigen::Affine3d transformation_;
    int ndt_iter;
    int m_pose_num;

    pcl::PointCloud<pcl::PointXYZ>::Ptr m_filtered_map;

    uint32_t m_cfg_uint32_message_count = 0;

    std::vector<pcl::PointXYZ> inputed_target_points;
    std::vector<pcl::PointXYZ> inputed_source_points;

    Eigen::Matrix<double, 6, 6> m_cfg_mat6_d_hessian;
    Eigen::Matrix<double, 6, 1> m_cfg_mat61_d_score_grad;

    //백그라운드 ndt실행
    std::queue<sensor_msgs::PointCloud2ConstPtr> m_lidar_data_queue;
    std::mutex m_queue_mutex;
    std::condition_variable m_data_condition;
    std::thread m_ndt_thread;
    bool m_is_running;
    bool m_cfg_b_debug_mode;

    double f_l;
    double g_l;
    double f_u;
    double g_u;
    double output;


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

    Eigen::Matrix<double, 8, 4> angular_jacobian_;
    Eigen::Matrix<double, 15, 4> angular_hessian_; 
    Eigen::Matrix<double, 3, 6> point_jacobian_; 
    Eigen::Matrix<double, 18, 6> point_hessian_;

    int nr_iterations_;                 // 반복 횟수
    bool converged_;                    // 수렴 여부
    pcl::VoxelGrid<pcl::PointXYZ> target_cells_;            // 타겟 셀 (VoxelGrid 형태)
    double outlier_ratio_;              // 아웃라이어 비율
    Eigen::Affine3d final_transformation_; // 최종 변환 행렬
    Eigen::Matrix4d previous_transformation_;

    std::vector<std::array<std::array<float, 4>, 4>> m_vt_f_44_poses;



};

#endif