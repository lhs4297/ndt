#ifndef CERES_HPP
#define CERES_HPP

#include "my_loc.hpp"  // my_loc.hpp 포함

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>

#include <Eigen/Dense>
#include <memory>


class NormalDistributionsTransform
{
public:
    NormalDistributionsTransform();

    void computeTransformation(pcl::PointCloud<pcl::PointXYZ>& output,
                                const Eigen::Matrix4d& cT_matrix4d_initial_esti);

private:
    // 헬퍼 메서드들
    double computeDerivatives(Eigen::Matrix<double, 6, 1>& score_gradient,
                                Eigen::Matrix<double, 6, 6>& hessian,
                                const pcl::PointCloud<pcl::PointXYZ>& output,
                                const Eigen::Matrix<double, 6, 1>& transform);

    double computeStepLengthMT(const Eigen::Matrix<double, 6, 1>& transform,
                                const Eigen::Matrix<double, 6, 1>& delta,
                                double delta_norm,
                                double step_size,
                                double epsilon,
                                double& score,
                                Eigen::Matrix<double, 6, 1>& score_gradient,
                                Eigen::Matrix<double, 6, 6>& hessian,
                                pcl::PointCloud<pcl::PointXYZ>& output);

    void convertTransform(const Eigen::Matrix<double, 6, 1>& delta,
                            Eigen::Matrix4d& transformation);

    void NormalDistributionsTransform::computeAngleDerivatives(const Eigen::Matrix<double, 6, 1>& transform,
                                                                 bool compute_hessian);

private:
    int nr_iterations_;
    bool converged_;
    Eigen::Matrix4d m_mat4d_final_trans; // final transformaion
    Eigen::Matrix4d m_mat4d_pre_trans; // previous transformaion
    double trans_likelihood_;
    double gauss_d1_, gauss_d2_;
    double resolution_;
    double outlier_ratio_;
    double m_d_step_size;
    pcl::VoxelGrid<pcl::PointXYZ>::Ptr target_cells_;

    // Additional internal parameters
    double m_d_trans_delta;
    double m_d_rot_delta;
    int max_iterations_;
    Eigen::Matrix4d m_mat4d_trans; // transformation

    pcl::PointCloud<pcl::PointXYZ>::ConstPtr m_cptr_input;

    Eigen::Matrix<double, 3, 6> point_jacobian_; //임시
    Eigen::Matrix<double, 18, 6> point_hessian_; //임시
    Eigen::Matrix<double, 8, 4> angular_jacobian_; // 임시
    Eigen::Matrix<double, 15, 4> angular_hessian_; //임시
};

#endif // CERES_NDT_HPP