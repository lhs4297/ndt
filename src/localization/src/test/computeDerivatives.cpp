#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/transforms.h>

// Gaussian Cell 구조체
struct GaussianCell {
    Eigen::Vector3d stGauCell_vt3_d_mean;
    Eigen::Matrix3d stGauCell_mat3_d_inv_cov;
};

// Mock 클래스 및 데이터 구조
class LOCALIZATION {
public:
    // 필요한 데이터 구조
    std::vector<pcl::PointXYZ> m_vt_pcl_inputed_source_points;
    std::vector<struct GaussianCell> m_vt_stGauCell_gaussian_cells;
    Eigen::Matrix<double, 3, 6> point_jacobian_;
    Eigen::Matrix<double, 18, 6> point_hessian_;

    // 함수 선언
    double computeDerivatives(Eigen::Matrix<double, 6, 1>& score_gradient,
                              Eigen::Matrix<double, 6, 6>& hessian,
                              const pcl::PointCloud<pcl::PointXYZ>& trans_cloud,
                              const Eigen::Matrix<double, 6, 1>& transform,
                              bool compute_hessian);

    double updateDerivatives(Eigen::Matrix<double, 6, 1>& score_gradient,
                             Eigen::Matrix<double, 6, 6>& hessian,
                             const Eigen::Vector3d& x_trans,
                             const Eigen::Matrix3d& c_inv,
                             bool compute_hessian) const {
                // e^(-d_2/2 * (x_k - mu_k)^T Sigma_k^-1 (x_k - mu_k)) Equation 6.9 [Magnusson 2009]

        double outlier_ratio_ = 0.55;
        double resolution_ = 2.0;
        const double gauss_c1 = 10 * (1 - outlier_ratio_);
        const double gauss_c2 = outlier_ratio_ / pow(resolution_, 3);
        const double gauss_d3 = -std::log(gauss_c2);
        double gauss_d1_ = -std::log(gauss_c1 + gauss_c2) - gauss_d3;
        double gauss_d2_ =
            -2 * std::log((-std::log(gauss_c1 * std::exp(-0.5) + gauss_c2) - gauss_d3) /
                            gauss_d1_);

        //std::cerr << "gauss_d1_: " << gauss_d1_ << std::endl;

        double e_x_cov_x = std::exp(-gauss_d2_ * x_trans.dot(c_inv * x_trans) / 2);
        // Calculate likelihood of transformed points existence, Equation 6.9 [Magnusson
        // 2009]
        if (std::isnan(e_x_cov_x)) {
            std::cerr << "e_x_cov_x is NaN! x_trans: " << x_trans.transpose() << ", c_inv: " << c_inv << std::endl;
            return 0;
        }
        const double score_inc = -gauss_d1_ * e_x_cov_x;

        e_x_cov_x = gauss_d2_ * e_x_cov_x;

        // Error checking for invalid values.
        if (e_x_cov_x > 1 || e_x_cov_x < 0 || std::isnan(e_x_cov_x)) {
            return 0;
        }

        // Reusable portion of Equation 6.12 and 6.13 [Magnusson 2009]
        e_x_cov_x *= -gauss_d1_;
        //std::cout << "e_x_cov_x : " << e_x_cov_x << std::endl;

        if (hessian.diagonal().minCoeff() < 1e-9) {
            hessian += Eigen::Matrix<double, 6, 6>::Identity() * 1e-6;
        }

        // 예제: x, y, z 편미분 값으로 채우기
        // 임시 point_jacobian_ 초기화 (3x6)
        Eigen::Matrix<double, 3, 6> point_jacobian_;
        point_jacobian_.setZero(); // 모든 요소를 0으로 초기화

        const Eigen::Vector3d x = x_trans;
        point_jacobian_.col(0) = Eigen::Vector3d(1.0, 0.0, 0.0); // dx/dtx
        point_jacobian_.col(1) = Eigen::Vector3d(0.0, 1.0, 0.0); // dy/dty
        point_jacobian_.col(2) = Eigen::Vector3d(0.0, 0.0, 1.0); // dz/dtz
        point_jacobian_.col(3) = Eigen::Vector3d(1.0, -x[2], x[1]); // 회전 x
        point_jacobian_.col(4) = Eigen::Vector3d(x[2], 1.0, -x[0]); // 회전 y
        point_jacobian_.col(5) = Eigen::Vector3d(-x[1], x[0], 1.0); // 회전 z

        for (int i = 0; i < 6; i++) {
            // Sigma_k^-1 d(T(x,p))/dpi, Reusable portion of Equation 6.12 and 6.13 [Magnusson
            // 2009]
            const Eigen::Vector3d cov_dxd_pi = c_inv * point_jacobian_.col(i)*5;
            //std::cout << "cov_dxd_pi : " << cov_dxd_pi << std::endl;

            // if (e_x_cov_x < 0) {
            //     std::cerr << "e_x_cov_x is negative: " << e_x_cov_x << std::endl;
            //     std::cerr << "x_trans: " << x_trans.transpose() << ", c_inv: " << c_inv << std::endl;
            // }

            // if (i > 2) { // 회전 부분 디버깅
            //     std::cout << "i: " << i << std::endl;
            //     std::cout << "x_trans: " << x_trans.transpose() << std::endl;
            //     std::cout << "cov_dxd_pi: " << cov_dxd_pi.transpose() << std::endl;
            //     std::cout << "point_jacobian_.col(i): " << point_jacobian_.col(i).transpose() << std::endl;
            //     std::cout << "score_gradient(i): " << score_gradient(i) << std::endl;
            // }

            if ( i > 2){
                std::cout << "-----------------------------------------"<<std::endl;
                std::cout << "i : " << i << std::endl;
                std::cout << "c_inv : " << c_inv << std::endl;
                std::cout << "x_trans : " << x_trans << std::endl;
                std::cout << "cov_dxd_pi : " << cov_dxd_pi << std::endl;
                std::cout << "e_x_cov_x : " << e_x_cov_x << std::endl;
                std::cout << "point_jacobian_.col(i) : " << point_jacobian_.col(i) << std::endl;
            }



            // Update gradient, Equation 6.12 [Magnusson 2009]
            score_gradient(i) += x_trans.dot(cov_dxd_pi) * e_x_cov_x;
            std::cout << "score_gradient : " << score_gradient << std::endl;
            
            if (compute_hessian) {
                for (Eigen::Index j = 0; j < hessian.cols(); j++) {
                    // Update hessian, Equation 6.13 [Magnusson 2009]
                    hessian(i, j) +=
                        e_x_cov_x * (-gauss_d2_ * x_trans.dot(cov_dxd_pi) *
                                        x_trans.dot(c_inv * point_jacobian_.col(j)) +
                                    x_trans.dot(c_inv * point_hessian_.block<3, 1>(3 * i, j)) +
                                    point_jacobian_.col(j).dot(cov_dxd_pi));
                }
            }
        }

        return score_inc;
    }
};

double LOCALIZATION::computeDerivatives(Eigen::Matrix<double, 6, 1>& score_gradient,
                                        Eigen::Matrix<double, 6, 6>& hessian,
                                        const pcl::PointCloud<pcl::PointXYZ>& trans_cloud,
                                        const Eigen::Matrix<double, 6, 1>& transform,
                                        bool compute_hessian) {
    score_gradient.setZero();
    hessian.setZero();
    double score = 0.0;

    // Update gradient and hessian for each point
    for (std::size_t idx = 0; idx < m_vt_pcl_inputed_source_points.size(); idx++) {
        const auto& x_trans_pt = trans_cloud[idx];

        // 현재 예제에서는 이웃 찾기를 생략하고 동일 인덱스를 사용
        const auto& cell = m_vt_stGauCell_gaussian_cells[idx];
        const Eigen::Vector3d x_trans = x_trans_pt.getVector3fMap().cast<double>() - cell.stGauCell_vt3_d_mean;
        const Eigen::Matrix3d c_inv = cell.stGauCell_mat3_d_inv_cov;

        std::cout << "cov : " << cell.stGauCell_mat3_d_inv_cov << std::endl;

        // Update score, gradient and hessian
        score += updateDerivatives(score_gradient, hessian, x_trans, c_inv, compute_hessian);
    }

    return score;
}

int main() {
    LOCALIZATION loc;

    // 입력 포인트 클라우드 생성
    pcl::PointCloud<pcl::PointXYZ> trans_cloud;
    loc.m_vt_pcl_inputed_source_points = {
        pcl::PointXYZ(1.0, 2.0, 3.0),
        pcl::PointXYZ(4.0, 5.0, 6.0),
        pcl::PointXYZ(7.0, 8.0, 9.0)
    };
    for (const auto& point : loc.m_vt_pcl_inputed_source_points) {
        trans_cloud.points.push_back(point);
    }
    trans_cloud.width = trans_cloud.points.size();
    trans_cloud.height = 1;
    trans_cloud.is_dense = true;

    // 타겟 포인트 (가우시안 셀) 생성
    loc.m_vt_stGauCell_gaussian_cells = {
        {Eigen::Vector3d(11.1, 2.1, 8.1), (1.0 / 2.0) * Eigen::Matrix3d::Identity()},
        {Eigen::Vector3d(4.1, 5.1, 1.1), (1.0 / 3.0) * Eigen::Matrix3d::Identity()},
        {Eigen::Vector3d(7.1, 18.1, 9.1), (1.0 / 4.0) * Eigen::Matrix3d::Identity()}
    };

    // 변환 벡터
    Eigen::Matrix<double, 6, 1> transform;
    transform.setZero();

    // 구배 및 헤시안 초기화
    Eigen::Matrix<double, 6, 1> score_gradient;
    Eigen::Matrix<double, 6, 6> hessian;

    // 임시 point_hessian_ 초기화 (18x6)
    Eigen::Matrix<double, 18, 6> point_hessian_;
    point_hessian_.setZero(); // 모든 요소를 0으로 초기화

    int max_iterations = 10;
    double score;

    for (int i = 0; i < max_iterations; ++i) {
        // 변환 클라우드 업데이트
        pcl::PointCloud<pcl::PointXYZ> updated_cloud;
        Eigen::Affine3d transform_matrix = Eigen::Translation3d(transform.head<3>()) *
                                           Eigen::AngleAxisd(transform[3], Eigen::Vector3d::UnitX()) *
                                           Eigen::AngleAxisd(transform[4], Eigen::Vector3d::UnitY()) *
                                           Eigen::AngleAxisd(transform[5], Eigen::Vector3d::UnitZ());
        pcl::transformPointCloud(trans_cloud, updated_cloud, transform_matrix);

        // computeDerivatives 호출
        score = loc.computeDerivatives(score_gradient, hessian, updated_cloud, transform, true);
        std::cout << "hessian : " << hessian << std::endl;

        // Delta 계산
        if (hessian.diagonal().minCoeff() == 0) {
            hessian += Eigen::Matrix<double, 6, 6>::Identity() * 1e-6; // 작은 값을 추가
        }
        Eigen::Matrix<double, 6, 1> delta = hessian.ldlt().solve(score_gradient);
        if (!delta.array().isFinite().all()) {
            std::cerr << "Delta contains NaN! Skipping update." << std::endl;
            continue;
        }
        transform -= delta;

        // Iteration 결과 출력
        std::cout << "Iteration " << i << ": transform = " << transform << std::endl;
        std::cout << "Iteration " << i << ": Score = " << score << ", Delta norm = " << delta.norm() << std::endl;

        // 수렴 조건 확인
        if (delta.norm() < 1e-6) {
            break;
        }
    }

    return 0;
}
