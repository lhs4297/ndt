#include "header/my_loc.hpp"
#include <iostream>
#include <filesystem>
#include <vector>
#include <string>
#include <algorithm>

#include <Eigen/Dense>

#include <pcl/io/pcd_io.h> // LoadPcdFile
#include <pcl/filters/approximate_voxel_grid.h> // voxel_grid_filter
#include <pcl_conversions/pcl_conversions.h> // fromROSMsg
#include <pcl/filters/crop_box.h>



//백그라운드 ndt 실행
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>




// 클래스 생성
LOCALIZATION::LOCALIZATION(ros::NodeHandle& nh) {
    std::string pcd_dir = "/home/ctu/LHS/approach/my_code/map_ws/src/result";
    m_pc_map_cloud = LoadPcdFile(pcd_dir);


    // NDT 파라미터 설정
    m_cfg_f_trans_error_allow = 0.2; // 0.001
    m_cfg_f_step_size_m = 1.0; // 5.0
    m_cfg_f_grid_size_m = 1.5; // 2.0, 1.8
    m_cfg_int_iterate_max = 100; //30
    // 기타 멤버 변수
    m_cfg_f_outlier_ratio = 0.55;
    m_cfg_d_gauss_k1 = m_cfg_d_gauss_k2 = 0.0;

    m_cfg_b_debug_mode = false;


    computeNormalizationConstants();

    // NaN 값 확인 및 출력
    size_t nan_count = 0;
    for (const auto& point : m_pc_map_cloud->points) {
        if (!pcl::isFinite(point)) { // 포인트가 유한한지 확인
            nan_count++;
        }
    }

    if (nan_count > 0) {
        std::cout << "[DEBUG_000]m_pc_map_cloud contains " << nan_count << " NaN values." << std::endl;
        return;
    } else {
        std::cout << "[ERROR_000]m_pc_map_cloud does not contain any NaN values." << std::endl;
    }

    // 궤적 변수 초기화
    m_pc_trajectory_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>());

    // 퍼블리셔 초기화
    m_ros_map_pub = nh.advertise<sensor_msgs::PointCloud2>("map_cloud", 1);
    m_ros_aligned_pub = nh.advertise<sensor_msgs::PointCloud2>("aligned_cloud", 1);
    m_ros_trajectory_pub = nh.advertise<sensor_msgs::PointCloud2>("trajectory_cloud", 1);
    m_ros_filtered_map_pub = nh.advertise<sensor_msgs::PointCloud2>("filtered_map_cloud", 1);
    m_ros_filtered_input_pub = nh.advertise<sensor_msgs::PointCloud2>("filtered_input_cloud", 1);

    // 서브스크라이버 설정
    m_ros_point_cloud_sub = nh.subscribe("/kitti/velo/pointcloud", 1, &LOCALIZATION::NDTCallback, this);

    if (m_pc_map_cloud->empty()) {
        ROS_ERROR("Target cloud is empty!");
        return;
    }

    // 초기 변환 행렬
    m_matrix4d_initial_esti = Eigen::Matrix4d::Identity(); // 클래스 멤버로 선언
    
    // 초기 source target
    setInputTarget(LOCALIZATION::convertToPointVector(m_pc_map_cloud));

}

LOCALIZATION::~LOCALIZATION() {
    m_is_running = false;
    m_data_condition.notify_all();
    if (m_ndt_thread.joinable()) {
        m_ndt_thread.join();
    }
}

void LOCALIZATION::startNDTThread() {
    m_is_running = true;
    m_ndt_thread = std::thread(&LOCALIZATION::NDTProcessingThread, this);
}

// pcl::PointCloud를 std::vector로 변환하는 함수
std::vector<pcl::PointXYZ> LOCALIZATION::convertToPointVector(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {

    std::vector<pcl::PointXYZ> points;
    if (!cloud){
        std::cerr << "[ERROR] Input cloud pointer is null." << std::endl;
    }

    std::cout << "[DEBUG003] Cloud size: " << cloud->points.size() << std::endl;
    points.reserve(cloud->points.size());
    for (const auto& point : cloud->points) {
        points.push_back(point);
    }
    return points;
}

// 파일 이름에서 숫자 추출
int LOCALIZATION::extract_number(const std::string& i_file_path) {
    size_t last_slash_idx = i_file_path.find_last_of("/\\");
    size_t expender_idx = i_file_path.find_last_of(".");
    std::string file_name;
    if (last_slash_idx != std::string::npos && expender_idx != std::string::npos) {
        file_name = i_file_path.substr(last_slash_idx + 1, expender_idx - last_slash_idx - 1);
        return std::stoi(file_name);
    } else {
        return -1;
    }
}

// map 파일 가져오는 함수
pcl::PointCloud<pcl::PointXYZ>::Ptr LOCALIZATION::LoadPcdFile(const std::string& pcd_dir) {
    std::vector<std::string> file_lists;

    for (const auto& entry : std::filesystem::recursive_directory_iterator(pcd_dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".pcd") {
            file_lists.push_back(entry.path().string());
        }
    }

    std::sort(file_lists.begin(), file_lists.end(),
        [this](const std::string& a, const std::string& b) {
            return this->extract_number(a) < this->extract_number(b);
        });

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());

    for (const auto& file : file_lists) {
        pcl::PointCloud<pcl::PointXYZ> temp_cloud;
        if (pcl::io::loadPCDFile<pcl::PointXYZ>(file, temp_cloud) == -1) {
            PCL_ERROR("Couldn't read file %s \n", file.c_str());
            continue;
        }
        *cloud += temp_cloud;
    }
    

    return cloud;
}


void LOCALIZATION::setInputTarget(const std::vector<pcl::PointXYZ>& target_points) {
    inputed_target_points = target_points;
    buildTargetCells();
}

void LOCALIZATION::setInputSource(const std::vector<pcl::PointXYZ>& source_points) {
    inputed_source_points = source_points;
}




void LOCALIZATION::buildTargetCells() {
    // 공간의 최소 및 최대 좌표 계산
    double d_min_x = std::numeric_limits<double>::max();
    double d_min_y = std::numeric_limits<double>::max();
    double d_min_z = std::numeric_limits<double>::max();
    double d_max_x = std::numeric_limits<double>::lowest();
    double d_max_y = std::numeric_limits<double>::lowest();
    double d_max_z = std::numeric_limits<double>::lowest();

    for (const auto& point : inputed_target_points) {
        d_min_x = std::min(d_min_x, static_cast<double>(point.x));
        d_min_y = std::min(d_min_y, static_cast<double>(point.y));
        d_min_z = std::min(d_min_z, static_cast<double>(point.z));
        d_max_x = std::max(d_max_x, static_cast<double>(point.x));
        d_max_y = std::max(d_max_y, static_cast<double>(point.y));
        d_max_z = std::max(d_max_z, static_cast<double>(point.z));
    }

    // 격자 셀 개수 계산
    int i_num_cells_x = static_cast<int>((d_max_x - d_min_x) / m_cfg_f_grid_size_m) + 1;
    int i_num_cells_y = static_cast<int>((d_max_y - d_min_y) / m_cfg_f_grid_size_m) + 1;
    int i_num_cells_z = static_cast<int>((d_max_z - d_min_z) / m_cfg_f_grid_size_m) + 1;

    // 3D 격자 생성
    std::vector<std::vector<std::vector<std::vector<pcl::PointXYZ>>>> grid(
        i_num_cells_x, std::vector<std::vector<std::vector<pcl::PointXYZ>>>(
            i_num_cells_y, std::vector<std::vector<pcl::PointXYZ>>(i_num_cells_z)));

    // 포인트를 각 셀에 할당
    for (const auto& point : inputed_target_points) {
        int idx = static_cast<int>((static_cast<double>(point.x) - d_min_x) / m_cfg_f_grid_size_m);  // float을 int로 변경
        int idy = static_cast<int>((static_cast<double>(point.y) - d_min_y) / m_cfg_f_grid_size_m);
        int idz = static_cast<int>((static_cast<double>(point.z) - d_min_z) / m_cfg_f_grid_size_m);

        grid[idx][idy][idz].push_back(point);
    }

    // 각 셀에 대해 가우시안 모델 생성
    m_vt_stGauCell_gaussian_cells.clear();

    for (int ix = 0; ix < i_num_cells_x; ++ix) {
        for (int iy = 0; iy < i_num_cells_y; ++iy) {
            for (int iz = 0; iz < i_num_cells_z; ++iz) {
                const auto& cell_points = grid[ix][iy][iz];
                if (cell_points.size() >= 5) { // 최소 5개의 포인트 필요
                    GaussianCell cell;

                    // 평균 계산
                    Eigen::Vector3d vt3_mean = Eigen::Vector3d::Zero();
                    for (const auto& p : cell_points) {
                        vt3_mean += Eigen::Vector3d(static_cast<double>(p.x), static_cast<double>(p.y), static_cast<double>(p.z));
                    }
                    vt3_mean /= static_cast<double>(cell_points.size());
                    cell.stGauCell_vt3_d_mean = vt3_mean;

                    // 공분산 계산
                    Eigen::Matrix3d bTC_mat3_d_cov = Eigen::Matrix3d::Zero();
                    for (const auto& p : cell_points) {
                        Eigen::Vector3d bTC_vt3_diff = Eigen::Vector3d(static_cast<double>(p.x), static_cast<double>(p.y), static_cast<double>(p.z)) - vt3_mean;
                        bTC_mat3_d_cov += bTC_vt3_diff * bTC_vt3_diff.transpose();
                    }
                    bTC_mat3_d_cov /= static_cast<double>(cell_points.size() - 1);

                    //분모 0 방지
                    double epsilon = 1e-6;
                    bTC_mat3_d_cov += epsilon * Eigen::Matrix3d::Identity();
                    //1109삭제cell.stGauCell_mat3_d_cov = bTC_mat3_d_cov;
                    Eigen::Matrix3d mat3_d_inv_cov = bTC_mat3_d_cov.inverse();



                    // 공분산 행렬의 역과 행렬식 계산
                    // 공분산 행렬이 양정치 행렬인지 확인
                    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigensolver(mat3_d_inv_cov);
                    if (eigensolver.info() != Eigen::Success) continue;
                    Eigen::Vector3d eigenvalues = eigensolver.eigenvalues();
                    if (eigenvalues.minCoeff() <= 0) continue; // 양정치 아님

                    cell.stGauCell_mat3_d_inv_cov = mat3_d_inv_cov;
                    cell.stGauCell_d_det = mat3_d_inv_cov.determinant();

                    m_vt_stGauCell_gaussian_cells.push_back(cell);  // 가우시안들의 목록 : 평균,공분산 갖고있음
                }
            }
        }
    }
}


// 정규화 상수 계산 함수
void LOCALIZATION::computeNormalizationConstants() {
    double cNC_d_gauss_c1 = (1 - m_cfg_f_outlier_ratio) / (sqrt((2 * M_PI)));
    double cNC_d_gauss_c2 = m_cfg_f_outlier_ratio / (m_cfg_f_grid_size_m);
    m_cfg_d_gauss_k1 = -log(cNC_d_gauss_c1);
    m_cfg_d_gauss_k2 = -log(cNC_d_gauss_c2);
}

void LOCALIZATION::computeAngleDerivatives(const Eigen::Matrix<double, 6, 1>& transform, bool compute_hessian)
{
    // Simplified math for near 0 angles
    const auto calculate_cos_sin = [](double angle, double& c, double& s) {
        if (std::abs(angle) < 10e-5) {
        c = 1.0;
        s = 0.0;
        }
        else {
        c = std::cos(angle);
        s = std::sin(angle);
        }
    };

    double cx, cy, cz, sx, sy, sz;
    calculate_cos_sin(transform(3), cx, sx);
    calculate_cos_sin(transform(4), cy, sy);
    calculate_cos_sin(transform(5), cz, sz);

    // Precomputed angular gradient components. Letters correspond to Equation 6.19
    // [Magnusson 2009]
    angular_jacobian_.setZero();
    angular_jacobian_.row(0).noalias() = Eigen::Vector4d(
        (-sx * sz + cx * sy * cz), (-sx * cz - cx * sy * sz), (-cx * cy), 1.0); // a
    angular_jacobian_.row(1).noalias() = Eigen::Vector4d(
        (cx * sz + sx * sy * cz), (cx * cz - sx * sy * sz), (-sx * cy), 1.0); // b
    angular_jacobian_.row(2).noalias() =
        Eigen::Vector4d((-sy * cz), sy * sz, cy, 1.0); // c
    angular_jacobian_.row(3).noalias() =
        Eigen::Vector4d(sx * cy * cz, (-sx * cy * sz), sx * sy, 1.0); // d
    angular_jacobian_.row(4).noalias() =
        Eigen::Vector4d((-cx * cy * cz), cx * cy * sz, (-cx * sy), 1.0); // e
    angular_jacobian_.row(5).noalias() =
        Eigen::Vector4d((-cy * sz), (-cy * cz), 0, 1.0); // f
    angular_jacobian_.row(6).noalias() =
        Eigen::Vector4d((cx * cz - sx * sy * sz), (-cx * sz - sx * sy * cz), 0, 1.0); // g
    angular_jacobian_.row(7).noalias() =
        Eigen::Vector4d((sx * cz + cx * sy * sz), (cx * sy * cz - sx * sz), 0, 1.0); // h

    if (compute_hessian) {
        // Precomputed angular hessian components. Letters correspond to Equation 6.21 and
        // numbers correspond to row index [Magnusson 2009]
        angular_hessian_.setZero();
        angular_hessian_.row(0).noalias() = Eigen::Vector4d(
            (-cx * sz - sx * sy * cz), (-cx * cz + sx * sy * sz), sx * cy, 0.0f); // a2
        angular_hessian_.row(1).noalias() = Eigen::Vector4d(
            (-sx * sz + cx * sy * cz), (-cx * sy * sz - sx * cz), (-cx * cy), 0.0f); // a3

        angular_hessian_.row(2).noalias() =
            Eigen::Vector4d((cx * cy * cz), (-cx * cy * sz), (cx * sy), 0.0f); // b2
        angular_hessian_.row(3).noalias() =
            Eigen::Vector4d((sx * cy * cz), (-sx * cy * sz), (sx * sy), 0.0f); // b3

        // The sign of 'sx * sz' in c2 is incorrect in the thesis, and is fixed here.
        angular_hessian_.row(4).noalias() = Eigen::Vector4d(
            (-sx * cz - cx * sy * sz), (sx * sz - cx * sy * cz), 0, 0.0f); // c2
        angular_hessian_.row(5).noalias() = Eigen::Vector4d(
            (cx * cz - sx * sy * sz), (-sx * sy * cz - cx * sz), 0, 0.0f); // c3

        angular_hessian_.row(6).noalias() =
            Eigen::Vector4d((-cy * cz), (cy * sz), (-sy), 0.0f); // d1
        angular_hessian_.row(7).noalias() =
            Eigen::Vector4d((-sx * sy * cz), (sx * sy * sz), (sx * cy), 0.0f); // d2
        angular_hessian_.row(8).noalias() =
            Eigen::Vector4d((cx * sy * cz), (-cx * sy * sz), (-cx * cy), 0.0f); // d3

        angular_hessian_.row(9).noalias() =
            Eigen::Vector4d((sy * sz), (sy * cz), 0, 0.0f); // e1
        angular_hessian_.row(10).noalias() =
            Eigen::Vector4d((-sx * cy * sz), (-sx * cy * cz), 0, 0.0f); // e2
        angular_hessian_.row(11).noalias() =
            Eigen::Vector4d((cx * cy * sz), (cx * cy * cz), 0, 0.0f); // e3

        angular_hessian_.row(12).noalias() =
            Eigen::Vector4d((-cy * cz), (cy * sz), 0, 0.0f); // f1
        angular_hessian_.row(13).noalias() = Eigen::Vector4d(
            (-cx * sz - sx * sy * cz), (-cx * cz + sx * sy * sz), 0, 0.0f); // f2
        angular_hessian_.row(14).noalias() = Eigen::Vector4d(
            (-sx * sz + cx * sy * cz), (-cx * sy * sz - sx * cz), 0, 0.0f); // f3
    }
}

double LOCALIZATION::updateDerivatives(Eigen::Matrix<double, 6, 1>& score_gradient,
                                        Eigen::Matrix<double, 6, 6>& hessian,
                                        const Eigen::Vector3d& x_trans,
                                        const Eigen::Matrix3d& c_inv,
                                        bool compute_hessian) const 
{
    // 마할라노비스 거리 기반의 확률 계산 (Equation 6.9)
    double e_x_cov_x = std::exp(-m_cfg_d_gauss_k2 * x_trans.dot(c_inv * x_trans) / 2);
    double score_inc = -m_cfg_d_gauss_k1 * e_x_cov_x;

    e_x_cov_x *= m_cfg_d_gauss_k2;

    // 유효성 검사로, e_x_cov_x가 올바른 범위에 있는지 확인합니다.
    if (e_x_cov_x > 1 || e_x_cov_x < 0 || std::isnan(e_x_cov_x)) {
        return 0;
    }

    e_x_cov_x *= m_cfg_d_gauss_k1;

    for (int i = 0; i < 6; i++) {
        Eigen::Vector3d cov_dxd_pi = c_inv * point_jacobian_.col(i); 

        // 그래디언트 업데이트 (Equation 6.12)
        score_gradient(i) += x_trans.dot(cov_dxd_pi) * e_x_cov_x;

        if (compute_hessian) {
            for (int j = i; j < 6; j++) {
                hessian(i, j) += e_x_cov_x * (-m_cfg_d_gauss_k2 * x_trans.dot(cov_dxd_pi) *
                                  x_trans.dot(c_inv * point_jacobian_.col(j)) +
                              x_trans.dot(c_inv * point_hessian_.block<3, 1>(3 * i, j)) +
                              point_jacobian_.col(j).dot(cov_dxd_pi));

                if (i != j) {
                    hessian(j, i) = hessian(i, j); 
                }
            }
        }
    }

    return score_inc;
}

void LOCALIZATION::computePointDerivatives(const Eigen::Vector3d& x, Eigen::Matrix4d& transform_matrix, bool compute_hessian)
{
    // Calculate first derivative of Transformation Equation 6.17 w.r.t. transform vector.
    // Derivative w.r.t. ith element of transform vector corresponds to column i,
    // Equation 6.18 and 6.19 [Magnusson 2009]

    // 평행 이동 요소를 `point_jacobian_`에 추가
    point_jacobian_.col(3).head<3>() = transform_matrix.block<3, 1>(0, 3);

    Eigen::Matrix<double, 8, 1> point_angular_jacobian =
        angular_jacobian_ * Eigen::Vector4d(x[0], x[1], x[2], 0.0);
    point_jacobian_.setZero();
    point_jacobian_.block<3, 3>(0, 0) = transform_matrix.block<3, 3>(0, 0);

    point_jacobian_(1, 3) = point_angular_jacobian[0];
    point_jacobian_(2, 3) = point_angular_jacobian[1];
    point_jacobian_(0, 4) = point_angular_jacobian[2];
    point_jacobian_(1, 4) = point_angular_jacobian[3];
    point_jacobian_(2, 4) = point_angular_jacobian[4];
    point_jacobian_(0, 5) = point_angular_jacobian[5];
    point_jacobian_(1, 5) = point_angular_jacobian[6];
    point_jacobian_(2, 5) = point_angular_jacobian[7];

    if (compute_hessian) {
        Eigen::Matrix<double, 15, 1> point_angular_hessian =
            angular_hessian_ * Eigen::Vector4d(x[0], x[1], x[2], 0.0);

        // Vectors from Equation 6.21 [Magnusson 2009]
        const Eigen::Vector3d a(0, point_angular_hessian[0], point_angular_hessian[1]);
        const Eigen::Vector3d b(0, point_angular_hessian[2], point_angular_hessian[3]);
        const Eigen::Vector3d c(0, point_angular_hessian[4], point_angular_hessian[5]);
        const Eigen::Vector3d d = point_angular_hessian.block<3, 1>(6, 0);
        const Eigen::Vector3d e = point_angular_hessian.block<3, 1>(9, 0);
        const Eigen::Vector3d f = point_angular_hessian.block<3, 1>(12, 0);

        // Calculate second derivative of Transformation Equation 6.17 w.r.t. transform
        // vector. Derivative w.r.t. ith and jth elements of transform vector corresponds to
        // the 3x1 block matrix starting at (3i,j), Equation 6.20 and 6.21 [Magnusson 2009]
        point_hessian_.setZero();
        point_hessian_.block<3, 1>(9, 3) = a;
        point_hessian_.block<3, 1>(12, 3) = b;
        point_hessian_.block<3, 1>(15, 3) = c;
        point_hessian_.block<3, 1>(9, 4) = b;
        point_hessian_.block<3, 1>(12, 4) = d;
        point_hessian_.block<3, 1>(15, 4) = e;
        point_hessian_.block<3, 1>(9, 5) = c;
        point_hessian_.block<3, 1>(12, 5) = e;
        point_hessian_.block<3, 1>(15, 5) = f;
    }
}



// 정합 스코어 및 그래디언트 계산 함수
double LOCALIZATION::computeScoreAndGradient(Eigen::Matrix<double, 6, 1>& score_gradient,
                                            Eigen::Matrix<double, 6, 6>& hessian,
                                            const pcl::PointCloud<pcl::PointXYZ>& trans_cloud,
                                            const Eigen::Matrix<double, 6, 1>& transform,
                                            bool compute_hessian) 
{
    // score_gradient.setZero();
    // hessian.setZero();
    double score = 0.0;

    // 변환 행렬을 4x4 행렬로 변환
    Eigen::Matrix4d transform_matrix = computeTransformationMatrix(transform);

    // 각 포인트에 대해 파라미터의 각도에 대한 파생 변수 계산
    computeAngleDerivatives(transform);

    // 각 포인트에 대해 업데이트
    for (std::size_t idx = 0; idx < inputed_source_points.size(); idx++) {
        const auto& point = inputed_source_points[idx];
        Eigen::Vector4d p(static_cast<double>(point.x), static_cast<double>(point.y), static_cast<double>(point.z), 1.0);

        // 변환 행렬과 곱셈
        Eigen::Vector4d p_transformed = transform_matrix * p;
        Eigen::Vector3d x_trans(p_transformed[0], p_transformed[1], p_transformed[2]);

        // 근처의 셀(노이즈를 포함한 셀) 검색
        double min_dist = std::numeric_limits<double>::max();
        const GaussianCell* closest_cell = nullptr;

        for (const auto& cell : m_vt_stGauCell_gaussian_cells) {
            Eigen::Vector3d diff = x_trans - cell.stGauCell_vt3_d_mean;
            double dist = diff.squaredNorm();
            if (dist < min_dist) {
                min_dist = dist;
                closest_cell = &cell;
            }
        }

        if (closest_cell) {
            Eigen::Vector3d x_trans_mean = x_trans - closest_cell->stGauCell_vt3_d_mean;
            Eigen::Matrix3d c_inv = closest_cell->stGauCell_mat3_d_inv_cov;

            // 파생 변수 계산
            computePointDerivatives(Eigen::Vector3d(p[0], p[1], p[2]), transform_matrix, true);

            // 스코어, 그래디언트, 헤시안 업데이트
            score += updateDerivatives(score_gradient, hessian, x_trans_mean, c_inv, compute_hessian);
        }
    }

    return score;
}



Eigen::Matrix4d LOCALIZATION::computeTransformationMatrix(const Eigen::VectorXd& parameters) {
    // 매개변수를 이용하여 변환 행렬 생성
    //[DEBUG013] Parmeters : -2.11008, -0.254241, 6.8303, -18.6958, -6.26707, 5.93876,
    double x = parameters[0];
    double y = parameters[1];
    double z = parameters[2];

    double deg_to_rad = M_PI / 180.0;
    double roll = parameters[3] * deg_to_rad;
    double pitch = parameters[4] * deg_to_rad;
    double yaw = parameters[5] * deg_to_rad;

    Eigen::Matrix3d rotation;
    rotation = Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX()) *
               Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
               Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ());

    Eigen::Matrix4d transformation = Eigen::Matrix4d::Identity();
    transformation.block<3, 3>(0, 0) = rotation;  // 4x4행렬에서 상위 3x3행렬은 회전
    transformation(0, 3) = x;  // 나머지 선형변환 행렬
    transformation(1, 3) = y;
    transformation(2, 3) = z;

    return transformation;  // /home/ctu/LHS/approach/my_code/map_ws/my_ndt_4x4matrix.png
}



Eigen::VectorXd LOCALIZATION::computeParameters(const Eigen::Matrix4d& transformation) {
    // 변환 행렬에서 매개변수 추출
    Eigen::Matrix<double, 6, 1> parameters;

    parameters[0] = transformation(0, 3); // x
    parameters[1] = transformation(1, 3); // y
    parameters[2] = transformation(2, 3); // z

    // 회전 행렬에서 roll, pitch, yaw 추출
    Eigen::Matrix3d rotation = transformation.block<3, 3>(0, 0);
    Eigen::Vector3d euler_angles = rotation.eulerAngles(0, 1, 2);

    parameters[3] = euler_angles[0]; // roll
    parameters[4] = euler_angles[1]; // pitch
    parameters[5] = euler_angles[2]; // yaw

    return parameters;  // x, y, z, roll, pitch, yaw
}


// 정합함수
void LOCALIZATION::align(Eigen::Matrix4d& m_matrix4d_initial_esti) {
    // 초기 파라미터를 추출합니다.
    Eigen::Matrix<double, 6, 1> parameters = computeParameters(m_matrix4d_initial_esti);

    // 매개변수 크기 확인
    if (parameters.size() != 6) {
        std::cerr << "[ERROR] parameters size is not 6." << std::endl;
        return;
    }


    // inputed_source_points를 pcl::PointCloud로 변환
    pcl::PointCloud<pcl::PointXYZ> trans_cloud;
    trans_cloud.points.assign(inputed_source_points.begin(), inputed_source_points.end());

    for (int iter = 0; iter < m_cfg_int_iterate_max; ++iter) {
        // 그래디언트와 헤시안을 초기화합니다.
        Eigen::Matrix<double, 6, 1> score_gradient = Eigen::Matrix<double, 6, 1>::Zero();
        Eigen::Matrix<double, 6, 6> hessian = Eigen::Matrix<double, 6, 6>::Zero();

        // 스코어를 계산하고, 그래디언트와 헤시안을 업데이트합니다.
        double score = computeScoreAndGradient(score_gradient, hessian, trans_cloud, parameters);

        if (m_cfg_b_debug_mode==false) {
            std::cout << "[DEBUG] Score at iteration " << iter << ": " << score << std::endl;
            std::cout << "[DEBUG033] Parameters: " 
              << parameters[0] << "," << parameters[1] << "," << parameters[2] << std::endl;
              //","              << parameters[3] << "," << parameters[4] << "," << parameters[5] << std::endl;
        }

        // 그래디언트의 노름을 계산하여 확인합니다.
        double gradient_norm = score_gradient.norm();
        if (std::isnan(gradient_norm) || std::isinf(gradient_norm)) {
            std::cerr << "[ERROR] Gradient norm is invalid at iteration " << iter << std::endl;
            break;
        }

        // 그래디언트 클리핑: 큰 값이 나올 경우 조정합니다.
        const double gradient_max_norm = 1000.0;
        if (gradient_norm > gradient_max_norm) {
            score_gradient *= (gradient_max_norm / gradient_norm);
            if (m_cfg_b_debug_mode) {
                std::cout << "[DEBUG] Gradient clipped at iteration " << iter << ":" << gradient_norm << std::endl;
            }
        }

        // 헤시안이 양의 정부호인지 확인하고, 선형 시스템을 해결합니다.
        Eigen::Matrix<double, 6, 6> hessian_norm = hessian;
        Eigen::Matrix<double, 6, 1> delta = hessian_norm.ldlt().solve(-score_gradient);

        // 파라미터 업데이트: 스텝 사이즈 조정 (라인 검색 또는 고정 스텝)
        double step_size = 0.5 * (1.0 / (1.0 + iter));//m_cfg_f_step_size_m;
        parameters += step_size * delta;

        if (m_cfg_b_debug_mode) {
            std::cout << "[DEBUG] Updated parameters at iteration " << iter << ": " << parameters.transpose() << std::endl;
        }

        // 변환 행렬 업데이트
        m_matrix4d_initial_esti = computeTransformationMatrix(parameters);


        if (abs(score) < 0.001 ) {
            std::cout << "Converged at iteration: " << iter << std::endl;
            break;
        }
    }

}




// NDT처리를 위한 스레드 함수
void LOCALIZATION::NDTProcessingThread() {
    while (m_is_running) {
        sensor_msgs::PointCloud2ConstPtr msg;

        // 큐에서 데이터 가져오기
        {
            std::unique_lock<std::mutex> lock(m_queue_mutex);
            m_data_condition.wait(lock, [this] {
                return !m_lidar_data_queue.empty() || !m_is_running;
            });

            if (!m_is_running && m_lidar_data_queue.empty()) {
                break;
            }

            msg = m_lidar_data_queue.front();
            m_lidar_data_queue.pop();
        }

        // NDT 처리 함수 호출
        std::cout << "[DEBUG_001] NDT 호출..." << std::endl;
        ProcessNDT(msg);
    }
}


// NDT 콜백 함수
void LOCALIZATION::NDTCallback(const sensor_msgs::PointCloud2ConstPtr& msg) {

    
    m_cfg_uint32_message_count++;
    std::cout << m_cfg_uint32_message_count << "번째 입력데이터 들어옴." << std::endl;

    std::lock_guard<std::mutex> lock(m_queue_mutex);
    // 큐의 크기를 제한하여 오래된 데이터를 버림
    if (m_lidar_data_queue.size() < 100) {
        m_lidar_data_queue.push(msg);
        std::cout << "queue size : " << m_lidar_data_queue.size() << std::endl;
        m_data_condition.notify_one();
    } else {
        // 큐가 가득 찬 경우 가장 오래된 데이터를 버리고 새로운 데이터를 추가
        m_lidar_data_queue.pop();
        m_lidar_data_queue.push(msg);
    }
}


void LOCALIZATION::ProcessNDT(const sensor_msgs::PointCloud2ConstPtr& msg)
{

    // PCL로 변환
    pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::fromROSMsg(*msg, *input_cloud);


    // target Map 크기 제한
    float range = 25.0; // 필터링할 반경 (단위: 미터)
    Eigen::Vector4f tar_min_point, tar_max_point;
    tar_min_point[0] = m_matrix4d_initial_esti(0, 3) - range;
    tar_min_point[1] = m_matrix4d_initial_esti(1, 3) - range;
    tar_min_point[2] = m_matrix4d_initial_esti(2, 3) - range;
    tar_min_point[3] = 1.0;

    tar_max_point[0] = m_matrix4d_initial_esti(0, 3) + range;
    tar_max_point[1] = m_matrix4d_initial_esti(1, 3) + range;
    tar_max_point[2] = m_matrix4d_initial_esti(2, 3) + range;
    tar_max_point[3] = 1.0;

    // input source 크기 제한
    Eigen::Vector4f src_min_point, src_max_point;
    src_min_point[0] = - range;
    src_min_point[1] = - range;
    src_min_point[2] = - 1.6;
    src_min_point[3] = 1.0;

    src_max_point[0] = + range;
    src_max_point[1] = + range;
    src_max_point[2] = + range + 10;
    src_max_point[3] = 1.0;

    if (m_pc_map_cloud->empty()) {
        ROS_ERROR("[ERROR_002] m_pc_map_cloud is empty.");
        return;
    }

    // 맵 포인트 클라우드 필터링
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_map(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::CropBox<pcl::PointXYZ> crop_box_filter;
    crop_box_filter.setInputCloud(m_pc_map_cloud);
    crop_box_filter.setMin(tar_min_point.cast<float>());
    crop_box_filter.setMax(tar_max_point.cast<float>());
    crop_box_filter.filter(*filtered_map);
    PublishPointCloud(filtered_map, m_ros_filtered_map_pub, "velo_link");

    // Input Scan data 필터링 및 크기 제한
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_input_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::ApproximateVoxelGrid<pcl::PointXYZ> voxel_grid_filter;
    voxel_grid_filter.setLeafSize(1.5, 1.5, 1.5);  // 다운샘플링 1.0
    voxel_grid_filter.setInputCloud(input_cloud);
    voxel_grid_filter.filter(*filtered_input_cloud);

    crop_box_filter.setInputCloud(filtered_input_cloud);  // VoxelGrid 적용된 스캔 데이터에 대해 크기 제한
    crop_box_filter.setMin(src_min_point);
    crop_box_filter.setMax(src_max_point);
    crop_box_filter.filter(*filtered_input_cloud);

    // 누적된 포인트 클라우드를 퍼블리시
    PublishPointCloud(filtered_input_cloud, m_ros_filtered_input_pub, "velo_link");

    
    std::cout << "[DEBUG002] filtered_map size: " << filtered_map->points.size() << std::endl;

    LOCALIZATION::setInputTarget(LOCALIZATION::convertToPointVector(filtered_map));
    LOCALIZATION::setInputSource(LOCALIZATION::convertToPointVector(filtered_input_cloud));


    // NDT 시작
    std::cout << "[DEBUG_005] NDT 정렬 시작..." << std::endl;
    

    // NDT 정렬 수행
    std::cout << "[DEBUG_011] 초기 추정치" << "(" << m_matrix4d_initial_esti(0,3) << ",      " << 
        m_matrix4d_initial_esti(1,3) << ",      "<< m_matrix4d_initial_esti(2,3) << ")" << std::endl;

    // 정합 수행
    LOCALIZATION::align(m_matrix4d_initial_esti); // 초기 추정치 사용
    std::cout << "[DEBUG_006] NDT 정렬 끝..." << std::endl;

    // 정합된 포인트 클라우드 생성
    pcl::PointCloud<pcl::PointXYZ>::Ptr aligned_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::transformPointCloud(*filtered_input_cloud, *aligned_cloud, m_matrix4d_initial_esti);

    // NDT 정렬 수행
    std::cout << "[DEBUG_012] aligned 추정치" << "(" << m_matrix4d_initial_esti(0,3) << ",      " << 
        m_matrix4d_initial_esti(1,3) << ",      "<< m_matrix4d_initial_esti(2,3) << ")" << std::endl;


    // 현재 위치를 포인트로 추출하여 누적 포인트 클라우드에 추가
    pcl::PointXYZ current_position;
    current_position.x = m_matrix4d_initial_esti(0, 3);
    current_position.y = m_matrix4d_initial_esti(1, 3);
    current_position.z = m_matrix4d_initial_esti(2, 3);
    m_pc_trajectory_cloud->points.push_back(current_position);

    // 누적된 포인트 클라우드를 퍼블리시
    PublishPointCloud(m_pc_trajectory_cloud, m_ros_trajectory_pub, "velo_link");
    // 정합 결과 퍼블리시
    PublishPointCloud(aligned_cloud, m_ros_aligned_pub, "velo_link");

    // // Marker 퍼블리시
    // publishMarker(m_matrix4d_initial_esti, quat);

}

// 맵 퍼블리시 함수
void LOCALIZATION::PublishPointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, ros::Publisher& pub, const std::string& frame_id) {
    if (cloud->empty()) {
        ROS_WARN("Point cloud is empty!");
        return;
    }

    sensor_msgs::PointCloud2 output;
    pcl::toROSMsg(*cloud, output);
    output.header.frame_id = frame_id;
    output.header.stamp = ros::Time::now();

    pub.publish(output);
}

int main(int argc, char** argv) {
    std::cout << "[DEBUG_008] node start.." << std::endl;

    ros::init(argc, argv, "my_localization_node");
    ros::NodeHandle nh;

    LOCALIZATION loc(nh);

    loc.startNDTThread();

    ros::Rate loop_rate(1);
    while (ros::ok()) {
        loc.PublishPointCloud(loc.m_pc_map_cloud, loc.m_ros_map_pub, "velo_link");
        ros::spinOnce();
        loop_rate.sleep();
    }

    return 0;
}
