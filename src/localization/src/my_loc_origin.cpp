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

#include <pcl/filters/voxel_grid.h>  // VoxelGrid 사용을 위한 헤더 추가
#include <iterator>

//백그라운드 ndt 실행
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>




// 클래스 생성
LOCALIZATION::LOCALIZATION(ros::NodeHandle& nh) {

    // 맵정보 불러옴
    std::string pcd_dir = "/LHS/approach/my_code/output";
    m_pc_map_cloud = LoadPcdFile(pcd_dir);

    // Pose정보 불러옴
    std::string pose_file_name = "/LHS/approach/my_code/dataset/00/poses.txt";
    std::string calib_file_name = "/LHS/approach/my_code/dataset/00/calib.txt";

    // pose와 calibration 파일 로드
    auto [Tr, Tr_inv] = load_calibration(calib_file_name);
    m_vt_f_44_poses = load_poses(pose_file_name, Tr, Tr_inv);

    // m_vt_f_44_poses의 x,y,z -> m_vt_f_44_poses[i번째 pose][row][cul];


    // NDT 파라미터 설정
    m_cfg_d_rot_error_allow = 0.01; // 0.001
    m_cfg_d_trans_error_allow = 0.01;
    m_cfg_f_step_size_m = 1.1; // 5.0
    m_cfg_f_grid_size_m = 2.0; // 2.0, 1.8
    m_cfg_int_iterate_max = 15; //30

    m_cfg_b_debug_mode = false;

    ndt_iter = 0;
    m_pose_num = 0;


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
    m_pc_predict_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>());

    // 퍼블리셔 초기화
    m_ros_map_pub = nh.advertise<sensor_msgs::PointCloud2>("map_cloud", 1);
    m_ros_aligned_pub = nh.advertise<sensor_msgs::PointCloud2>("aligned_cloud", 1);
    m_ros_trajectory_pub = nh.advertise<sensor_msgs::PointCloud2>("trajectory_cloud", 1);
    m_ros_filtered_map_pub = nh.advertise<sensor_msgs::PointCloud2>("filtered_map_cloud", 1);
    m_ros_filtered_input_pub = nh.advertise<sensor_msgs::PointCloud2>("filtered_input_cloud", 1);
    m_ros_predict_pub = nh.advertise<sensor_msgs::PointCloud2>("predict_pose", 1);

    // 서브스크라이버 설정
    m_ros_point_cloud_sub = nh.subscribe("/kitti/velo/pointcloud", 1, &LOCALIZATION::NDTCallback, this);

    if (m_pc_map_cloud->empty()) {
        ROS_ERROR("Target cloud is empty!");
        return;
    }

    // 초기 변환 행렬
    m_matrix4d_initial_esti = Eigen::Matrix4d::Identity(); // 클래스 멤버로 선언
    m_matrix4d_prev = Eigen::Matrix4d::Identity();
    
    // 초기 source target
    setInputTarget(convertToPointVector(m_pc_map_cloud));
    pcl::PointCloud<pcl::PointXYZ>::Ptr inputed_source_points{new pcl::PointCloud<pcl::PointXYZ>()};

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

// 4x4 행렬의 역행렬을 계산하는 함수
std::array<std::array<float, 4>, 4> inverse(const std::array<std::array<float, 4>, 4>& matrix) {
    Eigen::Matrix4f mat;
    
    // std::array를 Eigen::Matrix로 변환
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            mat(i, j) = matrix[i][j];
        }
    }
    
    // 역행렬 계산
    Eigen::Matrix4f inv_mat = mat.inverse();
    
    // Eigen::Matrix를 std::array로 변환
    std::array<std::array<float, 4>, 4> inv;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            inv[i][j] = inv_mat(i, j);
        }
    }
    
    return inv;
}

// 행렬 곱셈 함수 추가 (4x4 행렬 곱셈)
std::array<std::array<float, 4>, 4> matrix_multiply(const std::array<std::array<float, 4>, 4>& A, 
                                                    const std::array<std::array<float, 4>, 4>& B) {
    std::array<std::array<float, 4>, 4> result = {0};

    #pragma omp parallel for
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            for (int k = 0; k < 4; ++k) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return result;
}

// 캘리브레이션 데이터를 로드하는 함수
std::tuple<std::array<std::array<float, 4>, 4>, std::array<std::array<float, 4>, 4>> 
LOCALIZATION::load_calibration(const std::string& calib_file_name) 
{
    std::vector<std::array<std::array<float, 4>, 4>> calibs;
    std::ifstream infile(calib_file_name);
    std::array<std::array<float, 4>, 4> Tr = {};
    std::array<std::array<float, 4>, 4> Tr_inv = {};

    std::string line;
    while (std::getline(infile, line)) {
        std::stringstream ss(line);
        std::string key;
        std::string content;

        if (std::getline(ss, key, ':') && std::getline(ss, content)) {
            std::stringstream content_ss(content);
            std::vector<float> values;
            float value;

            while (content_ss >> value) {
                values.push_back(value);
            }

            if (values.size() == 12) {
                std::array<std::array<float, 4>, 4> calib;
                calib[0] = { values[0], values[1], values[2], values[3] };
                calib[1] = { values[4], values[5], values[6], values[7] };
                calib[2] = { values[8], values[9], values[10], values[11] };
                calib[3] = { 0.0f, 0.0f, 0.0f, 1.0f };

                // Tr 캘리브레이션 매트릭스 찾기
                if (key == "Tr") {
                    Tr = calib;  // Tr 행렬 저장
                    Tr_inv = inverse(Tr);
                }

                calibs.push_back(calib);
            }
        }
    }

    return std::make_tuple(Tr, Tr_inv);
}

// 포즈 데이터를 로드하는 함수
std::vector<std::array<std::array<float, 4>, 4>> LOCALIZATION::load_poses(const std::string& pose_file_name,
                                                                        const std::array<std::array<float, 4>, 4>& Tr,
                                                                        const std::array<std::array<float, 4>, 4>& Tr_inv) 
{
    std::vector<std::array<std::array<float, 4>, 4>> poses;
    std::ifstream infile(pose_file_name);
    std::string line;
    std::vector<std::array<std::array<float, 4>, 4>> calibrated_poses;


    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        std::array<float, 12> values;
        for (int i = 0; i < 12; ++i) {
            iss >> values[i];
        }

        std::array<std::array<float, 4>, 4> pose;
        pose[0] = { values[0], values[1], values[2], values[3] };
        pose[1] = { values[4], values[5], values[6], values[7] };
        pose[2] = { values[8], values[9], values[10], values[11] };
        pose[3] = { 0.0f, 0.0f, 0.0f, 1.0f };

        poses.push_back(pose);

        
        std::array<std::array<float, 4>, 4> mat_mul_pose = matrix_multiply(Tr_inv, matrix_multiply(pose, Tr));

        // std::array<std::array<float, 4>, 4> pose_queue;
        // // 첫 번째 행 설정 (mat_mul_pose의 2번째 행)
        // for (int i = 0; i < 4; ++i) {
        //     pose_queue[0][i] = mat_mul_pose[2][i];
        //     pose_queue[1][i] = mat_mul_pose[1][i];
        //     pose_queue[2][i] = mat_mul_pose[0][i];
        // }
        // pose_queue[3] = { 0.0f, 0.0f, 0.0f, 1.0f };

        // calibrated_poses.push_back(pose_queue);
        calibrated_poses.push_back(mat_mul_pose);


    }

    

    return calibrated_poses;
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

    std::sort(std::begin(file_lists), std::end(file_lists),
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
void LOCALIZATION::setInputTarget(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
    // cloud를 std::vector<pcl::PointXYZ>로 변환하여 기존 함수 호출
    setInputTarget(convertToPointVector(cloud)); 
}

void LOCALIZATION::setInputSource(const std::vector<pcl::PointXYZ>& source_points) {
    inputed_source_points = source_points;
}
void LOCALIZATION::setInputSource(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
    setInputSource(convertToPointVector(cloud)); // 기존 함수 호출
}




void LOCALIZATION::buildTargetCells() {

    pcl::PointCloud<pcl::PointXYZ>::Ptr inputed_target_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    for (const auto& point : inputed_target_points) {
        inputed_target_cloud->push_back(point);
    }

    // NaN 포인트 필터링
    pcl::PointCloud<pcl::PointXYZ> filtered_target_points;
    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*inputed_target_cloud, filtered_target_points, indices);

    // 공간의 최소 및 최대 좌표 계산
    double d_min_x = std::numeric_limits<double>::max();
    double d_min_y = std::numeric_limits<double>::max();
    double d_min_z = std::numeric_limits<double>::max();
    double d_max_x = std::numeric_limits<double>::lowest();
    double d_max_y = std::numeric_limits<double>::lowest();
    double d_max_z = std::numeric_limits<double>::lowest();

    for (const auto& point : filtered_target_points) {
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
                    cell.stGauCell_mat3_d_cov = bTC_mat3_d_cov;

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

// static void convertTransform(const Eigen::Matrix<double, 6, 1>& x, Eigen::Affine3d& trans) {
//      trans = Eigen::Translation<Scalar, 3>(x.head<3>().cast<Scalar>()) *
//           Eigen::AngleAxis<Scalar>(static_cast<Scalar>(x(3)), Vector3::UnitX()) *
//           Eigen::AngleAxis<Scalar>(static_cast<Scalar>(x(4)), Vector3::UnitY()) *
//           Eigen::AngleAxis<Scalar>(static_cast<Scalar>(x(5)), Vector3::UnitZ());
// }


// inline double
//   auxilaryFunction_PsiMT(
//       double a, double f_a, double f_0, double g_0, double mu = 1.e-4)
//   {
//     return f_a - f_0 - mu * g_0 * a;
//   }

// inline double
//   auxilaryFunction_dPsiMT(double g_a, double g_0, double mu = 1.e-4)
//   {
//     return g_a - mu * g_0;
//   }




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
    m_filtered_map = filtered_map;

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

    // LOCALIZATION::setInputTarget(LOCALIZATION::convertToPointVector(filtered_map));
    // LOCALIZATION::setInputSource(LOCALIZATION::convertToPointVector(filtered_input_cloud));

    LOCALIZATION::setInputTarget(filtered_map);
    LOCALIZATION::setInputSource(filtered_input_cloud);


    // // NDT 시작
    // std::cout << "[DEBUG_005] NDT 정렬 시작..." << std::endl;
    

    // NDT 정렬 수행
    std::cout << "[DEBUG_011] 초기 추정치" << "(" << m_matrix4d_initial_esti(0,3) << ",      " << 
        m_matrix4d_initial_esti(1,3) << ",      "<< m_matrix4d_initial_esti(2,3) << ")" << std::endl;



    Eigen::Matrix4d m_matrix4d_prev2 = Eigen::Matrix4d::Identity(); // t-2를 저장할 행렬
    Eigen::Matrix4d m_matrix4d_prev1 = Eigen::Matrix4d::Identity(); // t-1을 저장할 행렬

    // 초기 추정치 계산 (선형 예측 모델)
    if (ndt_iter > 2) {
        // t-2, t-1, t의 행렬을 이용하여 t+1의 초기 추정치를 계산합니다.
        
        // 이전 단계에서의 변환 계산
        Eigen::Matrix4d delta_prev = m_matrix4d_prev1 * m_matrix4d_prev2.inverse();
        Eigen::Matrix4d delta_curr = m_matrix4d_initial_esti * m_matrix4d_prev1.inverse();

        // 두 변환 행렬의 평균을 사용하여 다음 변환 예측
        Eigen::Matrix4d delta_mean = (delta_prev + delta_curr) / 2.0;

        // 다음 단계의 변환 예측
        Eigen::Matrix4d m_matrix4d_predict = m_matrix4d_initial_esti * delta_mean;

        // m_matrix4d_prev2, m_matrix4d_prev1 갱신
        m_matrix4d_prev2 = m_matrix4d_prev1;
        m_matrix4d_prev1 = m_matrix4d_initial_esti;

        // 예측된 위치를 포인트 클라우드로 변환
        pcl::PointXYZ predict_position;
        predict_position.x = m_matrix4d_predict(0, 3);
        predict_position.y = m_matrix4d_predict(1, 3);
        predict_position.z = m_matrix4d_predict(2, 3);

        // 이전 포인트 클라우드 초기화하고 새로운 예측 포인트 추가
        m_pc_predict_cloud->clear();
        m_pc_predict_cloud->points.push_back(predict_position);

        // 퍼블리시
        PublishPointCloud(m_pc_predict_cloud, m_ros_predict_pub, "velo_link");

        // align 함수 호출 (예측된 변환 행렬 사용)
        //m_ndt.computeTransformation(*filtered_input_cloud, m_matrix4d_initial_esti.cast<float>());

    } else if (ndt_iter > 1) {
        // t-1과 t의 행렬을 이용하여 초기 추정치를 계산합니다.
        Eigen::Matrix4d delta_init = m_matrix4d_initial_esti * m_matrix4d_prev1.inverse();
        Eigen::Matrix4d m_matrix4d_predict = m_matrix4d_initial_esti * delta_init;

        // m_matrix4d_prev1 갱신
        m_matrix4d_prev2 = m_matrix4d_prev1;
        m_matrix4d_prev1 = m_matrix4d_initial_esti;

        // 예측된 위치를 포인트 클라우드로 변환
        pcl::PointXYZ predict_position;
        predict_position.x = m_matrix4d_predict(0, 3);
        predict_position.y = m_matrix4d_predict(1, 3);
        predict_position.z = m_matrix4d_predict(2, 3);

        // 이전 포인트 클라우드 초기화하고 새로운 예측 포인트 추가
        m_pc_predict_cloud->clear();
        m_pc_predict_cloud->points.push_back(predict_position);

        // 퍼블리시
        PublishPointCloud(m_pc_predict_cloud, m_ros_predict_pub, "velo_link");

        // align 함수 호출 (예측된 변환 행렬 사용)
        //.computeTransformation(*filtered_input_cloud, m_matrix4d_initial_esti.cast<float>());

    } else {
        // 첫 씬일 경우 Identity 행렬 사용
        m_matrix4d_prev1 = m_matrix4d_initial_esti;

        //m_ndt.computeTransformation(*filtered_input_cloud, m_matrix4d_initial_esti.cast<float>());
    }

    // ndt_iter
    ndt_iter++;


    // ---------------------------정합 수행------------------------------------------------
    //LOCALIZATION::align(m_matrix4d_initial_esti, filtered_input_cloud); // 초기 추정치 사용
    std::cout << "[DEBUG_006] NDT 정렬 끝..." << std::endl;

    // 정합된 포인트 클라우드 생성
    pcl::PointCloud<pcl::PointXYZ>::Ptr aligned_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    for (const auto& point : filtered_input_cloud->points) {
        input_cloud->points.push_back(point);
    }
    pcl::transformPointCloud(*filtered_input_cloud, *aligned_cloud, m_matrix4d_initial_esti);

    pcl::PointXYZ curr_pose_data;
    curr_pose_data.x = m_vt_f_44_poses[m_pose_num][0][3];
    curr_pose_data.y = m_vt_f_44_poses[m_pose_num][1][3];
    curr_pose_data.z = m_vt_f_44_poses[m_pose_num][2][3];

    std::cout << m_pose_num <<" 번째 실제 Pose : (" << curr_pose_data.x << ",       " <<
     curr_pose_data.y << ",           " << curr_pose_data.z << ")" << std::endl;

    // NDT 정렬 수행
    std::cout << "[DEBUG_012] aligned 추정치 : " << "(" << m_matrix4d_initial_esti(0,3) << ",      " << 
        m_matrix4d_initial_esti(1,3) << ",      "<< m_matrix4d_initial_esti(2,3) << ")" << std::endl;


    // 현재 위치를 포인트로 추출하여 누적 포인트 클라우드에 추가
    pcl::PointXYZ current_position;
    current_position.x = m_matrix4d_initial_esti(0, 3);
    current_position.y = m_matrix4d_initial_esti(1, 3);
    current_position.z = m_matrix4d_initial_esti(2, 3);
    m_pc_trajectory_cloud->points.push_back(current_position);

    // 실제Pose - aligned 추정치 Pose -> MAE 
    std::cout << " XYZ좌표 오차 평균 : " << (abs(curr_pose_data.x-current_position.x)+abs(curr_pose_data.y-current_position.y)+abs(curr_pose_data.z-current_position.z))/3 << std::endl;
    std::cout << " X좌표 오차 : " << abs(curr_pose_data.x-current_position.x) << std::endl;
    std::cout << " Y좌표 오차 : " << abs(curr_pose_data.y-current_position.y) << std::endl;
    std::cout << " Z좌표 오차 : " << abs(curr_pose_data.z-current_position.z) << std::endl;

    // 누적된 포인트 클라우드를 퍼블리시
    PublishPointCloud(m_pc_trajectory_cloud, m_ros_trajectory_pub, "velo_link");
    // 정합 결과 퍼블리시
    PublishPointCloud(aligned_cloud, m_ros_aligned_pub, "velo_link");

    // // Marker 퍼블리시
    // publishMarker(m_matrix4d_initial_esti, quat);

    m_pose_num++;

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
