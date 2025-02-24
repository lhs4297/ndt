#include "header/my_loc.hpp"
#include <filesystem>  // LoadPcdFile

#include <Eigen/Dense> // eigensolver

#include <pcl/io/pcd_io.h> // LoadPcdFile
#include <pcl_conversions/pcl_conversions.h> // fromROSMsg
#include <pcl/filters/filter.h> // NaN 제거
#include <pcl/filters/extract_indices.h> // 필터링
#include <pcl/common/common.h> // PointCloud 처리
#include <pcl/filters/voxel_grid.h>  // VoxelGrid


// 클래스 생성
LOCALIZATION::LOCALIZATION(ros::NodeHandle& nh) {

    // 맵정보 불러옴
    std::string pcd_dir = "/LHS/approach/my_code/output/input_map";
    m_pcl_map_cloud = LoadPcdFile(pcd_dir);

    m_b_debugmode = false;

    m_f_radius_m = 20.0; // 필터링할 반경 (단위: 미터)
    m_cfg_f_grid_size_m = 2.0;

    m_matrix4d_initial_esti = Eigen::Matrix4d::Identity();
    
    // 퍼플리쉬 설정
    m_ros_map_pub = nh.advertise<sensor_msgs::PointCloud2>("map_cloud", 1);
    m_ros_filtered_map_pub = nh.advertise<sensor_msgs::PointCloud2>("filtered_map_cloud", 1);
    m_ros_filtered_input_pub = nh.advertise<sensor_msgs::PointCloud2>("filtered_input_cloud", 1);

    // 서브스크라이버 설정
    m_ros_point_cloud_sub = nh.subscribe("/kitti/velo/pointcloud", 1, &LOCALIZATION::NDTCallback, this);

    // 초기 타겟 포인트
    setInputTarget(convertToPointVector(m_pcl_map_cloud));
}

LOCALIZATION::~LOCALIZATION() {
    m_b_is_running = false;
    m_std_data_condition.notify_all();
    if (m_std_thread.joinable()) {
        m_std_thread.join();
    }
}

void LOCALIZATION::startNDTThread() {
    m_b_is_running = true;
    m_std_thread = std::thread(&LOCALIZATION::NDTProcessingThread, this);
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
    m_pcl_inputed_target_points = target_points;
    buildTargetCells();
}
void LOCALIZATION::setInputTarget(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
    // cloud를 std::vector<pcl::PointXYZ>로 변환하여 기존 함수 호출
    setInputTarget(convertToPointVector(cloud)); 
}

void LOCALIZATION::setInputSource(const std::vector<pcl::PointXYZ>& source_points) {
    m_pcl_inputed_source_points = source_points;
}
void LOCALIZATION::setInputSource(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
    setInputSource(convertToPointVector(cloud)); // 기존 함수 호출
}

void LOCALIZATION::buildTargetCells() {

    pcl::PointCloud<pcl::PointXYZ>::Ptr inputed_target_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    for (const auto& point : m_pcl_inputed_target_points) {
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
    for (const auto& point : m_pcl_inputed_target_points) {
        int idx = static_cast<int>((static_cast<double>(point.x) - d_min_x) / m_cfg_f_grid_size_m);  // float을 int로 변경
        int idy = static_cast<int>((static_cast<double>(point.y) - d_min_y) / m_cfg_f_grid_size_m);
        int idz = static_cast<int>((static_cast<double>(point.z) - d_min_z) / m_cfg_f_grid_size_m);

        grid[idx][idy][idz].push_back(point);
    }

    // 각 셀에 대해 가우시안 모델 생성
    m_vec_stGauCell_gaussian_cells.clear();

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
                    cell.stGauCell_vec3d_mean = vt3_mean;

                    // 공분산 계산
                    Eigen::Matrix3d bTC_mat3_d_cov = Eigen::Matrix3d::Zero();
                    for (const auto& p : cell_points) {
                        Eigen::Vector3d bTC_vt3_diff = Eigen::Vector3d(static_cast<double>(p.x), static_cast<double>(p.y), static_cast<double>(p.z)) - vt3_mean;
                        bTC_mat3_d_cov += bTC_vt3_diff * bTC_vt3_diff.transpose();
                    }
                    bTC_mat3_d_cov /= static_cast<double>(cell_points.size() - 1);
                    cell.stGauCell_mat3d_cov = bTC_mat3_d_cov;

                    //분모 0 방지
                    double epsilon = 1e-6;
                    bTC_mat3_d_cov += epsilon * Eigen::Matrix3d::Identity();
                    //1109삭제cell.stGauCell_mat3d_cov = bTC_mat3_d_cov;
                    Eigen::Matrix3d mat3d_inv_cov = bTC_mat3_d_cov.inverse();



                    // 공분산 행렬의 역과 행렬식 계산
                    // 공분산 행렬이 양정치 행렬인지 확인
                    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigensolver(mat3d_inv_cov);
                    if (eigensolver.info() != Eigen::Success) continue;
                    Eigen::Vector3d eigenvalues = eigensolver.eigenvalues();
                    if (eigenvalues.minCoeff() <= 0) continue; // 양정치 아님

                    cell.stGauCell_mat3d_inv_cov = mat3d_inv_cov;
                    cell.stGauCell_d_det = mat3d_inv_cov.determinant();

                    m_vec_stGauCell_gaussian_cells.push_back(cell);  // 가우시안들의 목록 : 평균,공분산 갖고있음
                }
            }
        }
    }
}



//구 형태로 필터링
void FilterPointCloudBySphere(
    pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud,
    pcl::PointCloud<pcl::PointXYZ>::Ptr output_cloud,
    const Eigen::Vector3f& center,
    float radius)
{
    // 포인트 클라우드 필터링 결과 저장
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices());

    // 각 점이 중심으로부터 반경 안에 있는지 확인
    for (size_t i = 0; i < input_cloud->points.size(); ++i) {
        const auto& point = input_cloud->points[i];

        if (std::abs(point.x - center[0]) > radius) continue;
        if (std::abs(point.y - center[1]) > radius) continue;

        float dx = point.x - center[0];
        float dy = point.y - center[1];

        if (dx * dx + dy * dy > radius*radius) continue;
        
        inliers->indices.push_back(i);
    }

    // 필터링 수행
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(input_cloud);
    extract.setIndices(inliers);
    extract.setNegative(false); // 반경 내 점만 추출
    extract.filter(*output_cloud);
}



// NDT 콜백 함수
void LOCALIZATION::NDTCallback(const sensor_msgs::PointCloud2ConstPtr& msg) {

    m_cfg_uint32_message_count++;
    std::cout << m_cfg_uint32_message_count << "번째 입력데이터 들어옴." << std::endl;

    std::lock_guard<std::mutex> lock(m_std_queue_mutex);
    // 큐의 크기를 제한하여 오래된 데이터를 버림
    if (m_std_lidar_data_queue.size() < 100) {
        m_std_lidar_data_queue.push(msg);
        std::cout << "queue size : " << m_std_lidar_data_queue.size() << std::endl;
        m_std_data_condition.notify_one();
    } else {
        // 큐가 가득 찬 경우 가장 오래된 데이터를 버리고 새로운 데이터를 추가
        m_std_lidar_data_queue.pop();
        m_std_lidar_data_queue.push(msg);
    }
}

// NDT처리를 위한 스레드 함수
void LOCALIZATION::NDTProcessingThread() {
    while (m_b_is_running) {
        sensor_msgs::PointCloud2ConstPtr msg;

        // 큐에서 데이터 가져오기
        {
            std::unique_lock<std::mutex> lock(m_std_queue_mutex);
            m_std_data_condition.wait(lock, [this] {
                return !m_std_lidar_data_queue.empty() || !m_b_is_running;
            });

            if (!m_b_is_running && m_std_lidar_data_queue.empty()) {
                break;
            }

            msg = m_std_lidar_data_queue.front();
            m_std_lidar_data_queue.pop();
        }

        // NDT 처리 함수 호출
        std::cout << "[DEBUG_001] NDT 호출..." << std::endl;
        ProcessNDT(msg);
    }
}

void LOCALIZATION::ProcessNDT(const sensor_msgs::PointCloud2ConstPtr& msg)
{

    // PCL로 변환
    pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::fromROSMsg(*msg, *input_cloud);

    // target Map 구형태로 필터링
    Eigen::Vector3f map_center(m_matrix4d_initial_esti(0, 3),
                        m_matrix4d_initial_esti(1, 3),
                        m_matrix4d_initial_esti(2, 3));

    if (m_pcl_map_cloud->empty()) {
        ROS_ERROR("[ERROR_002] m_pcl_map_cloud is empty.");
        return;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_map(new pcl::PointCloud<pcl::PointXYZ>());
    FilterPointCloudBySphere(m_pcl_map_cloud, filtered_map, map_center, m_f_radius_m + 2.0);
    if(m_b_debugmode){
        std::cout << "filtered_map size : " << filtered_map->size() << std::endl;
    }
    PublishPointCloud(filtered_map, m_ros_filtered_map_pub, "velo_link");
    m_pcl_filtered_map_ptr = filtered_map;

    // Input Scan data 필터링 및 크기 제한
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_input_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    //pcl::ApproximateVoxelGrid<pcl::PointXYZ> voxel_grid_filter;
    pcl::VoxelGrid<pcl::PointXYZ> voxel_grid_filter;
    voxel_grid_filter.setLeafSize(1.5, 1.5, 1.5);  // 다운샘플링 1.0
    voxel_grid_filter.setInputCloud(input_cloud);
    voxel_grid_filter.filter(*filtered_input_cloud);

    Eigen::Vector3f input_center(0, 0, 0);

    FilterPointCloudBySphere(filtered_input_cloud, filtered_input_cloud, input_center, m_f_radius_m);
    PublishPointCloud(filtered_input_cloud, m_ros_filtered_input_pub, "velo_link");

    LOCALIZATION::setInputTarget(filtered_map);
    LOCALIZATION::setInputSource(filtered_input_cloud);


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
        loc.PublishPointCloud(loc.m_pcl_map_cloud, loc.m_ros_map_pub, "velo_link");
        ros::spinOnce();
        loop_rate.sleep();
    }

    return 0;
}


