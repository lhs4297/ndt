#include <open3d/Open3D.h>
#include <filesystem>
#include <fstream>
#include <vector>
#include <iostream>
#include <utility> 
#include <algorithm>

#include <open3d/visualization/visualizer/Visualizer.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>



struct LabelData {
    std::vector<int> semantic_labels;
    std::vector<int> instance_labels;
};


class ExtendedPointCloud : public open3d::geometry::PointCloud {
public:
    // 추가 필드: intensity
    std::vector<double> intensities_;

    // Intensity 값을 설정하는 함수
    void SetIntensities(const std::vector<double>& intensities) {
            // std::cout << "points size : " << points_.size() << std::endl;
            // std::cout << "Intensity size : " << intensities.size() << std::endl;
        if (intensities.size() != points_.size()) {
            throw std::runtime_error("Intensity size must match the number of points");
        }
        intensities_ = intensities;
    }

    // Intensity를 반환하는 함수
    const std::vector<double>& GetIntensities() const {
        return intensities_;
    }
};

ExtendedPointCloud LoadBinToPCD(const std::string &file_path) {
    // Open the binary file
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file: " + file_path);
    }

    // Read the file size to determine the number of points
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    if (file_size % (sizeof(float) * 4) != 0) {
        throw std::runtime_error("Invalid binary file format: size mismatch");
    }

    size_t num_points = file_size / (sizeof(float) * 4);

    // Create vectors to store points and intensity values
    std::vector<Eigen::Vector3d> points;
    std::vector<double> intensities;

    points.reserve(num_points);
    intensities.reserve(num_points);

    for (size_t i = 0; i < num_points; ++i) {
        float x, y, z, intensity;
        file.read(reinterpret_cast<char*>(&x), sizeof(float));
        file.read(reinterpret_cast<char*>(&y), sizeof(float));
        file.read(reinterpret_cast<char*>(&z), sizeof(float));
        file.read(reinterpret_cast<char*>(&intensity), sizeof(float));

        points.emplace_back(x, y, z);
        intensities.push_back(static_cast<double>(intensity));
    }

    file.close();

    // Create an ExtendedPointCloud
    ExtendedPointCloud pcd;
    pcd.points_ = points;
    pcd.SetIntensities(intensities);

    return pcd;
}

pcl::PointCloud<pcl::PointXYZI>::Ptr ApplyVoxelGridFilter(
    pcl::PointCloud<pcl::PointXYZI>::Ptr input_cloud, float voxel_size) {
    pcl::VoxelGrid<pcl::PointXYZI> voxel_filter;
    voxel_filter.setInputCloud(input_cloud);
    voxel_filter.setLeafSize(voxel_size, voxel_size, voxel_size);

    auto filtered_cloud = boost::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    voxel_filter.filter(*filtered_cloud);

    return filtered_cloud;
}
// Open3D 점군 데이터를 PCL로 변환
pcl::PointCloud<pcl::PointXYZI>::Ptr ConvertToPCL(const ExtendedPointCloud& open3d_pcd) {
    auto pcl_pcd = boost::make_shared<pcl::PointCloud<pcl::PointXYZI>>();

    //std::cout << "First 5 points from Open3D:" << std::endl;
    // for (size_t i = 0; i < std::min(size_t(5), open3d_pcd.points_.size()); ++i) {
    //     std::cout << open3d_pcd.points_[i].transpose() << " "
    //             << open3d_pcd.intensities_[i] << std::endl;
    // }

    for (size_t i = 0; i < open3d_pcd.points_.size(); ++i) {
        pcl::PointXYZI point;
        point.x = open3d_pcd.points_[i].x();
        point.y = open3d_pcd.points_[i].y();
        point.z = open3d_pcd.points_[i].z();
        point.intensity = static_cast<float>(open3d_pcd.intensities_[i]);
        pcl_pcd->push_back(point);
    }
    //std::cout << "First 5 points from PCL:" << std::endl;
    for (size_t i = 0; i < std::min(size_t(5), pcl_pcd->points.size()); ++i) {
       // std::cout << pcl_pcd->points[i].x << " " << pcl_pcd->points[i].y << " "
          //      << pcl_pcd->points[i].z << " " << pcl_pcd->points[i].intensity << std::endl;
    }

    return pcl_pcd;
}

ExtendedPointCloud ConvertToOpen3D(pcl::PointCloud<pcl::PointXYZI>::Ptr pcl_pcd) {
    ExtendedPointCloud open3d_pcd;

    for (const auto& point : pcl_pcd->points) {
        open3d_pcd.points_.emplace_back(point.x, point.y, point.z);
        open3d_pcd.intensities_.push_back(static_cast<double>(point.intensity));
    }

    return open3d_pcd;
}

std::unordered_map<std::string, Eigen::Matrix4d> LoadCalibration(const std::string& file_path) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open calibration file: " + file_path);
    }

    std::unordered_map<std::string, Eigen::Matrix4d> calib;
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string key;
        iss >> key;
        key.pop_back(); // Remove the colon

        Eigen::Matrix4d matrix = Eigen::Matrix4d::Identity();
        for (int i = 0; i < 12; ++i) {
            iss >> matrix(i / 4, i % 4);
        }

        calib[key] = matrix;
    }

    file.close();
    return calib;
}

std::vector<Eigen::Matrix4d> LoadPosesLimited(const std::string& file_path,
                                              const Eigen::Matrix4d& Tr,
                                              size_t file_num)
{
    std::ifstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open poses file: " + file_path);
    }

    std::vector<Eigen::Matrix4d> poses;
    poses.reserve(file_num);  // 미리 공간 할당 (optional)

    Eigen::Matrix4d Tr_inv = Tr.inverse();

    std::string line;
    size_t count = 0;
    while (count < file_num && std::getline(file, line)) {
        std::istringstream iss(line);
        Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();

        // Read the pose elements in row-major order
        iss >> pose(0, 0) >> pose(0, 1) >> pose(0, 2) >> pose(0, 3)
            >> pose(1, 0) >> pose(1, 1) >> pose(1, 2) >> pose(1, 3)
            >> pose(2, 0) >> pose(2, 1) >> pose(2, 2) >> pose(2, 3);

        // Adjust for coordinate system transformation
        pose = Tr_inv * pose * Tr;

        poses.push_back(pose);
        ++count;
    }

    file.close();
    return poses;
}

LabelData LoadLabelFile(const std::string &file_path) {
    // Open the binary label file
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open label file: " + file_path);
    }

    // Read labels from the file
    std::vector<int> semantic_labels;
    std::vector<int> instance_labels;
    uint32_t label;

    while (file.read(reinterpret_cast<char*>(&label), sizeof(label))) {
        // Extract semantic and instance labels
        int semantic_label = label & 0xFFFF;         // 하위 16비트
        int instance_label = (label >> 16) & 0xFFFF; // 상위 16비트

        semantic_labels.push_back(semantic_label);
        instance_labels.push_back(instance_label);
    }

    file.close();

    // Return the labels as a structured object
    return {semantic_labels, instance_labels};
}

void TransformPointCloud(open3d::geometry::PointCloud& pcd, const Eigen::Matrix4d& transformation) {
    for (auto& point : pcd.points_) {
        Eigen::Vector4d p(point[0], point[1], point[2], 1.0);
        Eigen::Vector4d p_transformed = transformation * p;
        point = Eigen::Vector3d(p_transformed[0], p_transformed[1], p_transformed[2]);
    }
}

void removeMovingClass(std::vector<Eigen::Matrix<double, 3, 1>>& points,
                      std::vector<double>& remissions,
                      std::vector<int>& semLabel,
                      std::vector<int>& instLabel) {

    // Define moving class labels
    std::vector<int> movingClass = {
            252, // moving-car
            253, // moving-person
            254, // moving-motorcyclist
            255, // moving-on-rails
            256, // moving-bus
            257, // moving-truck
            258, // moving-other-vehicle
            259
        };

    // Create a mask to filter out points corresponding to moving classes
    std::vector<bool> mask(semLabel.size(), true);

    for (size_t i = 0; i < semLabel.size(); ++i) {
        if (std::find(movingClass.begin(), movingClass.end(), semLabel[i]) != movingClass.end()) {
            mask[i] = false; // Mark as false if it's in the movingClass
        }
    }

    // Filter points, remissions, semLabel, and instLabel using the mask
    auto filterData = [&mask](auto& data) {
        using T = typename std::decay<decltype(data)>::type::value_type; // Deduce type
        std::vector<T> filtered;
        for (size_t i = 0; i < data.size(); ++i) {
            if (mask[i]) {
                filtered.push_back(data[i]);
            }
        }
        data.swap(filtered);
    };

    // Specialize filterData for points (std::vector<Eigen::Matrix<double, 3, 1>>)
    auto filterPoints = [&mask](std::vector<Eigen::Matrix<double, 3, 1>>& points) {
        std::vector<Eigen::Matrix<double, 3, 1>> filtered;
        for (size_t i = 0; i < points.size(); ++i) {
            if (mask[i]) {
                filtered.push_back(points[i]);
            }
        }
        points.swap(filtered);
    };

    // Apply filters
    filterPoints(points);
    filterData(remissions);
    filterData(semLabel);
    filterData(instLabel);
}

void VisualizePointCloud(const ExtendedPointCloud& pcd) {
    // Open3D에서 지원하는 포인트 클라우드 시각화
    open3d::visualization::DrawGeometries({std::make_shared<ExtendedPointCloud>(pcd)});
}

#include <open3d/Open3D.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

// poses 벡터에서 x,y,z만 추출 후 Open3D PointCloud로 변환
open3d::geometry::PointCloud ConvertPosesToOpen3DPointCloud(const std::vector<Eigen::Matrix4d>& poses) {
    open3d::geometry::PointCloud pose_pcd;

    for (const auto& pose : poses) {
        // pose는 4x4 변환행렬
        double x = pose(0, 3);
        double y = pose(1, 3);
        double z = pose(2, 3);

        pose_pcd.points_.emplace_back(x, y, z);
    }

    return pose_pcd;
}

// poses 벡터에서 x,y,z만 추출 후 PCL PointCloud로 변환
pcl::PointCloud<pcl::PointXYZ>::Ptr ConvertPosesToPCLPointCloud(const std::vector<Eigen::Matrix4d>& poses) {
    auto pose_cloud = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    pose_cloud->reserve(poses.size());

    for (const auto& pose : poses) {
        pcl::PointXYZ pt;
        pt.x = static_cast<float>(pose(0, 3));
        pt.y = static_cast<float>(pose(1, 3));
        pt.z = static_cast<float>(pose(2, 3));
        pose_cloud->push_back(pt);
    }

    return pose_cloud;
}

int main() {
    try {
        // 원하는 개수
        size_t file_num = 4541;

        // 디렉토리 경로
        std::string directory_bin = "/LHS/approach/my_code/dataset/00/velodyne";
        std::string directory_label = "/LHS/approach/my_code/dataset/00/labels";
        std::string directory_pose = "/LHS/approach/my_code/dataset/00/poses.txt";
        std::string directory_calib = "/LHS/approach/my_code/dataset/00/calib.txt";

        // (1) Calibration
        auto calib = LoadCalibration(directory_calib);

        // (2) bin file 목록을 정렬해서 최대 file_num개만 추출
        std::vector<std::filesystem::path> bin_files;
        for (const auto& entry : std::filesystem::directory_iterator(directory_bin)) {
            if (entry.path().extension() == ".bin") {
                bin_files.push_back(entry.path());
            }
        }
        std::sort(bin_files.begin(), bin_files.end());
        if (bin_files.size() > file_num) {
            bin_files.resize(file_num);
        }

        // (3) label file 목록을 정렬해서 bin_files.size()까지
        std::vector<std::filesystem::path> label_files;
        for (const auto& entry : std::filesystem::directory_iterator(directory_label)) {
            if (entry.path().extension() == ".label") {
                label_files.push_back(entry.path());
            }
        }
        std::sort(label_files.begin(), label_files.end());
        if (label_files.size() > bin_files.size()) {
            label_files.resize(bin_files.size());
        }

        // (4) poses.txt에서 bin_files.size() 줄까지만 읽기
        auto poses = LoadPosesLimited(directory_pose, calib["Tr"], bin_files.size());
        // or
        // auto poses_all = LoadPoses(directory_pose, calib["Tr"]);
        // poses_all.resize(std::min(poses_all.size(), bin_files.size()));

        // ---- 누적할 PointCloud
        ExtendedPointCloud accumulated_pcd;

        // (5) 반복문
        for (size_t i = 0; i < bin_files.size(); ++i) {
            std::string bin_path = bin_files[i].string();
            std::string label_path = label_files[i].string();

            // Load bin & label
            ExtendedPointCloud pcd = LoadBinToPCD(bin_path);
            LabelData labels = LoadLabelFile(label_path);

            if (labels.semantic_labels.size() != pcd.points_.size()) {
                throw std::runtime_error("Mismatch bin/label: " + bin_path);
            }

            // Remove moving classes
            removeMovingClass(pcd.points_, pcd.intensities_, 
                              labels.semantic_labels, labels.instance_labels);

            // Transform by pose[i]
            TransformPointCloud(pcd, poses[i]);

            // 예: 개별 복셀화
            auto pcl_cloud = ConvertToPCL(pcd);
            auto voxelized_cloud = ApplyVoxelGridFilter(pcl_cloud, 0.4f);
            ExtendedPointCloud voxelized_pcd = ConvertToOpen3D(voxelized_cloud);

            // 누적
            accumulated_pcd.points_.insert(accumulated_pcd.points_.end(),
                                           voxelized_pcd.points_.begin(),
                                           voxelized_pcd.points_.end());
            accumulated_pcd.intensities_.insert(accumulated_pcd.intensities_.end(),
                                                voxelized_pcd.intensities_.begin(),
                                                voxelized_pcd.intensities_.end());

            std::cout << "Processed file " << (i+1) << " / " << bin_files.size() << std::endl;
        }

        // (6) 전체 누적 데이터 2차 복셀화 등...
        auto pcl_cloud_acc = ConvertToPCL(accumulated_pcd);
        float voxel_size = 0.4f;
        auto voxelized_acc_cloud = ApplyVoxelGridFilter(pcl_cloud_acc, voxel_size);
        ExtendedPointCloud final_pcd = ConvertToOpen3D(voxelized_acc_cloud);

        // 저장
        std::string output_path = "/LHS/approach/my_code/output/accumulated_pointcloud.pcd";
        open3d::io::WritePointCloud(output_path, final_pcd);

        // (추가) pose 위치만 pcd로 저장
        auto pose_pcd = ConvertPosesToOpen3DPointCloud(poses);
        std::string output_pose_pcd = "/LHS/approach/my_code/output/poses_xyz_only.pcd";
        open3d::io::WritePointCloud(output_pose_pcd, pose_pcd);

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    return 0;
}

// int main() {
//     try {
//         std::string directory_bin = "/LHS/approach/my_code/dataset/00/velodyne";
//         std::string directory_label = "/LHS/approach/my_code/dataset/00/labels";
//         std::string directory_pose = "/LHS/approach/my_code/dataset/00/poses.txt";
//         std::string directory_calib = "/LHS/approach/my_code/dataset/00/calib.txt";

//         // Load calibration and poses
//         auto calib = LoadCalibration(directory_calib);
//         auto poses = LoadPoses(directory_pose, calib["Tr"]);

//         float voxel_size_individual = 0.2f;
//         size_t idx = 0;
//         size_t total_files = 0;
//         size_t chunk_size = 1000; // 처리할 파일 수의 청크 크기
//         size_t chunk_count = 0;

//         for (const auto& entry : std::filesystem::directory_iterator(directory_bin)) {
//             if (entry.path().extension() == ".bin") {
//                 ++total_files;
//             }
//         }

//         ExtendedPointCloud accumulated_pcd; // 청크 내 누적된 PointCloud

//         for (const auto& entry : std::filesystem::directory_iterator(directory_bin)) {
//             if (entry.path().extension() == ".bin") {
//                 std::string file_path = entry.path().string();
//                 std::string file_name = entry.path().stem().string();

//                 // Load the .bin file as a PointCloud
//                 ExtendedPointCloud pcd = LoadBinToPCD(file_path);

//                 // Load the corresponding .label file
//                 std::string label_path = directory_label + "/" + file_name + ".label";
//                 LabelData labels = LoadLabelFile(label_path);

//                 if (labels.semantic_labels.size() != pcd.points_.size()) {
//                     throw std::runtime_error("Mismatch between points and labels for file: " + file_name);
//                 }

//                 // Remove moving classes
//                 removeMovingClass(pcd.points_, pcd.intensities_, labels.semantic_labels, labels.instance_labels);

//                 // Transform the PointCloud using the pose
//                 TransformPointCloud(pcd, poses[idx]);

//                 // 개별 복셀화: Open3D -> PCL 변환 후 복셀화
//                 auto pcl_cloud = ConvertToPCL(pcd);
//                 auto voxelized_cloud = ApplyVoxelGridFilter(pcl_cloud, voxel_size_individual);

//                 // PCL -> Open3D 변환
//                 ExtendedPointCloud voxelized_pcd = ConvertToOpen3D(voxelized_cloud);

//                 // 누적
//                 accumulated_pcd.points_.insert(accumulated_pcd.points_.end(), voxelized_pcd.points_.begin(), voxelized_pcd.points_.end());
//                 accumulated_pcd.intensities_.insert(accumulated_pcd.intensities_.end(), voxelized_pcd.intensities_.begin(), voxelized_pcd.intensities_.end());

//                 ++idx;

//                 // 청크 처리 완료 후 저장
//                 if (idx % chunk_size == 0 || idx == total_files) {
//                     std::string chunk_output_path = "/LHS/approach/my_code/output/chunk_" + std::to_string(chunk_count) + ".pcd";
//                     if (open3d::io::WritePointCloud(chunk_output_path, accumulated_pcd)) {
//                         std::cout << "Saved chunk to " << chunk_output_path << std::endl;
//                     } else {
//                         std::cerr << "Failed to save chunk to " << chunk_output_path << std::endl;
//                     }

//                     accumulated_pcd.Clear(); // 메모리 확보를 위해 누적된 데이터를 초기화
//                     ++chunk_count;
//                 }
//                 std::cout << "Processing file " << idx << " / " << total_files << std::endl;
//             }
//         }

//         // 모든 청크 파일 병합
//         ExtendedPointCloud final_accumulated_pcd;
//         for (size_t i = 0; i < chunk_count; ++i) {
//             std::string chunk_file_path = "/LHS/approach/my_code/output/chunk_" + std::to_string(i) + ".pcd";
//             ExtendedPointCloud chunk_pcd;
//             open3d::io::ReadPointCloud(chunk_file_path, chunk_pcd);
//             final_accumulated_pcd.points_.insert(final_accumulated_pcd.points_.end(), chunk_pcd.points_.begin(), chunk_pcd.points_.end());
//             final_accumulated_pcd.intensities_.insert(final_accumulated_pcd.intensities_.end(), chunk_pcd.intensities_.begin(), chunk_pcd.intensities_.end());
//         }

//         // 최종 누적 데이터를 복셀화
//         auto pcl_cloud = ConvertToPCL(final_accumulated_pcd);
//         float voxel_size = 0.8f; // 최종 복셀 크기
//         auto voxelized_cloud = ApplyVoxelGridFilter(pcl_cloud, voxel_size);
//         ExtendedPointCloud final_voxelized_pcd = ConvertToOpen3D(voxelized_cloud);

//         // 최종 파일 저장
//         std::string output_path = "/LHS/approach/my_code/output/accumulated_pointcloud.pcd";
//         if (open3d::io::WritePointCloud(output_path, final_voxelized_pcd)) {
//             std::cout << "Saved accumulated point cloud to " << output_path << std::endl;
//         } else {
//             std::cerr << "Failed to save point cloud to " << output_path << std::endl;
//         }

//     } catch (const std::exception& e) {
//         std::cerr << "Error: " << e.what() << std::endl;
//     }

//     return 0;
// }
