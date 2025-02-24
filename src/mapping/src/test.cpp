#include <open3d/Open3D.h>
#include <filesystem>
#include <fstream>
#include <vector>
#include <iostream>
#include <utility> 

#include <open3d/visualization/visualizer/Visualizer.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>


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
int main() {
    try {
        std::string directory_bin = "/LHS/approach/my_code/dataset/00/velodyne";


        ExtendedPointCloud accumulated_pcd;  // 누적된 PointCloud

        size_t total_files = 0;
        for (const auto& entry : std::filesystem::directory_iterator(directory_bin)) {
            if (entry.path().extension() == ".bin") {
                ++total_files;
            }
        }
        size_t idx = 0;
        for (const auto& entry : std::filesystem::directory_iterator(directory_bin)) {
            if (entry.path().extension() == ".bin") {
                std::string file_path = entry.path().string();
                std::string file_name = entry.path().stem().string();

                // Load the .bin file as a PointCloud
                ExtendedPointCloud pcd = LoadBinToPCD(file_path);

                //누적
                accumulated_pcd.points_.insert(accumulated_pcd.points_.end(), pcd.points_.begin(), pcd.points_.end());
  
               ++idx;
            }
        }

        // 저장할 파일 경로 설정
        std::string output_path = "/LHS/approach/my_code/output/accumulated_pointcloud.pcd";
        // Open3D의 WritePointCloud 함수로 저장
        if (open3d::io::WritePointCloud(output_path, accumulated_pcd)) {
            std::cout << "Saved accumulated point cloud to " << output_path << std::endl;
        } else {
            std::cerr << "Failed to save point cloud to " << output_path << std::endl;
        }


    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}