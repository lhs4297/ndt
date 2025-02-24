#include "header/ceres_ndt.hpp"
#include <iostream>

#include <pcl/common/transforms.h>
#include <ceres/ceres.h>

#include <pcl/kdtree/kdtree_flann.h>




void NormalDistributionsTransform::computeTransformation(pcl::PointCloud<pcl::PointXYZ>& output,
                                                         const Eigen::Matrix4d& cT_matrix4d_initial_esti)
{
  nr_iterations_ = 0;
  converged_ = false;
  if (target_cells_.getCentroids()->empty()) {
    PCL_ERROR("[%s::computeTransformation] Voxel grid is not searchable!\n",
              getClassName().c_str());
    return;
  }

  // Initializes the gaussian fitting parameters (eq. 6.8) [Magnusson 2009]
  const double gauss_c1 = 10 * (1 - outlier_ratio_);
  const double gauss_c2 = outlier_ratio_ / pow(resolution_, 3);
  const double gauss_d3 = -std::log(gauss_c2);
  gauss_d1_ = -std::log(gauss_c1 + gauss_c2) - gauss_d3;
  gauss_d2_ =
      -2 * std::log((-std::log(gauss_c1 * std::exp(-0.5) + gauss_c2) - gauss_d3) /
                    gauss_d1_);

  if (cT_matrix4d_initial_esti != Eigen::Matrix4d::Identity()) {
    // Initialise final transformation to the cT_matrix4d_initial_estied one
    m_mat4d_final_trans = cT_matrix4d_initial_esti;
    // Apply cT_matrix4d_initial_estied transformation prior to search for neighbours
    transformPointCloud(output, output, cT_matrix4d_initial_esti);
  }

  // Initialize Point Gradient and Hessian

  point_jacobian_.setZero();
  point_jacobian_.block<3, 3>(0, 0).setIdentity();
  point_hessian_.setZero();

  Eigen::Transform<double, 3, Eigen::Affine, Eigen::ColMajor> eig_transformation;
  eig_transformation.matrix() = m_mat4d_final_trans;

  // Convert initial cT_matrix4d_initial_esti matrix to 6 element transformation vector
  Eigen::Matrix<double, 6, 1> transform, score_gradient;
  Eigen::Matrix<double, 3, 1> init_translation = eig_transformation.translation();
  Eigen::Matrix<double, 3, 1> init_rotation = eig_transformation.rotation().eulerAngles(0, 1, 2);
  transform << init_translation.template cast<double>(),
      init_rotation.template cast<double>();

  Eigen::Matrix<double, 6, 6> hessian;

  // Calculate derivates of initial transform vector, subsequent derivative calculations
  // are done in the step length determination.
  double score = computeDerivatives(score_gradient, hessian, output, transform);

  while (!converged_) {
    // Store previous transformation
    m_mat4d_pre_trans = m_mat4d_trans;

    // Solve for decent direction using newton method, line 23 in Algorithm 2 [Magnusson
    // 2009]
    Eigen::JacobiSVD<Eigen::Matrix<double, 6, 6>> sv(
        hessian, Eigen::ComputeFullU | Eigen::ComputeFullV);
    // Negative for maximization as opposed to minimization
    Eigen::Matrix<double, 6, 1> delta = sv.solve(-score_gradient);

    // Calculate step length with guaranteed sufficient decrease [More, Thuente 1994]
    double delta_norm = delta.norm();

    if (delta_norm == 0 || std::isnan(delta_norm)) {
      trans_likelihood_ = score / static_cast<double>(m_cptr_input->size());
      converged_ = delta_norm == 0;
      return;
    }

    delta /= delta_norm;
    delta_norm = computeStepLengthMT(transform,
                                     delta,
                                     delta_norm,
                                     m_d_step_size,
                                     m_d_trans_delta / 2,
                                     score,
                                     score_gradient,
                                     hessian,
                                     output);
    delta *= delta_norm;

    // Convert delta into matrix form
    convertTransform(delta, m_mat4d_trans);

    transform += delta;

    // // 시각화 하는 부분이라 스킵
    // if (update_visualizer_)
    //   update_visualizer_(output, pcl::Indices(), *target_, pcl::Indices());

    const double cos_angle =
        0.5 * (m_mat4d_trans.template block<3, 3>(0, 0).trace() - 1);
    const double translation_sqr =
        m_mat4d_trans.template block<3, 1>(0, 3).squaredNorm();

    nr_iterations_++;

    if (nr_iterations_ >= max_iterations_ ||
        ((m_d_trans_delta > 0 && translation_sqr <= m_d_trans_delta) &&
         (m_d_rot_delta > 0 &&
          cos_angle >= m_d_rot_delta)) ||
        ((m_d_trans_delta <= 0) &&
         (m_d_rot_delta > 0 &&
          cos_angle >= m_d_rot_delta)) ||
        ((m_d_trans_delta > 0 && translation_sqr <= m_d_trans_delta) &&
         (m_d_rot_delta <= 0))) {
      converged_ = true;
    }
  }

  // Store transformation likelihood.  The relative differences within each scan
  // registration are accurate but the normalization constants need to be modified for
  // it to be globally accurate
  trans_likelihood_ = score / static_cast<double>(m_cptr_input->size());
}


double NormalDistributionsTransform::computeDerivatives(Eigen::Matrix<double, 6, 1>& score_gradient,
                                                        Eigen::Matrix<double, 6, 6>& hessian,
                                                        const pcl::PointCloud<pcl::PointXYZ>& trans_cloud,
                                                        const Eigen::Matrix<double, 6, 1>& transform,
                                                        bool compute_hessian)
{
    score_gradient.setZero();
    hessian.setZero();
    double score = 0;

    ceres::Problem problem;

    LOCALIZATION loc;
  
    // Precompute Angular Derivatives
    computeAngleDerivatives(transform, true);

    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    pcl::PointCloud<pcl::PointXYZ>::Ptr gaussian_centroids(new pcl::PointCloud<pcl::PointXYZ>());

    for (const auto& cell : loc.m_vec_stGauCell_gaussian_cells) {
        pcl::PointXYZ centroid(cell.stGauCell_vec3d_mean.x(),
                              cell.stGauCell_vec3d_mean.y(),
                              cell.stGauCell_vec3d_mean.z());
        gaussian_centroids->points.push_back(centroid);
    }
    kdtree.setInputCloud(gaussian_centroids);

    // Update gradient and hessian for each point
    for (std::size_t idx = 0; idx < m_cptr_input->size(); idx++) {
        const auto& x_trans_pt = trans_cloud[idx];

        const auto& x_pt = (*m_cptr_input)[idx];
        Eigen::Vector3d x = Eigen::Vector3d(x_pt.x, x_pt.y, x_pt.z);

        // Find neighbors
        std::vector<int> neighborhood; // Use appropriate data structure
        std::vector<float> distances;
        
        // Implement a radius search logic here
        if (kdtree.radiusSearch(x_trans_pt, resolution_, neighborhood, distances) > 0) {
            for (int neighbor_idx : neighborhood) {  // 반환된 인덱스 사용
                const GaussianCell& cell = loc.m_vec_stGauCell_gaussian_cells[neighbor_idx];
                Eigen::Vector3d mean = cell.getMean();
                Eigen::Matrix3d covariance_inverse = cell.getInverseCov();

                ceres::CostFunction* cost_function =
                    new ceres::AutoDiffCostFunction<NDTResidual, 1, 6>(
                        new NDTResidual(x, mean, covariance_inverse));  // `x`는 현재 점의 위치

                problem.AddResidualBlock(cost_function, nullptr, transform.data());
            }
        }
    }
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;  // 선형 솔버 선택
    options.max_num_iterations = 100;             // 최대 반복 횟수
    options.minimizer_progress_to_stdout = true;  // 진행 상황 출력

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    return score;
}

void NormalDistributionsTransform::computeAngleDerivatives(const Eigen::Matrix<double, 6, 1>& transform,
                                                           bool compute_hessian)
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


void NormalDistributionsTransform::computePointDerivatives(const Eigen::Vector3d& x, bool compute_hessian)
{
  // Calculate first derivative of Transformation Equation 6.17 w.r.t. transform vector.
  // Derivative w.r.t. ith element of transform vector corresponds to column i,
  // Equation 6.18 and 6.19 [Magnusson 2009]
  Eigen::Matrix<double, 8, 1> point_angular_jacobian =
      angular_jacobian_ * Eigen::Vector4d(x[0], x[1], x[2], 0.0);
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

double NormalDistributionsTransform::updateDerivatives(
    Eigen::Matrix<double, 6, 1>& score_gradient,
    Eigen::Matrix<double, 6, 6>& hessian,
    const Eigen::Vector3d& x_trans,
    const Eigen::Matrix3d& c_inv,
    bool compute_hessian)
{
    // Update derivatives logic
    return 0.0;
}

void CeresSolverLocalization(const Eigen::Affine3d CSL_matrix, const Eigen::Affine3d CSL_init_pose)
{
  Eigen::Quaternion<double> quaternion(track_pose_.linear());
  Eigen::Vector3d translation = track_pose_.translation();
  q_w_curr = quaternion;
	t_w_curr = translation;

  ceres::LossFunction *loss_function = NULL;
  ceres::LocalParameterization *q_parameterization = new ceres::EigenQuaternionParameterization();
  ceres::Problem::Options problem_options;
  ceres::Problem problem(problem_options);
  problem.AddParameterBlock(parameters, 4, q_parameterization);
  problem.AddParameterBlock(parameters + 4, 3);

  // Transform Map Point Cloud
  pcl::PointCloud<PointType>::Ptr transform_meas_ptr(new pcl::PointCloud<PointType>());
  pcl::transformPointCloud(*meas_ptr, *transform_meas_ptr, track_pose_.matrix());

  // Connect nearest point
  TicToc t_matching;
  // TicToc t_correspondence;
  for (int i = 0; i < static_cast<int>(meas_ptr->size());i++) {
    Eigen::Vector3d trans_point((*transform_meas_ptr)[i].x, (*transform_meas_ptr)[i].y, (*transform_meas_ptr)[i].z);
    Eigen::Vector3i cell_index = map_.ConvertToCellIndex(trans_point);
    int64_t hash_code = map_.ConvertToHashCode(cell_index);
    NDcPtr cell_ptr = map_.GetCell(hash_code);
    if(cell_ptr == NULL){
      continue;
    }

    Eigen::Vector3d curr_point((*meas_ptr)[i].x, (*meas_ptr)[i].y, (*meas_ptr)[i].z);
    Eigen::Vector3d mean = cell_ptr->GetMean();
    Eigen::Matrix3d eigenVectors = cell_ptr->GetEigenVector();
    Eigen::Matrix3d eigenSqrtValues = cell_ptr->GetSqrtEigenValue();

    ceres::CostFunction *cost_function = LidarNormalDistributionFactor::Create(curr_point, mean, eigenSqrtValues, eigenVectors);
    problem.AddResidualBlock(cost_function, loss_function, parameters, parameters + 4);
  }     
  // std::cout << "Correspondence: " << t_correspondence.toc() / 1e3 << std::endl;

  // Solve the solver
  // TicToc t_solver;
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  options.max_num_iterations = 100;
  options.minimizer_progress_to_stdout = false;
  options.check_gradients = false;
  options.gradient_check_relative_precision = 1e-4;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  // std::cout << "Solver: " << t_solver.toc() / 1e3 << std::endl;
  // std::cout << "next parameters: " << parameters[4] << ", "<< parameters[5] << ", "<< parameters[6]  << std::endl;
  double processing_time_ms = t_matching.toc();
  std::cout << "Matching of Points [" << meas_ptr->size() << "]: " << processing_time_ms << " ms" << std::endl;

  // Convert to affine3d
  q_w_curr.normalize();
  track_pose_.translation() = t_w_curr;
  track_pose_.linear() = q_w_curr.toRotationMatrix();

}