#include <iostream>
#include <vector>
#include <Eigen/Dense>

template <typename PointSource, typename PointTarget, typename Scalar>
void
NormalDistributionsTransform<PointSource, PointTarget, Scalar>::computeTransformation(
    PointCloudSource& cT_input, const Matrix4& guess)
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

  if (guess != Matrix4::Identity()) {
    // Initialise final transformation to the guessed one
    final_transformation_ = guess;
    // Apply guessed transformation prior to search for neighbours
    transformPointCloud(cT_input, output, guess);
  }

  // Initialize Point Gradient and Hessian
  point_jacobian_.setZero();
  point_jacobian_.block<3, 3>(0, 0).setIdentity();
  point_hessian_.setZero();

  Eigen::Transform<Scalar, 3, Eigen::Affine, Eigen::ColMajor> eig_transformation;
  eig_transformation.matrix() = final_transformation_;

  // Convert initial guess matrix to 6 element transformation vector
  Eigen::Matrix<double, 6, 1> transform, score_gradient;
  Vector3 init_translation = eig_transformation.translation();
  Vector3 init_rotation = eig_transformation.rotation().eulerAngles(0, 1, 2);
  transform << init_translation.template cast<double>(),
      init_rotation.template cast<double>();

  Eigen::Matrix<double, 6, 6> hessian;

  // Calculate derivates of initial transform vector, subsequent derivative calculations
  // are done in the step length determination.
  double score = computeDerivatives(score_gradient, hessian, output, transform);

  while (!converged_) {
    // Store previous transformation
    previous_transformation_ = transformation_;

    // Solve for decent direction using newton method, line 23 in Algorithm 2 [Magnusson
    // 2009]
    Eigen::JacobiSVD<Eigen::Matrix<double, 6, 6>> sv(
        hessian, Eigen::ComputeFullU | Eigen::ComputeFullV);
    // Negative for maximization as opposed to minimization
    Eigen::Matrix<double, 6, 1> delta = sv.solve(-score_gradient);

    // Calculate step length with guaranteed sufficient decrease [More, Thuente 1994]
    double delta_norm = delta.norm();

    if (delta_norm == 0 || std::isnan(delta_norm)) {
      trans_likelihood_ = score / static_cast<double>(input_->size());
      converged_ = delta_norm == 0;
      return;
    }

    double stepsize_ = 2.0;
    double transformation_epsilon_ = 2.0;

    delta /= delta_norm;
    delta_norm = computeStepLengthMT(transform,
                                     delta,
                                     delta_norm,
                                     step_size_,
                                     transformation_epsilon_ / 2,
                                     score,
                                     score_gradient,
                                     hessian,
                                     output);
    delta *= delta_norm;

    // Convert delta into matrix form
    convertTransform(delta, transformation_);

    transform += delta;

    // Update Visualizer (untested)
    if (update_visualizer_)
      update_visualizer_(output, pcl::Indices(), *target_, pcl::Indices());

    const double cos_angle =
        0.5 * (transformation_.template block<3, 3>(0, 0).trace() - 1);
    const double translation_sqr =
        transformation_.template block<3, 1>(0, 3).squaredNorm();

    nr_iterations_++;

    if (nr_iterations_ >= max_iterations_ ||
        ((transformation_epsilon_ > 0 && translation_sqr <= transformation_epsilon_) &&
         (transformation_rotation_epsilon_ > 0 &&
          cos_angle >= transformation_rotation_epsilon_)) ||
        ((transformation_epsilon_ <= 0) &&
         (transformation_rotation_epsilon_ > 0 &&
          cos_angle >= transformation_rotation_epsilon_)) ||
        ((transformation_epsilon_ > 0 && translation_sqr <= transformation_epsilon_) &&
         (transformation_rotation_epsilon_ <= 0))) {
      converged_ = true;
    }
  }

  // Store transformation likelihood.  The relative differences within each scan
  // registration are accurate but the normalization constants need to be modified for
  // it to be globally accurate
  trans_likelihood_ = score / static_cast<double>(input_->size());
}