/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2010-2012, Willow Garage, Inc.
 *  Copyright (c) 2012-, Open Perception, Inc.
 *
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the copyright holder(s) nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 * $Id$
 *
 */

#pragma once

#include <pcl/common/utils.h>
#include <pcl/filters/voxel_grid_covariance.h>
#include <pcl/registration/registration.h>
#include <memory.h>
#include <pcl/pcl_macros.h>
#include <pcl/PointIndices.h>

#include <unsupported/Eigen/NonLinearOptimization>

namespace pcl {
  // using pcl::Registration;
  // using pcl::VoxelGridCovariance;
  // using Ptr = std::shared_ptr<NormalDistributionsTransform<PointSource, PointTarget, Scalar>>;
  // using ConstPtr = std::shared_ptr<const NormalDistributionsTransform<PointSource, PointTarget, Scalar>>;
  namespace custom {
  /** \brief A 3D Normal Distribution Transform registration implementation for
   * point cloud data.
   * \note For more information please see <b>Magnusson, M. (2009). The
   * Three-Dimensional Normal-Distributions Transform — an Efﬁcient Representation
   * for Registration, Surface Analysis, and Loop Detection. PhD thesis, Orebro
   * University.  Orebro Studies in Technology 36.</b>, <b>More, J., and Thuente,
   * D. (1994). Line Search Algorithm with Guaranteed Sufficient Decrease In ACM
   * Transactions on Mathematical Software.</b> and Sun, W. and Yuan, Y, (2006)
   * Optimization Theory and Methods: Nonlinear Programming. 89-100
   * \note Math refactored by Todor Stoyanov.
   * \author Brian Okorn (Space and Naval Warfare Systems Center Pacific)
   * \ingroup registration
   */
  template <typename PointSource, typename PointTarget, typename Scalar = float>
  class NormalDistributionsTransform
  : public Registration<PointSource, PointTarget, Scalar> {
  protected:
    using PointCloudSource =
        typename Registration<PointSource, PointTarget, Scalar>::PointCloudSource;
    using PointCloudSourcePtr = typename PointCloudSource::Ptr;
    using PointCloudSourceConstPtr = typename PointCloudSource::ConstPtr;

    using PointCloudTarget =
        typename Registration<PointSource, PointTarget, Scalar>::PointCloudTarget;
    using PointCloudTargetPtr = typename PointCloudTarget::Ptr;
    using PointCloudTargetConstPtr = typename PointCloudTarget::ConstPtr;

    using PointIndicesPtr = pcl::PointIndices::Ptr;
    using PointIndicesConstPtr = pcl::PointIndices::ConstPtr;

    /** \brief Typename of searchable voxel grid containing mean and
     * covariance. */
    using TargetGrid = VoxelGridCovariance<PointTarget>;
    /** \brief Typename of pointer to searchable voxel grid. */
    using TargetGridPtr = TargetGrid*;
    /** \brief Typename of const pointer to searchable voxel grid. */
    using TargetGridConstPtr = const TargetGrid*;
    /** \brief Typename of const pointer to searchable voxel grid leaf. */
    using TargetGridLeafConstPtr = typename TargetGrid::LeafConstPtr;

  public:
    using Ptr =
        shared_ptr<NormalDistributionsTransform<PointSource, PointTarget, Scalar>>;
    using ConstPtr =
        shared_ptr<const NormalDistributionsTransform<PointSource, PointTarget, Scalar>>;
    using Vector3 = typename Eigen::Matrix<Scalar, 3, 1>;
    using Matrix4 = typename Registration<PointSource, PointTarget, Scalar>::Matrix4;
    using Affine3 = typename Eigen::Transform<Scalar, 3, Eigen::Affine>;

      virtual void
    computeTransformation(PointCloudSource& output)
    {
      computeTransformation(output, Matrix4::Identity());
    }

    /** \brief Estimate the transformation and returns the transformed source
     * (input) as output.
     * \param[out] output the resultant input transformed point cloud dataset
     * \param[in] guess the initial gross estimation of the transformation
     */
    void
    computeTransformation(PointCloudSource& output, const Matrix4& guess) override;

    

    /** \brief Constructor.  Sets \ref outlier_ratio_ to 0.55, \ref step_size_ to
     * 0.1 and \ref resolution_ to 1.0
     */
    NormalDistributionsTransform();

    /** \brief Empty destructor */
    ~NormalDistributionsTransform() override = default;

    /** \brief Provide a pointer to the input target (e.g., the point cloud that
     * we want to align the input source to).
     * \param[in] cloud the input point cloud target
     */
    inline void
    setInputTarget(const PointCloudTargetConstPtr& cloud) override
    {
      Registration<PointSource, PointTarget, Scalar>::setInputTarget(cloud);
      init();
    }

    /** \brief Set/change the voxel grid resolution.
     * \param[in] resolution side length of voxels
     */
    inline void
    setResolution(float resolution)
    {
      // Prevents unnecessary voxel initiations
      if (resolution_ != resolution) {
        resolution_ = resolution;
        if (input_) {
          init();
        }
      }
    }

    /** \brief Set the minimum number of points required for a cell to be used (must be 3
     * or greater for covariance calculation). Calls the function of the underlying
     * VoxelGridCovariance. This function must be called before `setInputTarget` and
     * `setResolution`. \param[in] min_points_per_voxel the minimum number of points
     * required for a voxel to be used
     */
    inline void
    setMinPointPerVoxel(unsigned int min_points_per_voxel)
    {
      target_cells_.setMinPointPerVoxel(min_points_per_voxel);
    }

    /** \brief Get voxel grid resolution.
     * \return side length of voxels
     */
    inline float
    getResolution() const
    {
      return resolution_;
    }

    /** \brief Get the newton line search maximum step length.
     * \return maximum step length
     */
    inline double
    getStepSize() const
    {
      return step_size_;
    }

    /** \brief Set/change the newton line search maximum step length.
     * \param[in] step_size maximum step length
     */
    inline void
    setStepSize(double step_size)
    {
      step_size_ = step_size;
    }

    /** \brief Get the point cloud outlier ratio.
     * \return outlier ratio
     */
    inline double
    getOutlierRatio() const
    {
      return outlier_ratio_;
    }

    /** \brief Get the point cloud outlier ratio.
     * \return outlier ratio
     */
    PCL_DEPRECATED(1,
                  18,
                  "The method `getOulierRatio` has been renamed to "
                  "`getOutlierRatio`.")
    inline double
    getOulierRatio() const
    {
      return outlier_ratio_;
    }

    /** \brief Set/change the point cloud outlier ratio.
     * \param[in] outlier_ratio outlier ratio
     */
    inline void
    setOutlierRatio(double outlier_ratio)
    {
      outlier_ratio_ = outlier_ratio;
    }

    PCL_DEPRECATED(1,
                  18,
                  "The method `setOulierRatio` has been renamed to "
                  "`setOutlierRatio`.")
    /** \brief Set/change the point cloud outlier ratio.
     * \param[in] outlier_ratio outlier ratio
     */
    inline void
    setOulierRatio(double outlier_ratio)
    {
      outlier_ratio_ = outlier_ratio;
    }

    /** \brief Get the registration alignment likelihood.
     * \return transformation likelihood
     */
    inline double
    getTransformationLikelihood() const
    {
      return trans_likelihood_;
    }

    /** \brief Get the registration alignment probability.
     * \return transformation probability
     */
    PCL_DEPRECATED(1,
                  16,
                  "The method `getTransformationProbability` has been renamed to "
                  "`getTransformationLikelihood`.")
    inline double
    getTransformationProbability() const
    {
      return trans_likelihood_;
    }

    /** \brief Get the number of iterations required to calculate alignment.
     * \return final number of iterations
     */
    inline int
    getFinalNumIteration() const
    {
      return nr_iterations_;
    }

    /** \brief Get access to the `VoxelGridCovariance` generated from target cloud
     * containing point means and covariances. Set the input target cloud before calling
     * this. Useful for debugging, e.g.
     * \code
     * pcl::PointCloud<PointXYZ> visualize_cloud;
     * ndt.getTargetCells().getDisplayCloud(visualize_cloud);
     * \endcode
     */
    inline const TargetGrid&
    getTargetCells() const
    {
      return target_cells_;
    }

    /** \brief Convert 6 element transformation vector to affine transformation.
     * \param[in] x transformation vector of the form [x, y, z, roll, pitch, yaw]
     * \param[out] trans affine transform corresponding to given transformation
     * vector
     */
    static void
    convertTransform(const Eigen::Matrix<double, 6, 1>& x, Affine3& trans)
    {
      trans = Eigen::Translation<Scalar, 3>(x.head<3>().cast<Scalar>()) *
              Eigen::AngleAxis<Scalar>(static_cast<Scalar>(x(3)), Vector3::UnitX()) *
              Eigen::AngleAxis<Scalar>(static_cast<Scalar>(x(4)), Vector3::UnitY()) *
              Eigen::AngleAxis<Scalar>(static_cast<Scalar>(x(5)), Vector3::UnitZ());
    }

    /** \brief Convert 6 element transformation vector to transformation matrix.
     * \param[in] x transformation vector of the form [x, y, z, roll, pitch, yaw]
     * \param[out] trans 4x4 transformation matrix corresponding to given
     * transformation vector
     */
    static void
    convertTransform(const Eigen::Matrix<double, 6, 1>& x, Matrix4& trans)
    {
      Affine3 _affine;
      convertTransform(x, _affine);
      trans = _affine.matrix();
    }

  protected:
    using Registration<PointSource, PointTarget, Scalar>::reg_name_;
    using Registration<PointSource, PointTarget, Scalar>::getClassName;
    using Registration<PointSource, PointTarget, Scalar>::input_;
    using Registration<PointSource, PointTarget, Scalar>::indices_;
    using Registration<PointSource, PointTarget, Scalar>::target_;
    using Registration<PointSource, PointTarget, Scalar>::nr_iterations_;
    using Registration<PointSource, PointTarget, Scalar>::max_iterations_;
    using Registration<PointSource, PointTarget, Scalar>::previous_transformation_;
    using Registration<PointSource, PointTarget, Scalar>::final_transformation_;
    using Registration<PointSource, PointTarget, Scalar>::transformation_;
    using Registration<PointSource, PointTarget, Scalar>::transformation_epsilon_;
    using Registration<PointSource, PointTarget, Scalar>::
        transformation_rotation_epsilon_;
    using Registration<PointSource, PointTarget, Scalar>::converged_;
    using Registration<PointSource, PointTarget, Scalar>::corr_dist_threshold_;
    using Registration<PointSource, PointTarget, Scalar>::inlier_threshold_;

    using Registration<PointSource, PointTarget, Scalar>::update_visualizer_;

    /** \brief Estimate the transformation and returns the transformed source
     * (input) as output.
     * \param[out] output the resultant input transformed point cloud dataset
     */

    // 이부분 public으로 옮김
    // virtual void
    // computeTransformation(PointCloudSource& output)
    // {
    //   computeTransformation(output, Matrix4::Identity());
    // }

    // /** \brief Estimate the transformation and returns the transformed source
    //  * (input) as output.
    //  * \param[out] output the resultant input transformed point cloud dataset
    //  * \param[in] guess the initial gross estimation of the transformation
    //  */
    // void
    // computeTransformation(PointCloudSource& output, const Matrix4& guess) override;

    /** \brief Initiate covariance voxel structure. */
    void inline init()
    {
      target_cells_.setLeafSize(resolution_, resolution_, resolution_);
      target_cells_.setInputCloud(target_);
      // Initiate voxel structure.
      target_cells_.filter(true);
      PCL_DEBUG("[pcl::%s::init] Computed voxel structure, got %zu voxels with valid "
                "normal distributions.\n",
                getClassName().c_str(),
                target_cells_.getCentroids()->size());
    }

    /** \brief Compute derivatives of likelihood function w.r.t. the
     * transformation vector.
     * \note Equation 6.10, 6.12 and 6.13 [Magnusson 2009].
     * \param[out] score_gradient the gradient vector of the likelihood function
     * w.r.t.  the transformation vector
     * \param[out] hessian the hessian matrix of the likelihood function
     * w.r.t. the transformation vector
     * \param[in] trans_cloud transformed point cloud
     * \param[in] transform the current transform vector
     * \param[in] compute_hessian flag to calculate hessian, unnecessary for step
     * calculation.
     */
    double
    computeDerivatives(Eigen::Matrix<double, 6, 1>& score_gradient,
                      Eigen::Matrix<double, 6, 6>& hessian,
                      const PointCloudSource& trans_cloud,
                      const Eigen::Matrix<double, 6, 1>& transform,
                      bool compute_hessian = true);

    /** \brief Compute individual point contributions to derivatives of
     * likelihood function w.r.t. the transformation vector.
     * \note Equation 6.10, 6.12 and 6.13 [Magnusson 2009].
     * \param[in,out] score_gradient the gradient vector of the likelihood
     * function w.r.t. the transformation vector
     * \param[in,out] hessian the hessian matrix of the likelihood function
     * w.r.t. the transformation vector
     * \param[in] x_trans transformed point minus mean of occupied covariance
     * voxel
     * \param[in] c_inv covariance of occupied covariance voxel
     * \param[in] compute_hessian flag to calculate hessian, unnecessary for step
     * calculation.
     */
    double
    updateDerivatives(Eigen::Matrix<double, 6, 1>& score_gradient,
                      Eigen::Matrix<double, 6, 6>& hessian,
                      const Eigen::Vector3d& x_trans,
                      const Eigen::Matrix3d& c_inv,
                      bool compute_hessian = true) const;

    /** \brief Precompute angular components of derivatives.
     * \note Equation 6.19 and 6.21 [Magnusson 2009].
     * \param[in] transform the current transform vector
     * \param[in] compute_hessian flag to calculate hessian, unnecessary for step
     * calculation.
     */
    void
    computeAngleDerivatives(const Eigen::Matrix<double, 6, 1>& transform,
                            bool compute_hessian = true);

    /** \brief Compute point derivatives.
     * \note Equation 6.18-21 [Magnusson 2009].
     * \param[in] x point from the input cloud
     * \param[in] compute_hessian flag to calculate hessian, unnecessary for step
     * calculation.
     */
    void
    computePointDerivatives(const Eigen::Vector3d& x, bool compute_hessian = true);

    /** \brief Compute hessian of likelihood function w.r.t. the transformation
     * vector.
     * \note Equation 6.13 [Magnusson 2009].
     * \param[out] hessian the hessian matrix of the likelihood function
     * w.r.t. the transformation vector
     * \param[in] trans_cloud transformed point cloud
     */
    void
    computeHessian(Eigen::Matrix<double, 6, 6>& hessian,
                  const PointCloudSource& trans_cloud);

    /** \brief Compute individual point contributions to hessian of likelihood
     * function w.r.t. the transformation vector.
     * \note Equation 6.13 [Magnusson 2009].
     * \param[in,out] hessian the hessian matrix of the likelihood function
     * w.r.t. the transformation vector
     * \param[in] x_trans transformed point minus mean of occupied covariance
     * voxel
     * \param[in] c_inv covariance of occupied covariance voxel
     */
    void
    updateHessian(Eigen::Matrix<double, 6, 6>& hessian,
                  const Eigen::Vector3d& x_trans,
                  const Eigen::Matrix3d& c_inv) const;

    /** \brief Compute line search step length and update transform and
     * likelihood derivatives using More-Thuente method.
     * \note Search Algorithm [More, Thuente 1994]
     * \param[in] transform initial transformation vector, \f$ x \f$ in Equation
     * 1.3 (Moore, Thuente 1994) and \f$ \vec{p} \f$ in Algorithm 2 [Magnusson
     * 2009]
     * \param[in] step_dir descent direction, \f$ p \f$ in Equation 1.3 (Moore,
     * Thuente 1994) and \f$ \delta \vec{p} \f$ normalized in Algorithm 2
     * [Magnusson 2009]
     * \param[in] step_init initial step length estimate, \f$ \alpha_0 \f$ in
     * Moore-Thuente (1994) and the normal of \f$ \delta \vec{p} \f$ in Algorithm
     * 2 [Magnusson 2009]
     * \param[in] step_max maximum step length, \f$ \alpha_max \f$ in
     * Moore-Thuente (1994)
     * \param[in] step_min minimum step length, \f$ \alpha_min \f$ in
     * Moore-Thuente (1994)
     * \param[out] score final score function value, \f$ f(x + \alpha p) \f$ in
     * Equation 1.3 (Moore, Thuente 1994) and \f$ score \f$ in Algorithm 2
     * [Magnusson 2009]
     * \param[in,out] score_gradient gradient of score function w.r.t.
     * transformation vector, \f$ f'(x + \alpha p) \f$ in Moore-Thuente (1994) and
     * \f$ \vec{g} \f$ in Algorithm 2 [Magnusson 2009]
     * \param[out] hessian hessian of score function w.r.t. transformation vector,
     * \f$ f''(x + \alpha p) \f$ in Moore-Thuente (1994) and \f$ H \f$ in
     * Algorithm 2 [Magnusson 2009]
     * \param[in,out] trans_cloud transformed point cloud, \f$ X \f$ transformed
     * by \f$ T(\vec{p},\vec{x}) \f$ in Algorithm 2 [Magnusson 2009]
     * \return final step length
     */
    double
    computeStepLengthMT(const Eigen::Matrix<double, 6, 1>& transform,
                        Eigen::Matrix<double, 6, 1>& step_dir,
                        double step_init,
                        double step_max,
                        double step_min,
                        double& score,
                        Eigen::Matrix<double, 6, 1>& score_gradient,
                        Eigen::Matrix<double, 6, 6>& hessian,
                        PointCloudSource& trans_cloud);

    /** \brief Update interval of possible step lengths for More-Thuente method,
     * \f$ I \f$ in More-Thuente (1994)
     * \note Updating Algorithm until some value satisfies \f$ \psi(\alpha_k) \leq
     * 0 \f$ and \f$ \phi'(\alpha_k) \geq 0 \f$ and Modified Updating Algorithm
     * from then on [More, Thuente 1994].
     * \param[in,out] a_l first endpoint of interval \f$ I \f$, \f$ \alpha_l \f$
     * in Moore-Thuente (1994)
     * \param[in,out] f_l value at first endpoint, \f$ f_l \f$ in Moore-Thuente
     * (1994), \f$ \psi(\alpha_l) \f$ for Update Algorithm and \f$ \phi(\alpha_l)
     * \f$ for Modified Update Algorithm
     * \param[in,out] g_l derivative at first endpoint, \f$ g_l \f$ in
     * Moore-Thuente (1994), \f$ \psi'(\alpha_l) \f$ for Update Algorithm and \f$
     * \phi'(\alpha_l) \f$ for Modified Update Algorithm
     * \param[in,out] a_u second endpoint of interval \f$ I \f$, \f$ \alpha_u \f$
     * in Moore-Thuente (1994)
     * \param[in,out] f_u value at second endpoint, \f$ f_u \f$ in Moore-Thuente
     * (1994), \f$ \psi(\alpha_u) \f$ for Update Algorithm and \f$ \phi(\alpha_u)
     * \f$ for Modified Update Algorithm
     * \param[in,out] g_u derivative at second endpoint, \f$ g_u \f$ in
     * Moore-Thuente (1994), \f$ \psi'(\alpha_u) \f$ for Update Algorithm and \f$
     * \phi'(\alpha_u) \f$ for Modified Update Algorithm
     * \param[in] a_t trial value, \f$ \alpha_t \f$ in Moore-Thuente (1994)
     * \param[in] f_t value at trial value, \f$ f_t \f$ in Moore-Thuente (1994),
     * \f$ \psi(\alpha_t) \f$ for Update Algorithm and \f$ \phi(\alpha_t) \f$ for
     * Modified Update Algorithm
     * \param[in] g_t derivative at trial value, \f$ g_t \f$ in Moore-Thuente
     * (1994), \f$ \psi'(\alpha_t) \f$ for Update Algorithm and \f$
     * \phi'(\alpha_t) \f$ for Modified Update Algorithm
     * \return if interval converges
     */
    bool
    updateIntervalMT(double& a_l,
                    double& f_l,
                    double& g_l,
                    double& a_u,
                    double& f_u,
                    double& g_u,
                    double a_t,
                    double f_t,
                    double g_t) const;

    /** \brief Select new trial value for More-Thuente method.
     * \note Trial Value Selection [More, Thuente 1994], \f$ \psi(\alpha_k) \f$ is
     * used for \f$ f_k \f$ and \f$ g_k \f$ until some value satisfies the test
     * \f$ \psi(\alpha_k) \leq 0 \f$ and \f$ \phi'(\alpha_k) \geq 0 \f$ then \f$
     * \phi(\alpha_k) \f$ is used from then on.
     * \note Interpolation Minimizer equations from Optimization Theory and
     * Methods: Nonlinear Programming By Wenyu Sun, Ya-xiang Yuan (89-100).
     * \param[in] a_l first endpoint of interval \f$ I \f$, \f$ \alpha_l \f$ in
     * Moore-Thuente (1994)
     * \param[in] f_l value at first endpoint, \f$ f_l \f$ in Moore-Thuente (1994)
     * \param[in] g_l derivative at first endpoint, \f$ g_l \f$ in Moore-Thuente
     * (1994)
     * \param[in] a_u second endpoint of interval \f$ I \f$, \f$ \alpha_u \f$ in
     * Moore-Thuente (1994)
     * \param[in] f_u value at second endpoint, \f$ f_u \f$ in Moore-Thuente
     * (1994)
     * \param[in] g_u derivative at second endpoint, \f$ g_u \f$ in Moore-Thuente
     * (1994)
     * \param[in] a_t previous trial value, \f$ \alpha_t \f$ in Moore-Thuente
     * (1994)
     * \param[in] f_t value at previous trial value, \f$ f_t \f$ in Moore-Thuente
     * (1994)
     * \param[in] g_t derivative at previous trial value, \f$ g_t \f$ in
     * Moore-Thuente (1994)
     * \return new trial value
     */
    double
    trialValueSelectionMT(double a_l,
                          double f_l,
                          double g_l,
                          double a_u,
                          double f_u,
                          double g_u,
                          double a_t,
                          double f_t,
                          double g_t) const;

    /** \brief Auxiliary function used to determine endpoints of More-Thuente
     * interval.
     * \note \f$ \psi(\alpha) \f$ in Equation 1.6 (Moore, Thuente 1994)
     * \param[in] a the step length, \f$ \alpha \f$ in More-Thuente (1994)
     * \param[in] f_a function value at step length a, \f$ \phi(\alpha) \f$ in
     * More-Thuente (1994)
     * \param[in] f_0 initial function value, \f$ \phi(0) \f$ in Moore-Thuente
     * (1994)
     * \param[in] g_0 initial function gradient, \f$ \phi'(0) \f$ in More-Thuente
     * (1994)
     * \param[in] mu the step length, constant \f$ \mu \f$ in Equation 1.1 [More,
     * Thuente 1994]
     * \return sufficient decrease value
     */
    inline double
    auxilaryFunction_PsiMT(
        double a, double f_a, double f_0, double g_0, double mu = 1.e-4) const
    {
      return f_a - f_0 - mu * g_0 * a;
    }

    /** \brief Auxiliary function derivative used to determine endpoints of
     * More-Thuente interval.
     * \note \f$ \psi'(\alpha) \f$, derivative of Equation 1.6 (Moore, Thuente
     * 1994)
     * \param[in] g_a function gradient at step length a, \f$ \phi'(\alpha) \f$ in
     * More-Thuente (1994)
     * \param[in] g_0 initial function gradient, \f$ \phi'(0) \f$ in More-Thuente
     * (1994)
     * \param[in] mu the step length, constant \f$ \mu \f$ in Equation 1.1 [More,
     * Thuente 1994]
     * \return sufficient decrease derivative
     */
    inline double
    auxilaryFunction_dPsiMT(double g_a, double g_0, double mu = 1.e-4) const
    {
      return g_a - mu * g_0;
    }

    /** \brief The voxel grid generated from target cloud containing point means
     * and covariances. */
    TargetGrid target_cells_;

    /** \brief The side length of voxels. */
    float resolution_{1.0f};

    /** \brief The maximum step length. */
    double step_size_{0.1};

    /** \brief The ratio of outliers of points w.r.t. a normal distribution,
     * Equation 6.7 [Magnusson 2009]. */
    double outlier_ratio_{0.55};

    /** \brief The normalization constants used fit the point distribution to a
     * normal distribution, Equation 6.8 [Magnusson 2009]. */
    double gauss_d1_{0.0}, gauss_d2_{0.0};

    /** \brief The likelihood score of the transform applied to the input cloud,
     * Equation 6.9 and 6.10 [Magnusson 2009]. */
    union {
      PCL_DEPRECATED(1,
                    16,
                    "`trans_probability_` has been renamed to `trans_likelihood_`.")
      double trans_probability_;
      double trans_likelihood_{0.0};
    };

    /** \brief Precomputed Angular Gradient
     *
     * The precomputed angular derivatives for the jacobian of a transformation
     * vector, Equation 6.19 [Magnusson 2009].
     */
    Eigen::Matrix<double, 8, 4> angular_jacobian_;

    /** \brief Precomputed Angular Hessian
     *
     * The precomputed angular derivatives for the hessian of a transformation
     * vector, Equation 6.19 [Magnusson 2009].
     */
    Eigen::Matrix<double, 15, 4> angular_hessian_;

    /** \brief The first order derivative of the transformation of a point
     * w.r.t. the transform vector, \f$ J_E \f$ in Equation 6.18 [Magnusson
     * 2009]. */
    Eigen::Matrix<double, 3, 6> point_jacobian_;

    /** \brief The second order derivative of the transformation of a point
     * w.r.t. the transform vector, \f$ H_E \f$ in Equation 6.20 [Magnusson
     * 2009]. */
    Eigen::Matrix<double, 18, 6> point_hessian_;

  public:
    PCL_MAKE_ALIGNED_OPERATOR_NEW
  };
  }; // namespace custom
} // namespace pcl

#include <pcl/registration/impl/ndt.hpp>
