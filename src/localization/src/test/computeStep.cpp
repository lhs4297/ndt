

inline double
  auxilaryFunction_PsiMT(
      double a, double f_a, double f_0, double g_0, double mu = 1.e-4) const
  {
    return f_a - f_0 - mu * g_0 * a;
  }

inline double
  auxilaryFunction_dPsiMT(double g_a, double g_0, double mu = 1.e-4) const
  {
    return g_a - mu * g_0;
  }


template <typename PointSource, typename PointTarget, typename Scalar>
double
NormalDistributionsTransform<PointSource, PointTarget, Scalar>::computeStepLengthMT(
    const Eigen::Matrix<double, 6, 1>& x,
    Eigen::Matrix<double, 6, 1>& step_dir,
    double step_init,
    double step_max,
    double step_min,
    double& score,
    Eigen::Matrix<double, 6, 1>& score_gradient,
    Eigen::Matrix<double, 6, 6>& hessian,
    PointCloudSource& trans_cloud)
{
  // Set the value of phi(0), Equation 1.3 [More, Thuente 1994]
  const double phi_0 = -score;
  // Set the value of phi'(0), Equation 1.3 [More, Thuente 1994]
  double d_phi_0 = -(score_gradient.dot(step_dir));

  if (d_phi_0 >= 0) {
    // Not a decent direction
    if (d_phi_0 == 0) {
      return 0;
    }
    // Reverse step direction and calculate optimal step.
    d_phi_0 *= -1;
    step_dir *= -1;
  }

  // The Search Algorithm for T(mu) [More, Thuente 1994]

  constexpr int max_step_iterations = 10;
  int step_iterations = 0;

  // Sufficient decrease constant, Equation 1.1 [More, Thuete 1994]
  constexpr double mu = 1.e-4;
  // Curvature condition constant, Equation 1.2 [More, Thuete 1994]
  constexpr double nu = 0.9;

  // Initial endpoints of Interval I,
  double a_l = 0, a_u = 0;

  // Auxiliary function psi is used until I is determined ot be a closed interval,
  // Equation 2.1 [More, Thuente 1994]
  double f_l = auxilaryFunction_PsiMT(a_l, phi_0, phi_0, d_phi_0, mu);
  double g_l = auxilaryFunction_dPsiMT(d_phi_0, d_phi_0, mu);

  double f_u = auxilaryFunction_PsiMT(a_u, phi_0, phi_0, d_phi_0, mu);
  double g_u = auxilaryFunction_dPsiMT(d_phi_0, d_phi_0, mu);

  // Check used to allow More-Thuente step length calculation to be skipped by making
  // step_min == step_max
  bool interval_converged = (step_max - step_min) < 0, open_interval = true;

  double a_t = step_init;
  a_t = std::min(a_t, step_max);
  a_t = std::max(a_t, step_min);

  Eigen::Matrix<double, 6, 1> x_t = x + step_dir * a_t;

  // Convert x_t into matrix form
  convertTransform(x_t, final_transformation_);

  // New transformed point cloud
  transformPointCloud(*input_, trans_cloud, final_transformation_);

  // Updates score, gradient and hessian.  Hessian calculation is unnecessary but
  // testing showed that most step calculations use the initial step suggestion and
  // recalculation the reusable portions of the hessian would entail more computation
  // time.
  score = computeDerivatives(score_gradient, hessian, trans_cloud, x_t, true);

  // Calculate phi(alpha_t)
  double phi_t = -score;
  // Calculate phi'(alpha_t)
  double d_phi_t = -(score_gradient.dot(step_dir));

  // Calculate psi(alpha_t)
  double psi_t = auxilaryFunction_PsiMT(a_t, phi_t, phi_0, d_phi_0, mu);
  // Calculate psi'(alpha_t)
  double d_psi_t = auxilaryFunction_dPsiMT(d_phi_t, d_phi_0, mu);

  // Iterate until max number of iterations, interval convergence or a value satisfies
  // the sufficient decrease, Equation 1.1, and curvature condition, Equation 1.2 [More,
  // Thuente 1994]
  while (!interval_converged && step_iterations < max_step_iterations &&
         (psi_t > 0 /*Sufficient Decrease*/ ||
          d_phi_t > -nu * d_phi_0 /*Curvature Condition*/)) {
    // Use auxiliary function if interval I is not closed
    if (open_interval) {
      a_t = trialValueSelectionMT(a_l, f_l, g_l, a_u, f_u, g_u, a_t, psi_t, d_psi_t);
    }
    else {
      a_t = trialValueSelectionMT(a_l, f_l, g_l, a_u, f_u, g_u, a_t, phi_t, d_phi_t);
    }

    a_t = std::min(a_t, step_max);
    a_t = std::max(a_t, step_min);

    x_t = x + step_dir * a_t;

    // Convert x_t into matrix form
    convertTransform(x_t, final_transformation_);

    // New transformed point cloud
    // Done on final cloud to prevent wasted computation
    transformPointCloud(*input_, trans_cloud, final_transformation_);

    // Updates score, gradient. Values stored to prevent wasted computation.
    score = computeDerivatives(score_gradient, hessian, trans_cloud, x_t, false);

    // Calculate phi(alpha_t+)
    phi_t = -score;
    // Calculate phi'(alpha_t+)
    d_phi_t = -(score_gradient.dot(step_dir));

    // Calculate psi(alpha_t+)
    psi_t = auxilaryFunction_PsiMT(a_t, phi_t, phi_0, d_phi_0, mu);
    // Calculate psi'(alpha_t+)
    d_psi_t = auxilaryFunction_dPsiMT(d_phi_t, d_phi_0, mu);

    // Check if I is now a closed interval
    if (open_interval && (psi_t <= 0 && d_psi_t >= 0)) {
      open_interval = false;

      // Converts f_l and g_l from psi to phi
      f_l += phi_0 - mu * d_phi_0 * a_l;
      g_l += mu * d_phi_0;

      // Converts f_u and g_u from psi to phi
      f_u += phi_0 - mu * d_phi_0 * a_u;
      g_u += mu * d_phi_0;
    }

    if (open_interval) {
      // Update interval end points using Updating Algorithm [More, Thuente 1994]
      interval_converged =
          updateIntervalMT(a_l, f_l, g_l, a_u, f_u, g_u, a_t, psi_t, d_psi_t);
    }
    else {
      // Update interval end points using Modified Updating Algorithm [More, Thuente
      // 1994]
      interval_converged =
          updateIntervalMT(a_l, f_l, g_l, a_u, f_u, g_u, a_t, phi_t, d_phi_t);
    }

    step_iterations++;
  }

  // If inner loop was run then hessian needs to be calculated.
  // Hessian is unnecessary for step length determination but gradients are required
  // so derivative and transform data is stored for the next iteration.
  if (step_iterations) {
    computeHessian(hessian, trans_cloud);
  }

  return a_t;
}
