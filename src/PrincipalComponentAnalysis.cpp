#include "L/PrincipalComponentAnalysis.hpp"
#include <Eigen/Eigenvalues> // For Eigenvalue decomposition
#include <stdexcept>         // For std::runtime_error

namespace L {

PrincipalComponentAnalysis::PrincipalComponentAnalysis(const Eigen::MatrixXd& X)
    : X_(X)
{
    // Center the data
    Eigen::VectorXd mean = X_.colwise().mean();
    X_.rowwise() -= mean.transpose();
}

void PrincipalComponentAnalysis::transform()
{
    // Compute the covariance matrix
    Eigen::MatrixXd covariance = (X_.transpose() * X_) / (X_.rows() - 1);

    // Perform eigenvalue decomposition
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen_solver(covariance);

    if (eigen_solver.info() != Eigen::Success) {
        throw std::runtime_error("Eigenvalue decomposition failed.");
    }

    // Eigenvalues are returned in increasing order; reverse for decreasing order
    eigen_values_ = eigen_solver.eigenvalues().reverse();
    eigen_vectors_ = eigen_solver.eigenvectors().rowwise().reverse();

    // Project the data onto the principal components
    principal_components_ = X_ * eigen_vectors_;
}

Eigen::MatrixXd PrincipalComponentAnalysis::principal_components(int n) const
{
    if (n <= 0 || n > principal_components_.cols()) {
        return principal_components_;
    } else {
        return principal_components_.leftCols(n);
    }
}

Eigen::MatrixXd PrincipalComponentAnalysis::eigen_vectors(int n) const
{
    if (n <= 0 || n > eigen_vectors_.cols()) {
        return eigen_vectors_;
    } else {
        return eigen_vectors_.leftCols(n);
    }
}

Eigen::VectorXd PrincipalComponentAnalysis::eigen_values(int n) const
{
    if (n <= 0 || n > eigen_values_.size()) {
        return eigen_values_;
    } else {
        return eigen_values_.head(n);
    }
}

} // namespace L
