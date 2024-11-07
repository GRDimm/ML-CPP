#ifndef L_PRINCIPALCOMPONENTANALYSIS_HPP
#define L_PRINCIPALCOMPONENTANALYSIS_HPP

#include <Eigen/Dense>

namespace L {

class PrincipalComponentAnalysis {
public:
    PrincipalComponentAnalysis(const Eigen::MatrixXd& X);

    // Perform PCA
    void transform();

    // Get the first n principal components
    Eigen::MatrixXd principal_components(int n = 0) const;

    // Get the first n eigenvectors of the covariance matrix
    Eigen::MatrixXd eigen_vectors(int n = 0) const;

    // Get the first n eigenvalues of the covariance matrix
    Eigen::VectorXd eigen_values(int n = 0) const;

private:
    Eigen::MatrixXd X_;                   // Centered data matrix
    Eigen::MatrixXd principal_components_; // Projected data
    Eigen::MatrixXd eigen_vectors_;        // Eigenvectors (principal axes)
    Eigen::VectorXd eigen_values_;         // Eigenvalues
};

} // namespace L

#endif // L_PRINCIPALCOMPONENTANALYSIS_HPP
