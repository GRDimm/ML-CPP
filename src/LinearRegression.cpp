#include "L/LinearRegression.hpp"
#include <Eigen/Dense>

namespace L {

LinearRegression::LinearRegression() : intercept(0) {}

void LinearRegression::fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
    // Add a column of 1s to X for the intercept
    Eigen::MatrixXd X_b(X.rows(), X.cols() + 1);
    X_b << Eigen::VectorXd::Ones(X.rows()), X;

    // Calculate the least squares solution: theta = (X_b^T * X_b).inverse() * X_b^T * y
    Eigen::VectorXd theta = (X_b.transpose() * X_b).inverse() * X_b.transpose() * y;

    // Separate the intercept and coefficients
    intercept = theta(0);
    coefficients = theta.tail(X.cols());
}

Eigen::VectorXd LinearRegression::predict(const Eigen::MatrixXd& X) const {
    // Add a column of 1s to X for the intercept in predictions
    Eigen::MatrixXd X_b(X.rows(), X.cols() + 1);
    X_b << Eigen::VectorXd::Ones(X.rows()), X;

    // Calculate predictions: y_pred = X_b * theta
    Eigen::VectorXd theta(coefficients.size() + 1);
    theta << intercept, coefficients; // Combine intercept and coefficients into one vector

    return X_b * theta;
}

Eigen::VectorXd LinearRegression::getCoefficients() const {
    return coefficients;
}

double LinearRegression::getIntercept() const {
    return intercept;
}

} // namespace L
