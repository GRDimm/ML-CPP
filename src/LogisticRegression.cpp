#include "L/LogisticRegression.hpp"
#include <Eigen/Dense>
#include <cmath>

namespace L {

LogisticRegression::LogisticRegression() : intercept(0) {}

void LogisticRegression::fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y, double learning_rate, int iterations) {
    // Add a column of 1s to X for the intercept
    Eigen::MatrixXd X_b(X.rows(), X.cols() + 1);
    X_b << Eigen::VectorXd::Ones(X.rows()), X;

    // Initialize theta with zeros
    Eigen::VectorXd theta = Eigen::VectorXd::Zero(X_b.cols());

    // Gradient descent
    for (int i = 0; i < iterations; ++i) {
        // Calculate the predictions using the sigmoid function
        Eigen::VectorXd predictions = (X_b * theta).unaryExpr([](double z) { return 1 / (1 + std::exp(-z)); });

        // Calculate the gradient
        Eigen::VectorXd gradient = X_b.transpose() * (predictions - y) / X.rows();

        // Update theta
        theta -= learning_rate * gradient;
    }

    // Separate the intercept and coefficients
    intercept = theta(0);
    coefficients = theta.tail(X.cols());
}

Eigen::VectorXd LogisticRegression::predict(const Eigen::MatrixXd& X) const {
    // Add a column of 1s to X for the intercept in predictions
    Eigen::MatrixXd X_b(X.rows(), X.cols() + 1);
    X_b << Eigen::VectorXd::Ones(X.rows()), X;

    // Combine intercept and coefficients into one vector
    Eigen::VectorXd theta(coefficients.size() + 1);
    theta << intercept, coefficients;

    // Calculate predictions: y_pred = sigmoid(X_b * theta)
    Eigen::VectorXd linear_preds = X_b * theta;
    return linear_preds.unaryExpr([](double z) { return 1 / (1 + std::exp(-z)); });
}


Eigen::VectorXd LogisticRegression::getCoefficients() const {
    return coefficients;
}

double LogisticRegression::getIntercept() const {
    return intercept;
}

} // namespace L
