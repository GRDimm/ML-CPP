#include "L/LogisticRegression.hpp"
#include <Eigen/Dense>
#include <cmath>

namespace L {

// Constructor with threshold and optimize_threshold parameters
LogisticRegression::LogisticRegression(double threshold, bool optimize_threshold)
    : intercept_(0), threshold_(threshold), optimize_threshold_(optimize_threshold) {}

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
    intercept_ = theta(0);
    coefficients_ = theta.tail(X.cols());

    // Optimize threshold if required
    if (optimize_threshold_) {
        optimizeThreshold(X, y);
    }
}

Eigen::VectorXd LogisticRegression::predict_proba(const Eigen::MatrixXd& X) const {
    // Add a column of 1s to X for the intercept in predictions
    Eigen::MatrixXd X_b(X.rows(), X.cols() + 1);
    X_b << Eigen::VectorXd::Ones(X.rows()), X;

    // Combine intercept and coefficients into one vector
    Eigen::VectorXd theta(coefficients_.size() + 1);
    theta << intercept_, coefficients_;

    // Calculate predictions: y_pred = sigmoid(X_b * theta)
    Eigen::VectorXd linear_preds = X_b * theta;
    return linear_preds.unaryExpr([](double z) { return 1 / (1 + std::exp(-z)); });
}

Eigen::VectorXd LogisticRegression::predict(const Eigen::MatrixXd& X) const {
    // Get probability predictions
    Eigen::VectorXd probabilities = predict_proba(X);

    // Apply threshold to get binary predictions
    return probabilities.unaryExpr([this](double p) { return p >= threshold_ ? 1.0 : 0.0; });
}

void LogisticRegression::optimizeThreshold(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
    // Generate probabilities with current coefficients
    Eigen::VectorXd probabilities = predict_proba(X);

    // Find the threshold that maximizes F1 score
    double best_threshold = 0.5;
    double best_f1_score = 0.0;

    // Try thresholds from 0 to 1 in small increments (e.g., 0.01)
    for (double t = 0.0; t <= 1.0; t += 0.01) {
        // Generate binary predictions based on the threshold
        Eigen::VectorXd binary_predictions = probabilities.unaryExpr([t](double p) { return p >= t ? 1.0 : 0.0; });

        // Calculate precision and recall
        int true_positive = 0, false_positive = 0, false_negative = 0;
        for (int i = 0; i < y.size(); ++i) {
            if (binary_predictions(i) == 1.0 && y(i) == 1.0) ++true_positive;
            else if (binary_predictions(i) == 1.0 && y(i) == 0.0) ++false_positive;
            else if (binary_predictions(i) == 0.0 && y(i) == 1.0) ++false_negative;
        }

        double precision = (true_positive + false_positive) > 0 ? static_cast<double>(true_positive) / (true_positive + false_positive) : 0.0;
        double recall = (true_positive + false_negative) > 0 ? static_cast<double>(true_positive) / (true_positive + false_negative) : 0.0;
        double f1_score = (precision + recall) > 0 ? 2 * (precision * recall) / (precision + recall) : 0.0;

        // Update best threshold if current F1 is better
        if (f1_score > best_f1_score) {
            best_f1_score = f1_score;
            best_threshold = t;
        }
    }

    // Set the threshold to the best one found
    threshold_ = best_threshold;
}

Eigen::VectorXd LogisticRegression::coefficients() const {
    return coefficients_;
}

double LogisticRegression::intercept() const {
    return intercept_;
}

double LogisticRegression::threshold() const {
    return threshold_;
}

} // namespace L
