#include "L/RegressionMetrics.hpp"
#include <stdexcept>
#include <cmath>

namespace L {

RegressionMetrics::RegressionMetrics(const Eigen::VectorXd& predictions, const Eigen::VectorXd& y_true)
    : predictions(predictions), y_true(y_true) {
    if (predictions.size() != y_true.size()) {
        throw std::invalid_argument("Predictions and actual values must have the same length.");
    }
}

double RegressionMetrics::mean(const Eigen::VectorXd& values) const {
    return values.mean();
}

double RegressionMetrics::r2Score() const {
    double y_mean = mean(y_true);
    double ss_tot = (y_true.array() - y_mean).square().sum();
    double ss_res = (y_true - predictions).array().square().sum();

    return 1 - (ss_res / ss_tot);
}

double RegressionMetrics::meanAbsoluteError() const {
    return (y_true - predictions).array().abs().mean();
}

double RegressionMetrics::meanSquaredError() const {
    return (y_true - predictions).array().square().mean();
}

double RegressionMetrics::rootMeanSquaredError() const {
    return std::sqrt(meanSquaredError());
}

} // namespace L
