#ifndef L_REGRESSIONMETRICS_HPP
#define L_REGRESSIONMETRICS_HPP

#include <Eigen/Dense>

namespace L {

class RegressionMetrics {
public:
    // Constructor that takes predictions and actual values
    RegressionMetrics(const Eigen::VectorXd& predictions, const Eigen::VectorXd& y_true);

    double r2Score() const;             // R² Score
    double meanAbsoluteError() const;   // MAE
    double meanSquaredError() const;    // MSE
    double rootMeanSquaredError() const; // RMSE

private:
    Eigen::VectorXd predictions;
    Eigen::VectorXd y_true;

    double mean(const Eigen::VectorXd& values) const;
};

} // namespace L

#endif // L_REGRESSIONMETRICS_HPP
