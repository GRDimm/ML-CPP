#ifndef L_LOGISTICREGRESSION_HPP
#define L_LOGISTICREGRESSION_HPP

#include <Eigen/Dense>

namespace L {

class LogisticRegression {
public:
    // Constructor with an optional threshold parameter
    LogisticRegression(double threshold = 0.5, bool optimize_threshold = false);

    void fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y, double learning_rate = 0.01, int iterations = 1000);
    Eigen::VectorXd predict(const Eigen::MatrixXd& X) const;         // Predictions using the set or optimized threshold
    Eigen::VectorXd predict_proba(const Eigen::MatrixXd& X) const;   // Returns probabilities without threshold application

    Eigen::VectorXd coefficients() const;  // Returns the coefficients (slopes for each feature)
    double intercept() const;              // Returns the intercept
    double threshold() const;              // Returns the threshold
private:
    void optimizeThreshold(const Eigen::MatrixXd& X, const Eigen::VectorXd& y); // Method to find the optimal threshold

    Eigen::VectorXd coefficients_; // Slopes for each feature
    double intercept_;             // Intercept
    double threshold_;             // Classification threshold, defaults to 0.5
    bool optimize_threshold_;      // Whether to find the optimal threshold during training
};

} // namespace L

#endif // L_LOGISTICREGRESSION_HPP
