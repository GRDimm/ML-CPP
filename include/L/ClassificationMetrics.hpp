#ifndef L_CLASSIFICATIONMETRICS_HPP
#define L_CLASSIFICATIONMETRICS_HPP

#include <Eigen/Dense>
#include <vector>

namespace L {

class ClassificationMetrics {
public:
    // Constructor that takes predictions, actual values, and an optional set of class labels
    ClassificationMetrics(const Eigen::VectorXd& predictions, const Eigen::VectorXd& y_true, const std::vector<int>& classes = {0, 1});

    // Metrics methods
    double accuracy();
    double precision(int class_label = 1);      // Default for binary (positive class = 1)
    double recall(int class_label = 1);         // Default for binary (positive class = 1)
    double f1_score(int class_label = 1);       // Default for binary (positive class = 1)
    Eigen::MatrixXd confusion_matrix() const;  

private:
    void check_computed(); 
    // Attributes
    Eigen::MatrixXd confusion_matrix_;
    Eigen::VectorXd predictions_;
    Eigen::VectorXd y_true_;
    std::vector<int> classes_;                         // Vector to store class labels
};

} // namespace L

#endif // L_CLASSIFICATIONMETRICS_HPP
