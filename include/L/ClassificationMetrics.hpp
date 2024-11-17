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
    double accuracy() const;
    double precision(int class_label = 1) const;      // Default for binary (positive class = 1)
    double recall(int class_label = 1) const;         // Default for binary (positive class = 1)
    double f1_score(int class_label = 1) const;       // Default for binary (positive class = 1)
    double roc_auc() const;                           // Placeholder for binary/multiclass ROC-AUC
    double pr_auc() const;                            // Placeholder for binary/multiclass PR-AUC
    double log_loss() const;                          // Logarithmic loss for probabilities

private:
    // Helper methods
    double mean(const Eigen::VectorXd& values) const;
    Eigen::MatrixXd confusion_matrix() const;         // For multiclass metrics
    double calculate_precision(int true_positive, int false_positive) const;
    double calculate_recall(int true_positive, int false_negative) const;
    double calculate_f1(double precision, double recall) const;

    // Attributes
    Eigen::VectorXd predictions_;
    Eigen::VectorXd y_true_;
    std::vector<int> classes_;                         // Vector to store class labels
};

} // namespace L

#endif // L_CLASSIFICATIONMETRICS_HPP
