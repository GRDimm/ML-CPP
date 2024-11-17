#include "L/ClassificationMetrics.hpp"

namespace L {

// Constructor
ClassificationMetrics::ClassificationMetrics(const Eigen::VectorXd& predictions, const Eigen::VectorXd& y_true, const std::vector<int>& classes)
    : predictions_(predictions), y_true_(y_true), classes_(classes) {}

// Accuracy
double ClassificationMetrics::accuracy() const {
    int correct = 0;
    for (int i = 0; i < predictions_.size(); ++i) {
        if (predictions_(i) == y_true_(i)) {
            ++correct;
        }
    }
    return static_cast<double>(correct) / predictions_.size();
}

// Precision for a specific class
double ClassificationMetrics::precision(int class_label) const {
    int true_positive = 0;
    int false_positive = 0;

    for (int i = 0; i < predictions_.size(); ++i) {
        if (predictions_(i) == class_label) {
            if (y_true_(i) == class_label) {
                ++true_positive;
            } else {
                ++false_positive;
            }
        }
    }

    return (true_positive + false_positive) > 0 ? static_cast<double>(true_positive) / (true_positive + false_positive) : 0.0;
}

// Recall for a specific class
double ClassificationMetrics::recall(int class_label) const {
    int true_positive = 0;
    int false_negative = 0;

    for (int i = 0; i < y_true_.size(); ++i) {
        if (y_true_(i) == class_label) {
            if (predictions_(i) == class_label) {
                ++true_positive;
            } else {
                ++false_negative;
            }
        }
    }

    return (true_positive + false_negative) > 0 ? static_cast<double>(true_positive) / (true_positive + false_negative) : 0.0;
}

// F1 Score for a specific class
double ClassificationMetrics::f1_score(int class_label) const {
    double prec = precision(class_label);
    double rec = recall(class_label);
    return (prec + rec) > 0 ? 2 * (prec * rec) / (prec + rec) : 0.0;
}

} // namespace L
