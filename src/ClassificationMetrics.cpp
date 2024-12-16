#include "L/ClassificationMetrics.hpp"

namespace L {

// Constructor
ClassificationMetrics::ClassificationMetrics(const Eigen::VectorXd& predictions, const Eigen::VectorXd& y_true, const std::vector<int>& classes)
    : predictions_(predictions), y_true_(y_true), classes_(classes) {}



void ClassificationMetrics::check_computed(){
    if(confusion_matrix_.size() == 0){
        confusion_matrix_ = confusion_matrix();
    }
}

// Confusion matrix
Eigen::MatrixXd ClassificationMetrics::confusion_matrix() const {
    if(confusion_matrix_.size() == 0){
        Eigen::MatrixXd matrix = Eigen::MatrixXd::Zero(classes_.size(), classes_.size());

        for (size_t i = 0; i < predictions_.size(); ++i) {
            int actual = y_true_[i];
            int predicted = predictions_[i];
            matrix(actual, predicted)++;
        }

        return matrix;

    }else{

        return confusion_matrix_;

    }
}

// Accuracy
double ClassificationMetrics::accuracy() {
    check_computed();

    int correct = 0;
    for (int i = 0; i < confusion_matrix_.rows(); ++i) {
        correct += confusion_matrix_(i, i);
    }
    return static_cast<double>(correct) / predictions_.size();
}

// Precision for a specific class
double ClassificationMetrics::precision(int class_label) {
    check_computed();

    double true_positive = confusion_matrix_(class_label, class_label);
    double false_positive = confusion_matrix_.col(class_label).sum() - true_positive;

    if (true_positive + false_positive == 0) {
        return 0.0;
    }

    return true_positive / (true_positive + false_positive);
}


// Recall for a specific class
double ClassificationMetrics::recall(int class_label) {
    check_computed();

    double true_positive = confusion_matrix_(class_label, class_label);
    double false_negative = confusion_matrix_.row(class_label).sum() - true_positive;

    if (true_positive + false_negative == 0) {
        return 0.0;
    }

    return true_positive / (true_positive + false_negative);
}


// F1 Score for a specific class
double ClassificationMetrics::f1_score(int class_label) {
    double prec = precision(class_label);
    double rec = recall(class_label);
    return (prec + rec) > 0 ? 2 * (prec * rec) / (prec + rec) : 0.0;
}

} // namespace L
