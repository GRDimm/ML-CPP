#include <unordered_set>
#include <unordered_map>
#include <functional>
#include "L/DecisionTreeClassifier.hpp"
#include "U/TreeUtils.hpp"
#include "U/MatrixUtils.hpp"

namespace L {

DecisionTreeClassifier::DecisionTreeClassifier(const int max_depth)
    : max_depth_(max_depth), root_(nullptr) {}

void DecisionTreeClassifier::fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
    root_ = buildTree(X, y, 0);
}

Eigen::VectorXd DecisionTreeClassifier::predict(const Eigen::MatrixXd& X) const {
    Eigen::VectorXd predictions(X.rows());
    for (int i = 0; i < X.rows(); ++i) {
        predictions[i] = predictInstance(X.row(i), root_);
    }
    return predictions;
}

Eigen::MatrixXd DecisionTreeClassifier::predict_proba(const Eigen::MatrixXd& X) const {
    // Determine the number of classes
    std::unordered_set<int> class_labels;

    std::function<void(U::TreeNode*)> gatherLabels;

    gatherLabels = [this, &class_labels, &gatherLabels](U::TreeNode* node) {
        if (!node) return;
        if (!node->left && !node->right) {
            class_labels.insert(node->class_label);
        } else {
            gatherLabels(node->left);
            gatherLabels(node->right);
        }
    };

    gatherLabels(root_);

    int num_classes = class_labels.size();
    std::vector<int> unique_classes(class_labels.begin(), class_labels.end());
    std::sort(unique_classes.begin(), unique_classes.end());

    // Initialize the probability matrix
    Eigen::MatrixXd probabilities(X.rows(), num_classes);
    probabilities.setZero();
    
    std::function<void(const Eigen::VectorXd&, U::TreeNode*, std::unordered_map<int, double>&, int)> predictProbaInstance;

    // Helper to compute probabilities for a single instance
    predictProbaInstance = [this, &unique_classes, &predictProbaInstance] (const Eigen::VectorXd& instance, U::TreeNode* node, std::unordered_map<int, double>& class_counts, int depth) {
        if (!node->left && !node->right) {
            class_counts[node->class_label] += 1.0;
            return;
        }
        if (instance[node->feature_index] <= node->threshold) {
            predictProbaInstance(instance, node->left, class_counts, depth + 1);
        } else {
            predictProbaInstance(instance, node->right, class_counts, depth + 1);
        }
    };


    for (int i = 0; i < X.rows(); ++i) {
        std::unordered_map<int, double> class_counts;

        // Traverse the tree and collect class counts
        predictProbaInstance(X.row(i), root_, class_counts, 0);

        // Normalize the counts into probabilities
        double total = 0;
        for (const auto& [label, count] : class_counts) {
            total += count;
        }
        for (size_t j = 0; j < unique_classes.size(); ++j) {
            int class_label = unique_classes[j];
            probabilities(i, j) = class_counts.count(class_label) ? class_counts[class_label] / total : 0.0;
        }
    }

    assert(probabilities.rows() == X.rows() && probabilities.cols() > 0);
    assert((probabilities.array() >= 0).all() && (probabilities.array() <= 1).all());


    return probabilities;
}


U::TreeNode* DecisionTreeClassifier::buildTree(const Eigen::MatrixXd& X, const Eigen::VectorXd& y, int depth) {
    if (depth >= max_depth_ || y.size() <= 1 || std::unordered_set<double>(y.data(), y.data() + y.size()).size() == 1) {
        auto* leaf = new U::TreeNode();
        leaf->class_label = U::computeMode(y); // Assign most common class
        return leaf;
    }

    int best_feature;
    double best_threshold, best_gini;
    U::TreeUtils::findBestSplit(X, y, best_feature, best_threshold, best_gini);

    if (best_feature == -1) {
        auto* leaf = new U::TreeNode();
        leaf->class_label = U::computeMode(y);
        return leaf;
    }

    Eigen::MatrixXd X_left, X_right;
    Eigen::VectorXd y_left, y_right;

    // Initialize empty matrices/vectors
    X_left.resize(0, X.cols());
    X_right.resize(0, X.cols());
    y_left.resize(0);
    y_right.resize(0);

    for (int i = 0; i < X.rows(); ++i) {
        if (X(i, best_feature) <= best_threshold) {
            X_left.conservativeResize(X_left.rows() + 1, X.cols());
            X_left.row(X_left.rows() - 1) = X.row(i);

            y_left.conservativeResize(y_left.size() + 1);
            y_left(y_left.size() - 1) = y[i]; // Correct indexing
        } else {
            X_right.conservativeResize(X_right.rows() + 1, X.cols());
            X_right.row(X_right.rows() - 1) = X.row(i);

            y_right.conservativeResize(y_right.size() + 1);
            y_right(y_right.size() - 1) = y[i]; // Correct indexing
        }
    }

    auto* node = new U::TreeNode();
    node->feature_index = best_feature;
    node->threshold = best_threshold;
    node->left = buildTree(X_left, y_left, depth + 1);
    node->right = buildTree(X_right, y_right, depth + 1);

    return node;
}

int DecisionTreeClassifier::predictInstance(const Eigen::VectorXd& instance, U::TreeNode* node) const {
    if (!node->left && !node->right) {
        return node->class_label;
    }
    if (instance[node->feature_index] <= node->threshold) {
        return predictInstance(instance, node->left);
    } else {
        return predictInstance(instance, node->right);
    }
}

void DecisionTreeClassifier::deleteTree(U::TreeNode* node) {
    if (!node) return;
    deleteTree(node->left);
    deleteTree(node->right);
    delete node;
}

DecisionTreeClassifier::~DecisionTreeClassifier() {
    deleteTree(root_);
}

} // namespace L
