#ifndef U_TREEUTILS_HPP
#define U_TREEUTILS_HPP

#include <Eigen/Dense>

namespace U {

// Structure to represent a tree node
struct TreeNode {
    int feature_index;      // Feature used for the split
    double threshold;       // Threshold value for the split
    int class_label;        // Class label for leaf nodes
    TreeNode* left;         // Pointer to left child
    TreeNode* right;        // Pointer to right child

    TreeNode() 
        : feature_index(-1), threshold(0.0), class_label(-1), left(nullptr), right(nullptr) {}
};

// Utility class for tree operations
class TreeUtils {
public:
    // Calculate Gini impurity for a split
    static double calculateGini(const Eigen::VectorXd& y_left, const Eigen::VectorXd& y_right);

    // Find the best split for a given dataset
    static void findBestSplit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y, int& best_feature,
                              double& best_threshold, double& best_gini);
};

} // namespace U

#endif // U_TREEUTILS_HPP
