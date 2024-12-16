#ifndef L_DECISIONTREECLASSIFIER_HPP
#define L_DECISIONTREECLASSIFIER_HPP

#include <Eigen/Dense>
#include "../U/TreeUtils.hpp"

namespace L {

class DecisionTreeClassifier {
public:
    explicit DecisionTreeClassifier(const int max_depth = 1000);

    void fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y);
    Eigen::VectorXd predict(const Eigen::MatrixXd& X) const;
    Eigen::MatrixXd predict_proba(const Eigen::MatrixXd& X) const;

    ~DecisionTreeClassifier();


private:
    const int max_depth_;
    U::TreeNode* root_; // Use TreeNode from U namespace

    U::TreeNode* buildTree(const Eigen::MatrixXd& X, const Eigen::VectorXd& y, int depth);
    int predictInstance(const Eigen::VectorXd& instance, U::TreeNode* node) const;
    void deleteTree(U::TreeNode* node);
};

} // namespace L

#endif // L_DECISIONTREECLASSIFIER_HPP
