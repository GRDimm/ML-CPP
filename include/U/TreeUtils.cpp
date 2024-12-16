#include "TreeUtils.hpp"
#include <limits>

namespace U {

double TreeUtils::calculateGini(const Eigen::VectorXd& y_left, const Eigen::VectorXd& y_right) {
    auto calculateProportions = [](const Eigen::VectorXd& y) -> std::unordered_map<int, double> {
        std::unordered_map<int, double> proportions;
        for (int i = 0; i < y.size(); ++i) {
            proportions[y[i]]++;
        }
        for (auto& [key, count] : proportions) {
            proportions[key] = count / y.size();
        }
        return proportions;
    };

    auto gini = [](const std::unordered_map<int, double>& proportions) -> double {
        double impurity = 1.0;
        for (const auto& [_, p] : proportions) {
            impurity -= p * p;
        }
        return impurity;
    };

    double total = y_left.size() + y_right.size();
    double gini_left = gini(calculateProportions(y_left));
    double gini_right = gini(calculateProportions(y_right));

    return (y_left.size() / total) * gini_left + (y_right.size() / total) * gini_right;
}

void TreeUtils::findBestSplit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y, int& best_feature,
                              double& best_threshold, double& best_gini) {
    best_feature = -1;
    best_threshold = 0.0;
    best_gini = std::numeric_limits<double>::max();

    for (int feature = 0; feature < X.cols(); ++feature) {
        std::vector<double> thresholds;
        for (int i = 0; i < X.rows(); ++i) {
            thresholds.push_back(X(i, feature));
        }
        std::sort(thresholds.begin(), thresholds.end());
        thresholds.erase(std::unique(thresholds.begin(), thresholds.end()), thresholds.end());

        for (const double& threshold : thresholds) {
            Eigen::VectorXd y_left, y_right;
            for (int i = 0; i < X.rows(); ++i) {
                if (X(i, feature) <= threshold) {
                    y_left.conservativeResize(y_left.size() + 1);
                    y_left[y_left.size() - 1] = y[i];
                } else {
                    y_right.conservativeResize(y_right.size() + 1);
                    y_right[y_right.size() - 1] = y[i];
                }
            }

            double gini = calculateGini(y_left, y_right);
            if (gini < best_gini) {
                best_gini = gini;
                best_feature = feature;
                best_threshold = threshold;
            }
        }
    }
}

} // namespace U
