#ifndef L_LOGISTICREGRESSION_HPP
#define L_LOGISTICREGRESSION_HPP

#include <Eigen/Dense>

namespace L {

class LogisticRegression {
public:
    LogisticRegression();
    void fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y, double learning_rate = 0.01, int iterations = 1000);
    Eigen::VectorXd predict(const Eigen::MatrixXd& X) const;         // Prédictions avec une matrice Eigen

    Eigen::VectorXd getCoefficients() const;  // Renvoie les coefficients (les pentes pour chaque feature)
    double getIntercept() const;              // Renvoie l'ordonnée à l'origine
    
private:
    Eigen::VectorXd coefficients; // Pentes pour chaque feature
    double intercept;             // Ordonnée à l'origine
};

} // namespace L

#endif // L_LOGISTICREGRESSION_HPP
