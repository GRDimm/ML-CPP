#ifndef L_LINEARREGRESSION_HPP
#define L_LINEARREGRESSION_HPP

#include <Eigen/Dense>

namespace L {

class LinearRegression {
public:
    LinearRegression();
    void fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y);    // Utilise Eigen pour les données d'entrée
    Eigen::VectorXd predict(const Eigen::MatrixXd& X) const;         // Prédictions avec une matrice Eigen

    Eigen::VectorXd getCoefficients() const;  // Renvoie les coefficients (les pentes pour chaque feature)
    double getIntercept() const;              // Renvoie l'ordonnée à l'origine
    
private:
    Eigen::VectorXd coefficients; // Pentes pour chaque feature
    double intercept;             // Ordonnée à l'origine
};

} // namespace L

#endif // L_LINEARREGRESSION_HPP
