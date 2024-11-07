#include <iostream>
#include "L/DataFrame.hpp"
#include "L/LinearRegression.hpp"
#include "L/RegressionMetrics.hpp"
#include "L/PrincipalComponentAnalysis.hpp"
#include <numeric>

#include <Eigen/Dense>

int main() {
    L::DataFrame train_df;
    L::DataFrame test_df;

    const bool use_PCA = true;

    // Load training data
    if (!train_df.readCSV("train.csv")) {
        std::cerr << "Failed to load train.csv" << std::endl;
        return -1;
    }

    train_df.printColumnNames();

    // Select feature columns and convert to Eigen matrix
    std::vector<std::string> feature_columns = {"Genre", "Height", "Weight"};
    L::DataFrame features_df = train_df.selectColumns(feature_columns);
    Eigen::MatrixXd X_train = features_df.toMatrix();

    if(use_PCA){
        L::PrincipalComponentAnalysis train_PCA_object = L::PrincipalComponentAnalysis(X_train);

        train_PCA_object.transform();

        X_train = train_PCA_object.principal_components();
    }

    // Select target column and convert to Eigen vector
    L::DataFrame target_df = train_df.selectColumns({"Age"});
    Eigen::VectorXd y_train = target_df.toMatrix().col(0);

    // Train the Linear Regression model
    L::LinearRegression model;
    model.fit(X_train, y_train);

    // Load test data
    if (!test_df.readCSV("test.csv")) {
        std::cerr << "Failed to load test.csv" << std::endl;
        return -1;
    }

    // Prepare test features and convert to Eigen matrix
    L::DataFrame test_features_df = test_df.selectColumns(feature_columns);
    Eigen::MatrixXd X_test = test_features_df.toMatrix();

    if(use_PCA){
        L::PrincipalComponentAnalysis test_PCA_object = L::PrincipalComponentAnalysis(X_test);

        test_PCA_object.transform();

        X_test = test_PCA_object.principal_components();
    }

    // Make predictions on the test set
    Eigen::VectorXd predictions = model.predict(X_test);

    // Retrieve the "Name" column from test data
    std::vector<L::DataFrame::DataType> name_column = test_df.getColumn("Name");

    // Output predictions with the corresponding name
    std::cout << "Predictions:" << std::endl;
    for (int i = 0; i < predictions.size(); ++i) {
        std::string name = std::get<std::string>(name_column[i]);  // Assuming "Name" column is of type string
        std::cout << "Name: " << name << ", prediction: " << predictions(i) << std::endl;
    }

    Eigen::VectorXd y_true = test_df.selectColumns({"Age"}).toMatrix().col(0);

    L::RegressionMetrics metrics(predictions, y_true);

    std::cout << "R² Score: " << metrics.r2Score() << std::endl;
    std::cout << "Mean Absolute Error: " << metrics.meanAbsoluteError() << std::endl;
    std::cout << "Mean Squared Error: " << metrics.meanSquaredError() << std::endl;
    std::cout << "Root Mean Squared Error: " << metrics.rootMeanSquaredError() << std::endl;

    L::DataFrame predictions_dataframe = L::DataFrame(predictions, "Age");
    predictions_dataframe.toCsv("predictions.csv");


    

    return 0;
}