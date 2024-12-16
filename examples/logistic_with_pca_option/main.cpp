#include <iostream>
#include "L/DataFrame.hpp"
#include "L/LinearRegression.hpp"
#include "L/RegressionMetrics.hpp"
#include "L/PrincipalComponentAnalysis.hpp"
#include "L/LogisticRegression.hpp"
#include "L/ClassificationMetrics.hpp"

#include <numeric>

#include <Eigen/Dense>

int main() {
    L::DataFrame train_df;
    L::DataFrame test_df;

    const bool use_PCA = true;

    // Load training data
    if (!train_df.readCSV("examples/datasets/persons/train.csv")) {
        std::cerr << "Failed to load train.csv" << std::endl;
        return -1;
    }

    train_df.printColumnNames();

    // Select feature columns and convert to Eigen matrix
    std::vector<std::string> feature_columns = {"Age", "Height", "Weight"};
    std::string target_column = "Genre";
    L::DataFrame features_df = train_df.selectColumns(feature_columns);
    Eigen::MatrixXd X_train = features_df.toMatrix();

    if(use_PCA){
        L::PrincipalComponentAnalysis train_PCA_object = L::PrincipalComponentAnalysis(X_train);

        train_PCA_object.transform();

        X_train = train_PCA_object.principal_components();
    }

    // Select target column and convert to Eigen vector
    L::DataFrame target_df = train_df.selectColumns({target_column});
    Eigen::VectorXd y_train = target_df.toMatrix().col(0);

    // Train the Linear Regression model
    L::LogisticRegression model;
    model.fit(X_train, y_train);

    std::cout << "Model threshold : " << model.threshold() << std::endl;

    // Load test data
    if (!test_df.readCSV("examples/datasets/persons/test.csv")) {
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

    Eigen::VectorXd predictions_proba = model.predict_proba(X_test);

    // Retrieve the "Name" column from test data
    std::vector<L::DataFrame::DataType> name_column = test_df.getColumn("Name");

    // Output predictions with the corresponding name
    std::cout << "Predictions:" << std::endl;
    for (int i = 0; i < predictions.size(); ++i) {
        std::string name = std::get<std::string>(name_column[i]);  // Assuming "Name" column is of type string
        std::cout << "Name: " << name << ", prediction: " << predictions(i) << ", proba : " << predictions_proba(i) << std::endl;
    }

    Eigen::VectorXd y_true = test_df.selectColumns({target_column}).toMatrix().col(0);

    L::ClassificationMetrics metrics(predictions, y_true);

    std::cout << "Accuracy : " << metrics.accuracy() << std::endl;
    std::cout << "Precision : " << metrics.precision() << std::endl;
    std::cout << "Recall : " << metrics.recall() << std::endl;
    std::cout << "F1 : " << metrics.f1_score() << std::endl;

    std::cout << "Confusion matrix : " << std::endl;
    std::cout << metrics.confusion_matrix() << std::endl;

    L::DataFrame predictions_dataframe = L::DataFrame(predictions, target_column);
    predictions_dataframe.toCsv("predictions.csv");

    return 0;
}