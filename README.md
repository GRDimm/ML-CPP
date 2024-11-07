# ML_CPP Project

This project is a C++ library for basic machine learning and data manipulation.

## Features

### 1. DataFrame
- **Description**: A class for handling data in a tabular format, similar to data frames in Python.
- **Current Capabilities**:
  - Read data from a CSV file.
  - Select specific columns.
  - Convert the data frame to an `Eigen::MatrixXd`.
  - Write the data frame to a CSV file.
  - Constructors for creating a `DataFrame` from an `Eigen::VectorXd` or `Eigen::MatrixXd`.

### 2. LinearRegression
- **Description**: A simple linear regression model.
- **Current Capabilities**:
  - Fit a linear regression model to training data.
  - Predict output values for test data.

### 3. RegressionMetrics
- **Description**: A class for computing regression evaluation metrics.
- **Current Capabilities**:
  - Calculate the R² score.
  - Calculate the Mean Absolute Error (MAE).
  - Calculate the Mean Squared Error (MSE).
  - Calculate the Root Mean Squared Error (RMSE).

### 4. PrincipalComponentAnalysis

- **Description**:  
  The `PrincipalComponentAnalysis` class is designed to perform Principal Component Analysis (PCA) on datasets.

- **Current Capabilities**:
  - **Compute Principal Components**:  
    Calculate the principal components.
  
  - **Transform Data**:  
    Project the original dataset onto the principal component space.

### 5. LogisticRegression
- **Description**: Logistic regression model.
- **Current Capabilities**:
  - Fit a logistic regression model to training data.
  - Predict output values for test data.

## Getting Started

1. **Clone the repository**:
   ```
   git clone https://github.com/yourusername/yourproject.git
   cd yourproject
   ```

2. **Build the project** using CMake:
   ```
   mkdir build
   cd build
   cmake ..
   make
   ```

3. **Run the example**:
   ```
   ./build/bin/executable
   ```

## Usage

Example usage be found in `main.cpp`.

## Requirements

- C++17 or later
- [Eigen library](https://eigen.tuxfamily.org/) for linear algebra operations

## Being implemented next

- Classification Metrics
- Decision Trees (Pruning, Bagging)
- Random Forest
- K-Nearest Neighbors (KNN)
- Support Vector Machines (SVM)
- Naive Bayes
- K-Means Clustering
- Gradient Boosting (e.g., XGBoost)
