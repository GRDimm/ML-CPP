# ML_CPP Project

This project is a C++ library for basic machine learning and data manipulation, featuring modules for data handling, linear regression, and regression metrics.

## Project Structure

```
.
|-- CMakeLists.txt         # CMake build configuration
|-- LICENSE                # License for the project
|-- README.md              # Project documentation
|-- include/
|   `-- L/
|       |-- DataFrame.hpp          # DataFrame class for CSV handling and data manipulation
|       |-- LinearRegression.hpp   # LinearRegression class for linear regression models
|       `-- RegressionMetrics.hpp  # RegressionMetrics class for evaluation metrics
`-- src/
    |-- DataFrame.cpp              # Implementation of the DataFrame class
    |-- LinearRegression.cpp       # Implementation of the LinearRegression class
    |-- RegressionMetrics.cpp      # Implementation of the RegressionMetrics class
    `-- main.cpp                   # Example usage and test of the classes
```

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
   ./build/executable
   ```

## Usage

Example usage for each module can be found in `main.cpp`.

## Requirements

- C++17 or later
- [Eigen library](https://eigen.tuxfamily.org/) for linear algebra operations
