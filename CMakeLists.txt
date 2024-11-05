cmake_minimum_required(VERSION 3.14)
project(ML_CPP)

include(FetchContent)

# Add the include directory for headers
include_directories(${PROJECT_SOURCE_DIR}/include)

# Define the library and include both source files
add_library(L STATIC src/LinearRegression.cpp src/DataFrame.cpp src/RegressionMetrics.cpp)

# Fetch Eigen library
FetchContent_Declare(
  Eigen3
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4.0
)

FetchContent_MakeAvailable(Eigen3)

# Link Eigen to the library L
target_link_libraries(L PRIVATE Eigen3::Eigen)

# Define the executable and link both L and Eigen
add_executable(executable src/main.cpp)
target_link_libraries(executable PRIVATE L Eigen3::Eigen)
