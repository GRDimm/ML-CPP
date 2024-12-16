cmake_minimum_required(VERSION 3.14)
project(ML_CPP)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set the default build type to Debug if not specified
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
  set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
endif()

# Specify the output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# FetchContent is included by default in CMake >= 3.11
include(FetchContent)

# Fetch Eigen library
FetchContent_Declare(
  Eigen3
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4.0
)
FetchContent_MakeAvailable(Eigen3)

# Add library for TreeUtils in the U namespace
add_library(U STATIC
    include/U/TreeUtils.cpp
    include/U/MatrixUtils.cpp
)

target_include_directories(U 
    PUBLIC ${PROJECT_SOURCE_DIR}/src/U
)
target_link_libraries(U PRIVATE Eigen3::Eigen)

# Define the L library with DecisionTreeClassifier.cpp and link TreeUtils
add_library(L STATIC 
    src/LinearRegression.cpp 
    src/DataFrame.cpp 
    src/RegressionMetrics.cpp 
    src/PrincipalComponentAnalysis.cpp
    src/LogisticRegression.cpp
    src/ClassificationMetrics.cpp
    src/DecisionTreeClassifier.cpp
)

# Specify include directories for the library
target_include_directories(L 
    PUBLIC ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(L PRIVATE Eigen3::Eigen U)

# Define the executable and link both L and Eigen
add_executable(logistic_with_pca_option examples/logistic_with_pca_option/main.cpp)

# Specify include directories for the executable
target_include_directories(logistic_with_pca_option 
    PUBLIC ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(logistic_with_pca_option PRIVATE L Eigen3::Eigen)

# Define the executable for decision_tree_classifier
add_executable(decision_tree_classifier examples/decision_tree_classifier/main.cpp)

# Specify include directories for the executable
target_include_directories(decision_tree_classifier 
    PUBLIC ${PROJECT_SOURCE_DIR}/include
)

# Link the executable to the library L and Eigen
target_link_libraries(decision_tree_classifier PRIVATE L Eigen3::Eigen)
