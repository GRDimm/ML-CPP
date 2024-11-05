#include "L/DataFrame.hpp"

namespace L {

// Constructor that creates a DataFrame from an Eigen::VectorXd with a specified column name
DataFrame::DataFrame(const Eigen::VectorXd& vector, const std::string& column_name) {
    column_names.push_back(column_name);
    column_indices[column_name] = 0;

    for (int i = 0; i < vector.size(); ++i) {
        Row row;
        row.push_back(vector(i));
        data.push_back(row);
    }
}

// Constructor that creates a DataFrame from an Eigen::MatrixXd with a list of column names
DataFrame::DataFrame(const Eigen::MatrixXd& matrix, const std::vector<std::string>& column_names) {
    if (matrix.cols() != column_names.size()) {
        throw std::invalid_argument("Number of columns in matrix does not match size of column_names.");
    }

    this->column_names = column_names;

    for (size_t i = 0; i < column_names.size(); ++i) {
        column_indices[column_names[i]] = i;
    }

    for (int i = 0; i < matrix.rows(); ++i) {
        Row row;
        for (int j = 0; j < matrix.cols(); ++j) {
            row.push_back(matrix(i, j));
        }
        data.push_back(row);
    }
}

bool DataFrame::readCSV(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return false;
    }

    std::string line;
    bool is_header = true;

    while (std::getline(file, line)) {
        std::istringstream line_stream(line);
        std::string cell;
        Row row;

        if (is_header) {
            while (std::getline(line_stream, cell, ',')) {
                column_names.push_back(cell);
                column_indices[cell] = column_names.size() - 1;
            }
            is_header = false;
        } else {
            while (std::getline(line_stream, cell, ',')) {
                row.push_back(parseValue(cell));
            }
            data.push_back(row);
        }
    }

    file.close();
    return true;
}

DataFrame::DataType DataFrame::parseValue(const std::string& value) const {
    try { return std::stoi(value); } catch (...) {}
    try { return std::stol(value); } catch (...) {}
    try { return std::stof(value); } catch (...) {}
    try { return std::stod(value); } catch (...) {}
    return value;
}

DataFrame DataFrame::selectColumns(const std::vector<std::string>& selected_column_names) const {
    DataFrame new_df;

    // Filter column indices based on the selected column names
    std::vector<size_t> selected_indices;
    for (const auto& name : selected_column_names) {
        auto it = column_indices.find(name);
        if (it != column_indices.end()) {
            selected_indices.push_back(it->second);
            new_df.column_names.push_back(name);
            new_df.column_indices[name] = new_df.column_names.size() - 1;
        } else {
            std::cerr << "Column " << name << " not found in DataFrame." << std::endl;
        }
    }

    // Add only the selected columns to the new DataFrame
    for (const auto& row : data) {
        DataFrame::Row new_row;
        for (auto index : selected_indices) {
            new_row.push_back(row[index]);
        }
        new_df.data.push_back(new_row);
    }

    return new_df;
}
Eigen::MatrixXd DataFrame::toMatrix() const {
    Eigen::MatrixXd matrix(getRowCount(), column_names.size());
    for (size_t i = 0; i < getRowCount(); ++i) {
        for (size_t j = 0; j < column_names.size(); ++j) {
            const auto& value = data[i][j];
            // Ensure all values are numeric for conversion to Eigen matrix
            if (std::holds_alternative<int>(value)) {
                matrix(i, j) = std::get<int>(value);
            } else if (std::holds_alternative<double>(value)) {
                matrix(i, j) = std::get<double>(value);
            } else if (std::holds_alternative<float>(value)) {
                matrix(i, j) = std::get<float>(value);
            } else if (std::holds_alternative<long>(value)) {
                matrix(i, j) = std::get<long>(value);
            } else {
                throw std::invalid_argument("Non-numeric value in DataFrame for toMatrix conversion");
            }
        }
    }
    return matrix;
}

std::vector<DataFrame::DataType> DataFrame::getColumn(const std::string& column_name) const {
    std::vector<DataType> column;
    auto it = column_indices.find(column_name);
    if (it == column_indices.end()) {
        std::cerr << "Column not found: " << column_name << std::endl;
        return column;
    }

    size_t col_index = it->second;
    for (const auto& row : data) {
        column.push_back(row[col_index]);
    }
    return column;
}

bool DataFrame::toCsv(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return false;
    }

    // Write column headers
    for (size_t i = 0; i < column_names.size(); ++i) {
        file << column_names[i];
        if (i < column_names.size() - 1) {
            file << ",";
        }
    }
    file << "\n";

    // Write each row of data
    for (const auto& row : data) {
        for (size_t i = 0; i < row.size(); ++i) {
            // Handle different data types in the DataFrame
            std::visit([&file](auto&& value) { file << value; }, row[i]);
            if (i < row.size() - 1) {
                file << ",";
            }
        }
        file << "\n";
    }

    file.close();
    return true;
}

} // namespace L
