#include "L/DataFrame.hpp"
#include <set>
#include <unordered_map>

namespace L {

    // Constructor that creates a DataFrame from an Eigen::VectorXd with a specified column name
    DataFrame::DataFrame(const Eigen::VectorXd& vector, const std::string& column_name) {
        column_names_.push_back(column_name);
        column_indices_[column_name] = 0;

        for (int i = 0; i < vector.size(); ++i) {
            Row row;
            row.push_back(vector(i));
            data_.push_back(row);
        }
    }

    // Constructor that creates a DataFrame from an Eigen::MatrixXd with a list of column names
    DataFrame::DataFrame(const Eigen::MatrixXd& matrix, const std::vector<std::string>& column_names) {
        if (matrix.cols() != column_names.size()) {
            throw std::invalid_argument("Number of columns in matrix does not match size of column_names.");
        }

        this->column_names_ = column_names;

        for (size_t i = 0; i < column_names.size(); ++i) {
            column_indices_[column_names[i]] = i;
        }

        for (int i = 0; i < matrix.rows(); ++i) {
            Row row;
            for (int j = 0; j < matrix.cols(); ++j) {
                row.push_back(matrix(i, j));
            }
            data_.push_back(row);
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
                    column_names_.push_back(cell);
                    column_indices_[cell] = column_names_.size() - 1;
                }
                is_header = false;
            } else {
                while (std::getline(line_stream, cell, ',')) {
                    row.push_back(parseValue(cell));
                }
                data_.push_back(row);
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
            auto it = column_indices_.find(name);
            if (it != column_indices_.end()) {
                selected_indices.push_back(it->second);
                new_df.column_names_.push_back(name);
                new_df.column_indices_[name] = new_df.column_names_.size() - 1;
            } else {
                std::cerr << "Column " << name << " not found in DataFrame." << std::endl;
            }
        }

        // Add only the selected columns to the new DataFrame
        for (const auto& row : data_) {
            DataFrame::Row new_row;
            for (auto index : selected_indices) {
                new_row.push_back(row[index]);
            }
            new_df.data_.push_back(new_row);
        }

        return new_df;
    }

    Eigen::MatrixXd DataFrame::toMatrix() const {
        Eigen::MatrixXd matrix(getRowCount(), column_names_.size());
        for (size_t i = 0; i < getRowCount(); ++i) {
            for (size_t j = 0; j < column_names_.size(); ++j) {
                const auto& value = data_[i][j];
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
        auto it = column_indices_.find(column_name);
        if (it == column_indices_.end()) {
            std::cerr << "Column not found: " << column_name << std::endl;
            return column;
        }

        size_t col_index = it->second;
        for (const auto& row : data_) {
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
        for (size_t i = 0; i < column_names_.size(); ++i) {
            file << column_names_[i];
            if (i < column_names_.size() - 1) {
                file << ",";
            }
        }
        file << "\n";

        // Write each row of data
        for (const auto& row : data_) {
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

    std::vector<std::string> DataFrame::columnNames() const {
        return column_names_;
    }

    std::map<std::string, size_t> DataFrame::columnIndices() const {
        return column_indices_;
    }

    bool DataFrame::hasColumn(std::string column) const {
        return std::find(column_names_.begin(), column_names_.end(), column) != column_names_.end();
    }

    void DataFrame::printColumnNames() const{
        for (const auto& column : column_names_){
            std::cout << column << ","; 
        }
        std::cout << std::endl;
    }

    DataFrame DataFrame::oneHotEncode(const std::vector<std::string>& column_names) const {
        // New DataFrame to hold the one-hot encoded data
        DataFrame encoded_df;

        // Maps to store unique categories for each column
        std::unordered_map<std::string, std::vector<std::string>> unique_categories;

        // Step 1: Identify unique categories for each specified column
        for (const std::string& col_name : column_names) {
            // Check if the column exists
            if (column_indices_.find(col_name) == column_indices_.end()) {
                throw std::runtime_error("Column '" + col_name + "' does not exist in the DataFrame.");
            }

            size_t col_idx = column_indices_.at(col_name);
            std::set<std::string> categories_set;

            for (const auto& row : data_) {
                const DataType& value = row[col_idx];
                std::string category;

                if (std::holds_alternative<std::string>(value)) {
                    category = std::get<std::string>(value);
                } else if (std::holds_alternative<int>(value)) {
                    category = std::to_string(std::get<int>(value));
                } else {
                    throw std::runtime_error("Column '" + col_name + "' must contain string or integer values for one-hot encoding.");
                }

                categories_set.insert(category);
            }

            // Store the unique categories for this column
            unique_categories[col_name] = std::vector<std::string>(categories_set.begin(), categories_set.end());
        }

        // Step 2: Prepare new column names and indices
        std::vector<std::string> new_column_names;
        std::map<std::string, size_t> new_column_indices;

        size_t new_col_idx = 0;

        // Process the one-hot encoded columns
        for (const auto& col_name : column_names) {
            const auto& categories = unique_categories[col_name];

            for (const auto& category : categories) {
                std::string new_col_name = col_name + "_" + category;
                new_column_names.push_back(new_col_name);
                new_column_indices[new_col_name] = new_col_idx++;
            }
        }

        // Add columns that are not being one-hot encoded
        for (const auto& col_name : column_names_) {
            if (std::find(column_names.begin(), column_names.end(), col_name) == column_names.end()) {
                // Column is not being one-hot encoded
                new_column_names.push_back(col_name);
                new_column_indices[col_name] = new_col_idx++;
            }
        }

        // Step 3: Build the new data rows
        std::vector<Row> new_data;
        new_data.reserve(data_.size());

        for (const auto& row : data_) {
            Row new_row(new_column_names.size());

            // Fill in one-hot encoded columns
            for (const auto& col_name : column_names) {
                size_t col_idx = column_indices_.at(col_name);
                const DataType& value = row[col_idx];
                std::string category;

                if (std::holds_alternative<std::string>(value)) {
                    category = std::get<std::string>(value);
                } else if (std::holds_alternative<int>(value)) {
                    category = std::to_string(std::get<int>(value));
                } else {
                    throw std::runtime_error("Column '" + col_name + "' must contain string or integer values for one-hot encoding.");
                }

                const auto& categories = unique_categories[col_name];
                for (const auto& cat : categories) {
                    std::string new_col_name = col_name + "_" + cat;
                    size_t new_col_idx = new_column_indices[new_col_name];
                    new_row[new_col_idx] = (category == cat) ? 1 : 0;
                }
            }

            // Copy over the other columns
            for (const auto& col_name : column_names_) {
                if (std::find(column_names.begin(), column_names.end(), col_name) == column_names.end()) {
                    // Column is not being one-hot encoded
                    size_t old_col_idx = column_indices_.at(col_name);
                    size_t new_col_idx = new_column_indices[col_name];
                    new_row[new_col_idx] = row[old_col_idx];
                }
            }

            new_data.push_back(new_row);
        }

        // Step 4: Update the encoded DataFrame
        encoded_df.column_names_ = new_column_names;
        encoded_df.data_ = new_data;

        // Rebuild column_indices_ for the encoded_df
        encoded_df.column_indices_.clear();
        for (size_t i = 0; i < new_column_names.size(); ++i) {
            encoded_df.column_indices_[new_column_names[i]] = i;
        }

        return encoded_df;
    }

} // namespace L
