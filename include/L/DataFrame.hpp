#ifndef L_DATAFRAME_HPP
#define L_DATAFRAME_HPP

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <variant>
#include <map>
#include <Eigen/Dense>

namespace L {

class DataFrame {
public:
    using DataType = std::variant<int, double, float, long, std::string>;  // Supports multiple numeric types and strings
    using Row = std::vector<DataType>;

    DataFrame() = default;
    DataFrame(const Eigen::MatrixXd& matrix, const std::vector<std::string>& column_names);
    DataFrame(const Eigen::VectorXd& vector, const std::string& column_name);

    bool readCSV(const std::string& filename);

    // Export DataFrame to CSV
    bool toCsv(const std::string& filename) const;

    void print() const;

    // Column operations
    DataFrame selectColumns(const std::vector<std::string>& column_names) const;
    Eigen::MatrixXd toMatrix() const;

    // Row operations
    void head(size_t n = 5) const;
    void tail(size_t n = 5) const;

    // Get column or row
    std::vector<DataType> getColumn(const std::string& column_name) const;
    Row getRow(size_t index) const;
    size_t getRowCount() const { return data.size(); }  // Number of rows in DataFrame

private:
    std::vector<std::string> column_names;
    std::vector<Row> data;
    std::map<std::string, size_t> column_indices;

    DataType parseValue(const std::string& value) const;
};

} // namespace L

#endif // L_DATAFRAME_HPP
