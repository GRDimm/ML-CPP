#include <unordered_map>
#include <Eigen/Dense>

namespace U {

    // Function to calculate mode
    int computeMode(const Eigen::VectorXd& y) {
        std::unordered_map<int, int> counts; // To store occurrences
        int mode = y(0);
        int max_count = 0;

        for (int i = 0; i < y.size(); ++i) {
            int value = static_cast<int>(y(i)); // Assuming y contains integers
            counts[value]++;

            if (counts[value] > max_count) {
                max_count = counts[value];
                mode = value;
            }
        }
        return mode;
    }

}