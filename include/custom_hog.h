#ifndef CUSTOM_HOG_H
#define CUSTOM_HOG_H

#include <Eigen/Dense>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <vector>

/**
 * @class CustomHOGDescriptor
 * @brief Custom implementation of Histogram of Oriented Gradients (HOG)
 * descriptor
 *
 * This class provides a custom implementation of the HOG feature descriptor
 * algorithm without relying on OpenCV's HOGDescriptor class. It maintains the
 * same parameters and interface while implementing the algorithm from scratch.
 */
class CustomHOGDescriptor {
private:
    cv::Size win_size_;      ///< Window size for HOG computation
    cv::Size block_size_;    ///< Block size for HOG computation
    cv::Size block_stride_;  ///< Block stride for HOG computation
    cv::Size cell_size_;     ///< Cell size for HOG computation
    int nbins_;              ///< Number of bins for HOG computation

    /**
     * @brief Calculates gradient magnitude and orientation for an image
     *
     * @param img Input grayscale image
     * @param magnitude Output matrix for gradient magnitudes
     * @param orientation Output matrix for gradient orientations in radians
     */
    void computeGradient(const cv::Mat& img,
                         cv::Mat& magnitude,
                         cv::Mat& orientation) const {
        cv::Mat grad_x, grad_y;
        // Compute gradients
        cv::Sobel(img, grad_x, CV_32F, 1, 0, 1);
        cv::Sobel(img, grad_y, CV_32F, 0, 1, 1);

        // Calculate magnitude and orientation
        magnitude = cv::Mat(img.size(), CV_32F);
        orientation = cv::Mat(img.size(), CV_32F);

        for (int y = 0; y < img.rows; y++) {
            for (int x = 0; x < img.cols; x++) {
                float dx = grad_x.at<float>(y, x);
                float dy = grad_y.at<float>(y, x);

                magnitude.at<float>(y, x) = std::sqrt(dx * dx + dy * dy);
                // Calculate orientation in range [0, π]
                orientation.at<float>(y, x) =
                    std::atan2(std::abs(dy), std::abs(dx));
            }
        }
    }

    /**
     * @brief Calculate histogram of gradients for a cell
     *
     * @param magnitude Gradient magnitudes
     * @param orientation Gradient orientations
     * @param cell_x Cell X position
     * @param cell_y Cell Y position
     * @param histograms Output histogram array
     */
    void calculateCellHistogram(const cv::Mat& magnitude,
                                const cv::Mat& orientation,
                                int cell_x,
                                int cell_y,
                                float* histogram) const {
        // Initialize histogram
        for (int i = 0; i < nbins_; i++) {
            histogram[i] = 0.0f;
        }

        // Calculate histogram bounds
        int x_start = cell_x * cell_size_.width;
        int y_start = cell_y * cell_size_.height;
        int x_end = std::min(x_start + cell_size_.width, magnitude.cols);
        int y_end = std::min(y_start + cell_size_.height, magnitude.rows);

        // Calculate bin width in radians (π radians / nbins)
        float bin_width = CV_PI / nbins_;

        // Fill histogram
        for (int y = y_start; y < y_end; y++) {
            for (int x = x_start; x < x_end; x++) {
                float mag = magnitude.at<float>(y, x);
                float angle = orientation.at<float>(y, x);

                // Determine bin index with bilinear interpolation
                float bin_index = angle / bin_width;
                int bin_index_low = static_cast<int>(bin_index);
                int bin_index_high = (bin_index_low + 1) % nbins_;
                float weight_high = bin_index - bin_index_low;
                float weight_low = 1.0f - weight_high;

                // Add weighted vote to histogram
                histogram[bin_index_low] += weight_low * mag;
                histogram[bin_index_high] += weight_high * mag;
            }
        }
    }

    /**
     * @brief Normalize block of histograms
     *
     * @param block Block of histograms to normalize
     * @param block_size Number of elements in block
     */
    void normalizeBlock(float* block, int block_size) const {
        float norm = 0.0f;

        // Calculate L2 norm
        for (int i = 0; i < block_size; i++) {
            norm += block[i] * block[i];
        }

        // Add epsilon to avoid division by zero
        norm = std::sqrt(norm + 1e-5f);

        // L2 normalization
        for (int i = 0; i < block_size; i++) {
            block[i] /= norm;
        }
    }

public:
    /**
     * @brief Constructor for CustomHOGDescriptor
     *
     * @param win_size Window size for HOG computation
     * @param block_size Block size for HOG computation
     * @param block_stride Block stride for HOG computation
     * @param cell_size Cell size for HOG computation
     * @param nbins Number of bins for HOG computation
     */
    CustomHOGDescriptor(const cv::Size& win_size = cv::Size(64, 128),
                        const cv::Size& block_size = cv::Size(16, 16),
                        const cv::Size& block_stride = cv::Size(8, 8),
                        const cv::Size& cell_size = cv::Size(8, 8),
                        int nbins = 9)
        : win_size_(win_size),
          block_size_(block_size),
          block_stride_(block_stride),
          cell_size_(cell_size),
          nbins_(nbins) {}

    /**
     * @brief Compute HOG descriptors for an image
     *
     * @param img Input image
     * @param descriptors Output vector of HOG descriptors
     * @param locations Optional locations to compute descriptors (not
     * implemented)
     */
    void compute(const cv::Mat& img,
                 std::vector<float>& descriptors,
                 const std::vector<cv::Point>& locations =
                     std::vector<cv::Point>()) const {
        // Convert image to grayscale if needed
        cv::Mat gray;
        if (img.channels() == 3) {
            cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = img.clone();
        }

        // Resize image if necessary
        if (gray.size() != win_size_) {
            cv::resize(gray, gray, win_size_);
        }

        // Compute gradients
        cv::Mat magnitude, orientation;
        computeGradient(gray, magnitude, orientation);

        // Calculate cell histograms
        int cells_x = win_size_.width / cell_size_.width;
        int cells_y = win_size_.height / cell_size_.height;

        // Allocate memory for cell histograms
        std::vector<float> cell_histograms(cells_x * cells_y * nbins_);

        // Compute cell histograms
        for (int x = 0; x < cells_x; x++) {
            for (int y = 0; y < cells_y; y++) {
                int offset = (x * cells_y + y) * nbins_;
                calculateCellHistogram(
                    magnitude, orientation, x, y, &cell_histograms[offset]);
            }
        }

        // Calculate blocks
        int blocks_x = (cells_x - block_size_.width / cell_size_.width) /
                           (block_stride_.width / cell_size_.width) +
                       1;
        int blocks_y = (cells_y - block_size_.height / cell_size_.height) /
                           (block_stride_.height / cell_size_.height) +
                       1;

        // Calculate block histogram size
        int cells_per_block_x = block_size_.width / cell_size_.width;
        int cells_per_block_y = block_size_.height / cell_size_.height;
        int histogram_size_per_block =
            cells_per_block_x * cells_per_block_y * nbins_;

        // Allocate space for descriptors
        descriptors.resize(blocks_x * blocks_y * histogram_size_per_block);

        // For each block, normalize the histograms
        for (int by = 0; by < blocks_y; by++) {
            for (int bx = 0; bx < blocks_x; bx++) {
                // Block position in cells
                int cell_x = bx * (block_stride_.width / cell_size_.width);
                int cell_y = by * (block_stride_.height / cell_size_.height);

                // Output descriptor index
                int descriptor_idx =
                    (by * blocks_x + bx) * histogram_size_per_block;

                // Copy and normalize block histograms
                for (int cx = 0; cx < cells_per_block_x; cx++) {
                    for (int cy = 0; cy < cells_per_block_y; cy++) {
                        // Cell position in overall histogram array
                        int cell_idx =
                            ((cell_y + cy) * cells_x + (cell_x + cx)) * nbins_;

                        // Copy cell histogram to block
                        for (int bin = 0; bin < nbins_; bin++) {
                            int block_idx =
                                ((cy * cells_per_block_x) + cx) * nbins_ + bin;
                            descriptors[descriptor_idx + block_idx] =
                                cell_histograms[cell_idx + bin];
                        }
                    }
                }

                // Normalize block
                normalizeBlock(&descriptors[descriptor_idx],
                               histogram_size_per_block);
            }
        }
    }

    /**
     * @brief Get the descriptor size
     *
     * @return Size of the descriptor in elements
     */
    size_t getDescriptorSize() const {
        int cells_per_block_x = block_size_.width / cell_size_.width;
        int cells_per_block_y = block_size_.height / cell_size_.height;
        int blocks_x =
            (win_size_.width / cell_size_.width - cells_per_block_x) /
                (block_stride_.width / cell_size_.width) +
            1;
        int blocks_y =
            (win_size_.height / cell_size_.height - cells_per_block_y) /
                (block_stride_.height / cell_size_.height) +
            1;

        return blocks_x * blocks_y * cells_per_block_x * cells_per_block_y *
               nbins_;
    }
};

#endif  // CUSTOM_HOG_H