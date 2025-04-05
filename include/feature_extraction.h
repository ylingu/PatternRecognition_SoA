#ifndef FEATURE_EXTRACTION_H
#define FEATURE_EXTRACTION_H

#include <Eigen/Dense>
#include <opencv2/core/types.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

#include "custom_hog.h"

/**
 * @class FeatureExtraction
 * @brief Abstract class for extracting features from images.
 *
 * Provides a common interface for feature extraction techniques such as GLCM,
 * Gabor, HOG, and LBP.
 */
class FeatureExtraction {
protected:
    static int kLevel;  // Level
public:
    /**
     * @brief Compresses an image using specified parameters.
     *
     * This static method can be used to compress an image before feature
     * extraction to reduce computation.
     *
     * @param img The image to be compressed.
     * @return The compressed cv::Mat image.
     */
    static auto Compress(const cv::Mat &img) -> cv::Mat;

    /**
     * @brief Pure virtual method to extract features from a single image.
     *
     * Must be implemented by derived classes to extract features according to
     * specific algorithms.
     *
     * @param img The image from which features are to be extracted.
     * @return An Eigen::VectorXd containing the extracted features.
     */
    virtual auto Extract(const cv::Mat &img) -> Eigen::VectorXd = 0;

    /**
     * @brief Pure virtual method to extract features from a batch of images.
     *
     * Must be implemented by derived classes to extract features from multiple
     * images efficiently.
     *
     * @param data A vector of images from which features are to be extracted.
     * @return An Eigen::MatrixXd where each row contains the features extracted
     * from one image.
     */
    virtual auto BatchExtract(const std::vector<cv::Mat> &data)
        -> Eigen::MatrixXd = 0;

    /**
     * @brief Default destructor.
     */
    virtual ~FeatureExtraction() = default;
};

/**
 * @class GLCMFeatureExtraction
 * @brief GLCM (Gray Level Co-occurrence Matrix) feature extraction class.
 *
 * Inherits from FeatureExtraction and implements feature extraction using the
 * GLCM method. This class is designed to compute GLCM features such as
 * contrast, correlation, energy, and homogeneity from images. It supports
 * extracting features from both single images and batches of images.
 */
class GLCMFeatureExtraction : public FeatureExtraction {
private:
    int d_;  ///< Distance parameter for GLCM computation.
    std::vector<double>
        angles_;  ///< Angles to be considered in GLCM computation.
public:
    /**
     * @brief Constructs a GLCMFeatureExtraction object with specified angles
     * and distance.
     *
     * @param angles A vector of angles (in radians) to be used for GLCM
     * computation. Default angles are 0, 45, 90, and 135 degrees.
     * @param d The distance parameter for GLCM computation. Default is 1.
     */
    GLCMFeatureExtraction(const std::vector<double> &angles = {0.0,
                                                               CV_PI / 4,
                                                               CV_PI / 2,
                                                               CV_PI * 3 / 4},
                          int d = 1)
        : angles_(angles), d_(d) {}

    /**
     * @brief Extracts GLCM features from a single image.
     *
     * This method overrides the pure virtual method from the FeatureExtraction
     * class. It computes the GLCM for the given image and extracts features
     * such as contrast, correlation, energy, and homogeneity.
     *
     * @param img The image from which features are to be extracted.
     * @return An Eigen::VectorXd containing the extracted GLCM features.
     */
    auto Extract(const cv::Mat &img) -> Eigen::VectorXd override;

    /**
     * @brief Extracts GLCM features from a batch of images.
     *
     * This method implements efficient extraction of GLCM features from
     * multiple images. It is particularly useful for processing large datasets
     * or real-time applications.
     *
     * @param data A vector of images from which features are to be extracted.
     * @return An Eigen::MatrixXd where each row contains the GLCM features
     * extracted from one image.
     */
    auto BatchExtract(const std::vector<cv::Mat> &data)
        -> Eigen::MatrixXd override;
};


/**
 * @class GaborFeatureExtraction
 * @brief Class for Gabor feature extraction
 *
 * This class inherits from FeatureExtraction and implements Gabor feature
 * extraction. Gabor features are particularly useful for texture analysis,
 * especially in the fields of image processing and computer vision.
 */
class GaborFeatureExtraction : public FeatureExtraction {
private:
    std::vector<cv::Mat> kernels_;  ///< Gabor kernels.
    std::vector<double> thetas_;    ///< Orientations for Gabor kernels.
    std::vector<double> lambdas_;   ///< Wavelengths for Gabor kernels.
    cv::Size kernel_size_;          ///< Size of the Gabor kernel.
    double gamma_;                  ///< Gamma value for Gabor kernel.
    double sigma_;                  ///< Sigma value for Gabor kernel.
    double psi_;                    ///< Psi value for Gabor kernel.
public:
    /**
     * @brief This class is used for extracting Gabor features from images.
     *
     * GaborFeatureExtraction utilizes Gabor filters for texture analysis in
     * images. It applies Gabor filters with specified orientations and
     * wavelengths to the input image and extracts features that are useful for
     * various image processing and computer vision tasks, such as texture
     * classification and object recognition.
     *
     * @param thetas A constant reference to a std::vector<double> containing
     * the orientations (in radians) for the Gabor filters. These orientations
     * determine the direction of the texture features to be extracted.
     * @param lambdas A constant reference to a std::vector<double> containing
     * the wavelengths (in pixels) of the sinusoidal wave component of the Gabor
     * filters. The wavelengths determine the scale of the texture features to
     * be extracted.
     * @param kernel_size Optional parameter specifying the size of the Gabor
     * kernel. Default is cv::Size(17, 17). The kernel size affects the locality
     * of the texture features being analyzed.
     * @param gamma Optional parameter specifying the spatial aspect ratio of
     * the Gabor kernel. Default value is 1. Gamma influences the ellipticity of
     * the support of the Gabor function.
     * @param sigma Optional parameter specifying the standard deviation of the
     * Gaussian envelope of the Gabor kernel. Default value is CV_PI. Sigma
     * controls the width of the Gaussian envelope, affecting the scale of
     * texture features that are emphasized.
     * @param psi Optional parameter specifying the phase offset of the
     * sinusoidal wave component of the Gabor kernel. Default value is 0. Psi
     * allows adjustment of the phase of the Gabor filter, which can influence
     * the response of the filter to certain textures.
     */
    GaborFeatureExtraction(const std::vector<double> &thetas,
                           const std::vector<double> &lambdas,
                           const cv::Size kernel_size = cv::Size(17, 17),
                           const double gamma = 1,
                           const double sigma = CV_PI,
                           const double psi = 0)
        : thetas_(thetas),
          lambdas_(lambdas),
          kernel_size_(kernel_size),
          gamma_(gamma),
          sigma_(sigma),
          psi_(psi) {}

    /**
     * @brief Generates Gabor kernels for a given set of orientations and
     * wavelengths.
     *
     * This function populates internal structures with Gabor kernels calculated
     * for the specified orientations and wavelengths. Each combination of theta
     * and lambda results in a unique Gabor kernel, which can be used for
     * texture analysis, feature extraction, and other image processing tasks.
     * The generated kernels are stored internally and can be accessed through
     * other class methods.
     */
    inline auto GetGaborKernel() -> void {
        for (auto theta : thetas_) {
            for (auto lambda : lambdas_) {
                cv::Mat kernel = cv::getGaborKernel(
                    kernel_size_, sigma_, theta, lambda, gamma_, 0, CV_32F);
                kernels_.push_back(kernel);
            }
        }
    }

    /**
     * @brief Extracts Gabor features from a single image.
     *
     * This method overrides the pure virtual method from the
     * FeatureExtraction class. It computes the Gabor features for the given
     * image.
     *
     * @param img The image from which features are to be extracted.
     * @return An Eigen::VectorXd containing the extracted Gabor features.
     */
    auto Extract(const cv::Mat &img) -> Eigen::VectorXd override;

    /**
     * @brief Extracts Gabor features from a batch of images.
     *
     * This method implements efficient extraction of Gabor features from
     * multiple images. It is particularly useful for processing large datasets
     * or real-time applications.
     *
     * @param data A vector of images from which features are to be extracted.
     * @return An Eigen::MatrixXd where each row contains the Gabor features
     * extracted from one image.
     */
    auto BatchExtract(const std::vector<cv::Mat> &data)
        -> Eigen::MatrixXd override;
};

/**
 * @class HOGFeatureExtraction
 * @brief Class for Histogram of Oriented Gradients (HOG) feature extraction.
 *
 * This class inherits from the FeatureExtraction base class and implements
 * the extraction of Histogram of Oriented Gradients (HOG) features, which are
 * widely used in image processing and computer vision, particularly for object
 * detection tasks.
 */
class HOGFeatureExtraction : public FeatureExtraction {
private:
    CustomHOGDescriptor hog_;  ///< Custom HOG descriptor.
    cv::Size win_size_;        ///< Window size for HOG computation.
    cv::Size block_size_;      ///< Block size for HOG computation.
    cv::Size block_stride_;    ///< Block stride for HOG computation.
    cv::Size cell_size_;       ///< Cell size for HOG computation.
    int nbins_;                ///< Number of bins for HOG computation.
public:
    /**
     * @brief This class is used for extracting HOG features from images.
     *
     * HOGFeatureExtraction utilizes Histogram of Oriented Gradients (HOG) for
     * feature extraction. It computes the HOG descriptor for the input image,
     * which represents the distribution of gradient orientations in local
     * regions of the image. The HOG features are useful for object detection,
     * image classification, and other computer vision tasks.
     *
     * @param win_size Optional parameter specifying the window size for HOG
     * computation. Default is cv::Size(160, 120). The window size determines
     * the size of the detection window used for computing the HOG descriptor.
     * @param block_size Optional parameter specifying the block size for HOG
     * computation. Default is cv::Size(16, 16). The block size defines the
     * spatial region over which the HOG descriptor is computed.
     * @param block_stride Optional parameter specifying the block stride for
     * HOG computation. Default is cv::Size(8, 8). The block stride determines
     * the step size for moving the block window across the detection window.
     * @param cell_size Optional parameter specifying the cell size for HOG
     * computation. Default is cv::Size(8, 8). The cell size defines the size of
     * the spatial bins used for computing the histogram of gradient
     * orientations.
     * @param nbins Optional parameter specifying the number of bins for HOG
     * computation. Default is 9. The number of bins determines the granularity
     * of the gradient orientation histogram.
     */
    HOGFeatureExtraction(const cv::Size win_size = cv::Size(160, 120),
                         const cv::Size block_size = cv::Size(16, 16),
                         const cv::Size block_stride = cv::Size(8, 8),
                         const cv::Size cell_size = cv::Size(8, 8),
                         const int nbins = 9)
        : win_size_(win_size),
          block_size_(block_size),
          block_stride_(block_stride),
          cell_size_(cell_size),
          nbins_(nbins) {
        hog_ = CustomHOGDescriptor(
            win_size_, block_size_, block_stride_, cell_size_, nbins_);
    }

    /**
     * @brief Extracts HOG features from a single image.
     *
     * This method overrides the pure virtual method from the FeatureExtraction
     * class. It computes the HOG descriptor for the given image.
     *
     * @param img The image from which features are to be extracted.
     * @return An Eigen::VectorXd containing the extracted HOG features.
     */
    auto Extract(const cv::Mat &img) -> Eigen::VectorXd override;

    /**
     * @brief Extracts HOG features from a batch of images.
     *
     * This method implements efficient extraction of HOG features from multiple
     * images. It is particularly useful for processing large datasets or
     * real-time applications.
     *
     * @param data A vector of images from which features are to be extracted.
     * @return An Eigen::MatrixXd where each row contains the HOG features
     * extracted from one image.
     */
    auto BatchExtract(const std::vector<cv::Mat> &data)
        -> Eigen::MatrixXd override;
};

class LBPFeatureExtraction : public FeatureExtraction {
private:
    int radius_;     ///< Radius for LBP computation.
    int neighbors_;  ///< Number of neighbors for LBP computation.
public:
    /**
     * @brief Constructs an LBPFeatureExtraction object with specified radius
     * and number of neighbors.
     *
     * @param radius The radius parameter for LBP computation. Default is 1.
     * @param neighbors The number of neighbors parameter for LBP computation.
     * Default is 8.
     */
    LBPFeatureExtraction(int radius = 1, int neighbors = 8)
        : radius_(radius), neighbors_(neighbors) {}

    /**
     * @brief Performs bilinear interpolation on an image at a given point.
     *
     * This function calculates the intensity value at a non-integer (x, y)
     * position in an image using bilinear interpolation. Bilinear interpolation
     * is a method of interpolating the value of a function on a rectilinear 2D
     * grid for a non-grid point. It is based on linear interpolation first in
     * one direction, and then again in the other direction.
     *
     * @param img The source image as a cv::Mat object. The image should be a
     * single-channel (grayscale) image for the interpolation to work correctly.
     * @param x The x-coordinate of the point where interpolation is desired.
     * This is a floating-point value, and it does not need to be an integer.
     * @param y The y-coordinate of the point where interpolation is desired.
     * Similar to x, this is a floating-point value.
     * @return The interpolated intensity value at the given (x, y) position.
     * The return type is double, which allows for fractional intensity values.
     */
    inline static auto BilinearInterpolation(const cv::Mat &img,
                                             double x,
                                             double y) -> double {
        int x1 = static_cast<int>(x), x2 = x1 + 1;
        int y1 = static_cast<int>(y), y2 = y1 + 1;
        double dx = x - x1, dy = y - y1;
        return (1 - dx) * (1 - dy) * img.at<uchar>(y1, x1) +
               dx * (1 - dy) * img.at<uchar>(y1, x2) +
               (1 - dx) * dy * img.at<uchar>(y2, x1) +
               dx * dy * img.at<uchar>(y2, x2);
    }

    /**
     * @brief Extracts LBP features from a single image.
     *
     * This method overrides the pure virtual method from the FeatureExtraction
     * class. It computes the LBP for the given image.
     *
     * @param img The image from which features are to be extracted.
     * @return An Eigen::VectorXd containing the extracted LBP features.
     */
    auto Extract(const cv::Mat &img) -> Eigen::VectorXd override;

    /**
     * @brief Extracts LBP features from a batch of images.
     *
     * This method implements efficient extraction of LBP features from multiple
     * images. It is particularly useful for processing large datasets or
     * real-time applications.
     *
     * @param data A vector of images from which features are to be extracted.
     * @return An Eigen::MatrixXd where each row contains the LBP features
     * extracted from one image.
     */
    auto BatchExtract(const std::vector<cv::Mat> &data)
        -> Eigen::MatrixXd override;
};
#endif  // FEATURE_EXTRACTION_H