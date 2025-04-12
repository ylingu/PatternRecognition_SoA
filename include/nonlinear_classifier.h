#ifndef NONLINEAR_CLASSIFIER_H
#define NONLINEAR_CLASSIFIER_H

#include <Eigen/Dense>
#include <memory>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

#include "csv.h"
#include "feature_extraction.h"
#include "utils.h"

/**
 * @struct KNNTreeNode
 * @brief A node in the k tree.
 * @tparam T The type of data stored in the node.
 */
template <typename T>
struct KNNTreeNode {
    T data;                                 ///< The data stored in the node.
    int split;                              ///< The split criterion.
    int label;                              ///< The label of the node.
    std::unique_ptr<KNNTreeNode<T>> left;   ///< The left child node.
    std::unique_ptr<KNNTreeNode<T>> right;  ///< The right child node.

    /**
     * @brief Construct a new TreeNode object
     * @param data The data to be stored in the node.
     * @param split The split criterion.
     * @param label The label of the node.
     */
    KNNTreeNode(T data, int split, int label)
        : data(data),
          split(split),
          label(label),
          left(nullptr),
          right(nullptr) {}
};

/**
 * @class KNNClassifier
 * @brief A k-nearest neighbors classifier.
 */
class KNNClassifier {
public:
    /**
     * @brief Construct a new KNNClassifier object
     * @param k The number of neighbors to consider.
     */
    KNNClassifier(int k) : k_(k) {}

    /**
     * @brief Fit the model to the training data.
     * @param train_data The training data.
     */
    auto Fit(const std::vector<std::pair<Eigen::VectorXd, int>> &train_data)
        -> void;

    /**
     * @brief Predict the labels of the test data.
     * @param test_data The test data.
     * @return The predicted labels.
     */
    auto Predict(const std::vector<Eigen::VectorXd> &test_data) const
        -> std::vector<int>;

private:
    int k_;  ///< The number of neighbors to consider.
    std::unique_ptr<KNNTreeNode<Eigen::VectorXd>>
        root_;  ///< The root of the decision tree.
};

/**
 * @class SVMClassifier
 * @brief A support vector machine classifier.
 */
class SVMClassifier {
private:
    int class_num_;             ///< The number of classes.
    cv::Ptr<cv::ml::SVM> svm_;  ///< The SVM model.
    static int kEpochs;         ///< The number of training epochs.
    static double kEpsilon;     ///< The tolerance for stopping criterion.
    std::unique_ptr<Normalization>
        normalize_strategy_;  ///< Strategy for data normalization.
    std::unique_ptr<FeatureExtraction>
        feature_extractor_;  ///< Strategy for feature extraction.

public:
    /**
     * @brief Constructs a SVMClassifier with optional normalization and
     * feature extraction strategies.
     *
     * This constructor initializes a SVMClassifier with customizable
     * normalization and feature extraction strategies. If not provided, the
     * feature extraction strategy defaults to GLCMFeatureExtraction.
     *
     * @param class_num The number of classes in the data.
     * @param normalize_strategy_ Unique pointer to a Normalization strategy
     * object (default is nullptr).
     * @param feature_extractor Unique pointer to a FeatureExtraction strategy
     * object (default is GLCMFeatureExtraction).
     */
    SVMClassifier(const int class_num,
                  std::unique_ptr<Normalization> normalize_strategy_ = nullptr,
                  std::unique_ptr<FeatureExtraction> feature_extractor =
                      std::make_unique<GLCMFeatureExtraction>())
        : class_num_(class_num),
          normalize_strategy_(std::move(normalize_strategy_)),
          feature_extractor_(std::move(feature_extractor)) {}

    /**
     * @brief Preprocesses the input data.
     *
     * This function takes a vector of images as input and applies preprocessing
     * steps to prepare the data for classification. The preprocessing steps can
     * include normalization, feature extraction, and dimensionality reduction,
     * depending on the implementation. The preprocessed data is returned as an
     * Eigen::MatrixXd, where each row represents the feature vector of an
     * image.
     *
     * @param data A std::vector of cv::Mat objects representing the input
     * images.
     * @return An Eigen::MatrixXd where each row is the feature vector of the
     * corresponding input image.
     */
    inline auto Preprocess(const std::vector<cv::Mat> &data) -> cv::Mat {
        auto features = feature_extractor_->BatchExtract(data);
        if (normalize_strategy_ != nullptr) {
            features = normalize_strategy_->Normalize(features);
        }
        Eigen::MatrixXf temp = features.cast<float>();
        cv::Mat features_mat;
        cv::eigen2cv(temp, features_mat);
        return features_mat;
    }

    /**
     * @brief Train the model.
     * @param train_data The training data.
     * @param train_label The labels of the training data.
     * @param validate_data The validation data.
     * @param validate_label The labels of the validation data.
     * @param c_range The range of the C parameter.
     * @param gamma_range The range of the gamma parameter.
     */
    auto Train(const cv::Mat &train_data,
               const std::vector<int> &train_label,
               const cv::Mat &validate_data,
               const std::vector<int> &validate_label,
               const std::vector<double> &c_range,
               const std::vector<double> &gamma_range) -> void;

    /**
     * @brief Predict the labels of the test data.
     * @param test_data_mat The test data.
     * @param svm The SVM model. If not provided, use the trained model.
     * @return The predicted labels.
     */
    auto Predict(const cv::Mat &test_data_mat,
                 const std::optional<cv::Ptr<cv::ml::SVM>> &svm = std::nullopt)
        -> std::vector<int>;

    /**
     * @brief Predict the raw scores of the test data.
     * @param test_data_mat The test data.
     * @return The predicted raw scores.
     */
    auto PredictRaw(const cv::Mat &test_data_mat) -> std::vector<float>;

    /**
     * @brief Save the trained model.
     * @param filepath The path to save the model. Default is "model.xml".
     */
    inline auto SaveModel(const std::string &filepath = "model.xml") -> void {
        svm_->save(filepath);
    }

    /**
     * @brief Load a trained model.
     * @param filepath The path to load the model from. Default is "model.xml".
     */
    inline auto LoadModel(const std::string &filepath = "model.xml") -> void {
        svm_ = cv::ml::SVM::load(filepath);
    }
};

/**
 * @struct DecisionTreeNode
 * @brief A node in the decision tree.
 *
 * Each node contains an attribute to split on, a label, and a map of children
 * nodes.
 */
struct DecisionTreeNode {
    std::string attribute;  ///< The attribute to split on.
    std::string label;      ///< The label of the node.
    std::map<std::string, std::unique_ptr<DecisionTreeNode>>
        children;  ///< The children of the node.

    /**
     * @brief Construct a new DecisionTreeNode object.
     *
     * @param attribute The attribute to split on.
     * @param label The label of the node.
     */
    DecisionTreeNode(const std::string &attribute, const std::string &label)
        : attribute(attribute), label(label), children() {}
};

/**
 * @class DecisionTree
 * @brief A decision tree classifier.
 *
 * The DecisionTree class is a classifier that uses a decision tree to make
 * predictions.
 */
class DecisionTree {
private:
    std::unique_ptr<DecisionTreeNode>
        root_;  ///< The root of the decision tree.

public:
    /**
     * @brief Construct a new DecisionTree object
     */
    DecisionTree() : root_(nullptr) {}

    /**
     * @brief Fit the model to the training data.
     * @param train_data The training data.
     */
    auto Build(std::string csv_filepath) -> void;

    /**
     * @brief Predict the labels of the test data.
     * @param test_data The test data.
     * @return The predicted labels.
     */
    auto Predict(const CSV &test_data) const -> std::vector<std::string>;

private:
    /**
     * @brief Calculate the entropy of a set of labels.
     *
     * @param labels The labels.
     * @return The entropy.
     */
    static auto CalcEntropy(const std::vector<std::string> &labels) -> double;
};

#endif  // NONLINEAR_CLASSIFIER_H