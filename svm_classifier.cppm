module;

#include <opencv2/opencv.hpp>

export module svm_classifier;

export import bovw_feature;

export class SVMClassifier {
public:
    SVMClassifier(std::shared_ptr<BoVWFeatureExtraction> feature_extractor)
        : feature_extractor_(feature_extractor) {
    }  ///< Constructor for SVMClassifier

    inline auto Preprocess(const cv::Mat &data) -> cv::Mat {
        auto feature = feature_extractor_->Extract(data);
        if (feature.empty() || feature.rows != 1) {
            throw std::runtime_error("Feature extraction failed.");
        }
        return feature;
    }  ///< Preprocess the input data

    inline auto Train(const cv::Mat &train_data, const cv::Mat &train_label)
        -> void {
        svm_ = cv::ml::SVM::create();
        svm_->setKernel(cv::ml::SVM::RBF);
        svm_->setType(cv::ml::SVM::C_SVC);
        svm_->setC(1.0);
        svm_->setGamma(0.1);
        svm_->setTermCriteria(cv::TermCriteria(
            cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 1000, 1e-4));
        svm_->train(train_data, cv::ml::ROW_SAMPLE, train_label);
    }  ///< Train the SVM classifier

    inline auto Predict(const cv::Mat &test_data) -> std::vector<float> {
        assert(svm_ != nullptr);
        cv::Mat result;
        svm_->predict(test_data, result, cv::ml::StatModel::RAW_OUTPUT);
        if (result.empty() || result.type() != CV_32F ||
            result.rows != test_data.rows || result.cols != 1) {
            throw std::runtime_error("SVM prediction failed.");
        }
        return std::vector<float>(result.begin<float>(), result.end<float>());
    }

    inline auto SaveModel(std::string_view filepath = "model.xml") -> void {
        svm_->save(filepath.data());
    }  ///< Save the trained SVM model

    inline auto LoadModel(std::string_view filepath = "model.xml") -> void {
        svm_ = cv::ml::SVM::load(filepath.data());
    }  ///< Load a trained SVM model

private:
    std::shared_ptr<BoVWFeatureExtraction>
        feature_extractor_;     ///< Feature extraction strategy
    cv::Ptr<cv::ml::SVM> svm_;  ///< SVM model
};