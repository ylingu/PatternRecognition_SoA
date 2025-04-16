module;

#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

export module bovw_feature;

export class BoVWFeatureExtraction {
private:
    int dictionary_size_;               // Number of visual words
    int hessian_threshold_;             // SURF Hessian Threshold
    cv::BOWKMeansTrainer bow_trainer_;  // For building the vocabulary
    cv::Ptr<cv::Feature2D> detector_;   // Feature detector (e.g., SURF)
    cv::Ptr<cv::DescriptorMatcher>
        matcher_;         // Matcher for vocabulary building/assignment
    cv::Mat vocabulary_;  // The visual vocabulary (cluster centers)
    cv::BOWImgDescriptorExtractor
        bow_extractor_;  // For extracting BoVW histograms


public:
    /**
     * @brief Constructor for BoVWFeatureExtraction.
     * @param dictionary_size Number of visual words (clusters) in the
     * vocabulary.
     * @param hessian_threshold Threshold for SURF feature detection.
     */
    BoVWFeatureExtraction(int dictionary_size = 1000,
                          int hessian_threshold = 400)
        : dictionary_size_(dictionary_size),
          hessian_threshold_(hessian_threshold),
          bow_trainer_(dictionary_size),
          detector_(cv::xfeatures2d::SURF::create(hessian_threshold_)),
          matcher_(
              cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED)),
          bow_extractor_{cv::BOWImgDescriptorExtractor(detector_, matcher_)} {};

    /**
     * @brief Builds the visual vocabulary from a set of training images.
     * @param training_images Vector of images used to build the vocabulary.
     * @return True if vocabulary building was successful, false otherwise.
     */
    bool BuildVocabulary(const std::vector<cv::Mat> &training_images);

    /**
     * @brief Loads a pre-built vocabulary from a file.
     * @param filepath Path to the vocabulary file (.yml or .xml).
     * @return True if loading was successful, false otherwise.
     */
    bool LoadVocabulary(std::string_view filepath);

    /**
     * @brief Saves the built vocabulary to a file.
     * @param filepath Path to save the vocabulary file (.yml or .xml).
     * @return True if saving was successful, false otherwise.
     */
    bool SaveVocabulary(std::string_view filepath) const;

    /**
     * @brief Sets the vocabulary manually.
     * @param vocabulary The vocabulary matrix (CV_32F).
     */
    void SetVocabulary(const cv::Mat &vocabulary);

    /**
     * @brief Extracts BoVW features (histogram) from a single image.
     * Assumes vocabulary is already built or loaded.
     * @param img The image from which features are to be extracted.
     * @return An cv::Mat containing the BoVW histogram. Returns empty
     * matrix on error.
     */
    auto Extract(const cv::Mat &img) -> cv::Mat;

    /**
     * @brief Extracts BoVW features (histograms) from a batch of images.
     * Assumes vocabulary is already built or loaded.
     * @param data A vector of images from which features are to be extracted.
     * @return An cv::Mat where each row contains the BoVW histogram for
     * one image. Returns empty matrix on error.
     */
    auto BatchExtract(const std::vector<cv::Mat> &data) -> cv::Mat;

    /**
     * @brief Gets the size of the vocabulary (dictionary size).
     * @return int The number of visual words.
     */
    int GetDictionarySize() const { return dictionary_size_; }

    /**
     * @brief Gets the built vocabulary matrix.
     * @return cv::Mat The vocabulary matrix.
     */
    cv::Mat GetVocabulary() const { return vocabulary_; }

    /**
     * @brief Checks if the vocabulary has been built or loaded.
     * @return bool True if vocabulary is ready, false otherwise.
     */
    bool IsVocabularyReady() const { return !vocabulary_.empty(); }
};

// BuildVocabulary Implementation
bool BoVWFeatureExtraction::BuildVocabulary(
    const std::vector<cv::Mat> &training_images) {
    if (training_images.empty()) {
        std::cerr
            << "Error: No training images provided for vocabulary building."
            << std::endl;
        return false;
    }
    if (!detector_) {
        std::cerr << "Error: SURF detector is not initialized." << std::endl;
        return false;
    }

    cv::Mat all_descriptors;
    int image_count = 0;
    std::cout << "Building vocabulary from " << training_images.size()
              << " images..." << std::endl;

    for (const auto &img : training_images) {
        if (img.empty()) {
            std::cerr
                << "Warning: Skipping empty image during vocabulary building."
                << std::endl;
            continue;
        }
        cv::Mat gray_img;
        if (img.channels() == 3) {
            cv::cvtColor(img, gray_img, cv::COLOR_BGR2GRAY);
        } else {
            gray_img = img;
        }
        if (gray_img.type() != CV_8U) {
            gray_img.convertTo(gray_img, CV_8U);
        }


        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        detector_->detectAndCompute(
            gray_img, cv::noArray(), keypoints, descriptors);

        if (!descriptors.empty()) {
            // Ensure descriptors are CV_32F as required by BOWKMeansTrainer
            if (descriptors.type() != CV_32F) {
                descriptors.convertTo(descriptors, CV_32F);
            }
            bow_trainer_.add(descriptors);
            image_count++;
        } else {
            std::cerr << "Warning: No features detected in one of the training "
                         "images."
                      << std::endl;
        }
        if (image_count % 100 == 0 && image_count > 0) {
            std::cout << "Processed " << image_count
                      << " images for vocabulary..." << std::endl;
        }
    }

    if (bow_trainer_.descriptorsCount() == 0) {
        std::cerr << "Error: No descriptors were collected from training "
                     "images. Cannot build vocabulary."
                  << std::endl;
        return false;
    }

    std::cout << "Clustering " << bow_trainer_.descriptorsCount()
              << " descriptors into " << dictionary_size_ << " visual words..."
              << std::endl;
    vocabulary_ = bow_trainer_.cluster();  // Perform clustering

    if (vocabulary_.empty()) {
        std::cerr << "Error: Vocabulary clustering failed." << std::endl;
        return false;
    }

    std::cout << "Vocabulary built successfully with size: " << vocabulary_.rows
              << " x " << vocabulary_.cols << std::endl;

    // Set the vocabulary for the extractor
    SetVocabulary(vocabulary_);
    return true;
}

// LoadVocabulary Implementation
bool BoVWFeatureExtraction::LoadVocabulary(std::string_view filepath) {
    cv::FileStorage fs(filepath.data(), cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "Error: Could not open vocabulary file: " << filepath
                  << std::endl;
        return false;
    }
    fs["vocabulary"] >> vocabulary_;
    fs.release();

    if (vocabulary_.empty() || vocabulary_.type() != CV_32F) {
        std::cerr << "Error: Failed to load valid vocabulary from " << filepath
                  << std::endl;
        vocabulary_.release();  // Clear invalid data
        return false;
    }

    dictionary_size_ =
        vocabulary_.rows;  // Update dictionary size based on loaded data
    std::cout << "Vocabulary loaded successfully from " << filepath
              << " (Size: " << dictionary_size_ << ")" << std::endl;

    // Re-initialize trainer and set vocabulary for extractor
    bow_trainer_ = cv::BOWKMeansTrainer(dictionary_size_);
    SetVocabulary(vocabulary_);
    return true;
}

// SaveVocabulary Implementation
bool BoVWFeatureExtraction::SaveVocabulary(std::string_view filepath) const {
    if (vocabulary_.empty()) {
        std::cerr << "Error: Vocabulary is empty, cannot save." << std::endl;
        return false;
    }
    cv::FileStorage fs(filepath.data(), cv::FileStorage::WRITE);
    if (!fs.isOpened()) {
        std::cerr << "Error: Could not open file for writing vocabulary: "
                  << filepath << std::endl;
        return false;
    }
    fs << "vocabulary" << vocabulary_;
    fs.release();
    std::cout << "Vocabulary saved successfully to " << filepath << std::endl;
    return true;
}

// SetVocabulary Implementation
void BoVWFeatureExtraction::SetVocabulary(const cv::Mat &vocabulary) {
    if (vocabulary.empty() || vocabulary.type() != CV_32F) {
        std::cerr
            << "Error: Provided vocabulary is invalid (empty or not CV_32F)."
            << std::endl;
        return;
    }
    vocabulary_ = vocabulary;
    dictionary_size_ = vocabulary_.rows;  // Update size
    // Set the vocabulary for the BOW extractor
    bow_extractor_.setVocabulary(vocabulary_);
    std::cout << "Vocabulary set for BoVW extractor. Dictionary size: "
              << dictionary_size_ << std::endl;
}


// Extract Implementation (Single Image)
auto BoVWFeatureExtraction::Extract(const cv::Mat &img) -> cv::Mat {
    if (!IsVocabularyReady()) {
        std::cerr << "Error: Vocabulary is not ready for feature extraction."
                  << std::endl;
        return cv::Mat();  // Return empty vector
    }
    if (img.empty()) {
        std::cerr << "Error: Input image is empty for Extract." << std::endl;
        return cv::Mat();
    }

    cv::Mat gray_img;
    if (img.channels() == 3) {
        cv::cvtColor(img, gray_img, cv::COLOR_BGR2GRAY);
    } else {
        gray_img = img;
    }

    std::vector<cv::KeyPoint> keypoints;
    cv::Mat bow_descriptor;

    // Detect keypoints (needed by compute, even if descriptors are extracted
    // differently sometimes)
    detector_->detect(gray_img, keypoints);

    // Compute the BoVW histogram for the image
    // Note: compute() might internally call detectAndCompute or just use
    // detected keypoints depending on the extractor setup. Here we assume it
    // uses the vocabulary set earlier.
    try {
        bow_extractor_.compute(gray_img, keypoints, bow_descriptor);
    } catch (const cv::Exception &e) {
        std::cerr << "Error during BoVW compute: " << e.what() << std::endl;
        return cv::Mat();
    }


    if (bow_descriptor.empty()) {
        // If no keypoints were found or compute failed, return a zero vector of
        // correct size std::cerr << "Warning: No BoVW descriptor computed
        // (maybe no keypoints found)." << std::endl;
        bow_descriptor = cv::Mat::zeros(1, dictionary_size_, CV_32F);
    }

    // Ensure descriptor is float and has the correct size
    if (bow_descriptor.type() != CV_32F) {
        bow_descriptor.convertTo(bow_descriptor, CV_32F);
    }
    if (bow_descriptor.cols != dictionary_size_) {
        std::cerr << "Error: BoVW descriptor size mismatch. Expected "
                  << dictionary_size_ << ", Got " << bow_descriptor.cols
                  << std::endl;
        // Handle mismatch, e.g., return zero vector or resize (though resizing
        // might be wrong)
        bow_descriptor = cv::Mat::zeros(1, dictionary_size_, CV_32F);
    }

    return bow_descriptor;
}

// BatchExtract Implementation
auto BoVWFeatureExtraction::BatchExtract(const std::vector<cv::Mat> &data)
    -> cv::Mat {
    if (!IsVocabularyReady()) {
        std::cerr
            << "Error: Vocabulary is not ready for batch feature extraction."
            << std::endl;
        return cv::Mat();  // Return empty matrix
    }
    if (data.empty()) {
        std::cerr << "Error: Input data vector is empty for BatchExtract."
                  << std::endl;
        return cv::Mat();
    }

    cv::Mat all_features =
        cv::Mat::zeros(data.size(), dictionary_size_, CV_32F);
    bool error_occurred = false;

    for (size_t i = 0; i < data.size(); ++i) {
        auto features = Extract(data[i]);
        if (features.empty()) {
            // If single extraction failed, fill row with zeros
            all_features.row(i) = cv::Mat::zeros(1, dictionary_size_, CV_32F);
            error_occurred = true;  // Mark that at least one failed
        } else {
            all_features.row(i) = features;
        }
    }
    if (error_occurred) {
        std::cerr << "Warning: Feature extraction failed for one or more "
                     "images in the batch."
                  << std::endl;
    }

    return all_features;
}
