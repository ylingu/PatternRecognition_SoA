#include <opencv2/opencv.hpp>

import xml_parser;
import bovw_feature;
import svm_classifier;
import object_detection;
import std;

using namespace std::string_view_literals;
// --- Configuration ---
// Standard size for feature extraction
const auto standard_size = cv::Size(64, 64);
constexpr auto train_image_path = "../../../../data/images/train"sv;
constexpr auto train_annotation_path = "../../../../data/annotations/train"sv;
constexpr auto val_image_path = "../../../../data/images/val"sv;
constexpr auto val_annotation_path = "../../../../data/annotations/val"sv;
// Max IoU for a patch to be considered initial negative
constexpr auto neg_iou_threshold = 0.1f;
// Target initial negatives = positives * multiplier
constexpr auto initial_neg_multiplier = 2;
// How many random patches to try per image (adjust as needed)
constexpr auto samples_per_image = 20;
constexpr auto initial_svm_model_path = "svm_initial_bovw.xml"sv;
constexpr auto final_svm_model_path = "svm_final_bovw.xml"sv;
constexpr auto vocab_path = "bovw_vocabulary.yml"sv;
// Max IoU for a detection to be considered hard negative
constexpr auto hard_neg_iou_threshold = 0.3f;
// Lenient score threshold for mining hard negatives (adjust as needed)
constexpr auto score_threshold = 0.3f;
constexpr auto nms_threshold = 0.4f;  // NMS threshold for hard negatives

auto main() -> int {
    std::println("Starting Hard Negative Mining Pipeline...");
    std::println("Standard sample size: {}x{}",
                 standard_size.width,
                 standard_size.height);

    // --- Step 1: Load Data & Prepare Positives ---
    std::println(
        "\n--- Step 1: Loading Annotations and Extracting Positive Samples "
        "---");
    std::vector<cv::Mat> positive_samples;
    std::map<std::string, std::vector<cv::Rect>>
        image_gt_map;  // Map image path to its ground truth boxes

    auto annotations = XmlParser::parseDirectory(train_annotation_path);

    std::println("Found {} annotation files.", annotations.size());

    int processed_images = 0;
    for (const auto& annotation : annotations) {
        const auto img_path =
            std::filesystem::path(train_image_path) / annotation.filename;
        cv::Mat img = cv::imread(img_path.string());

        if (img.empty()) {
            std::println(
                stderr, "Warning: Could not load image: {}", img_path.string());
            continue;
        }

        std::vector<cv::Rect> current_gt_boxes;
        for (const auto& object : annotation.objects) {
            auto gt_box = object.toCvRect();
            current_gt_boxes.push_back(gt_box);

            // Crop positive patch
            auto pos_patch = img(gt_box);

            // Resize patch
            cv::Mat resized_patch;
            cv::resize(pos_patch, resized_patch, standard_size);
            std::println("index: {}, name: {}",
                         positive_samples.size(),
                         annotation.filename);
            positive_samples.push_back(resized_patch);
        }

        image_gt_map[img_path.string()] =
            current_gt_boxes;  // Store GT boxes only if image had valid
        // objects processed
        processed_images++;

        if (processed_images > 0 && processed_images % 100 == 0) {
            std::println("Processed {} images for positives...",
                         processed_images);
        }
    }

    if (positive_samples.empty()) {
        std::println(
            stderr,
            "Error: No positive samples were extracted. Check image paths "
            "and annotations.");
        return 1;
    }
    std::println("Extracted {} positive samples from {} images.",
                 positive_samples.size(),
                 image_gt_map.size());

    // --- Step 2: Generate Initial Negative Samples ---
    // std::cout << "\n--- Step 2: Generating Initial Negative Samples ---"
    //          << std::endl;
    std::println(
        "\n--- Step 2: Generating Initial Negative Samples (if needed) ---");
    std::vector<cv::Mat> initial_negative_samples;
    auto target_neg_count = positive_samples.size() * initial_neg_multiplier;
    std::size_t a = 0;
    std::println("Target number of initial negatives: {}", target_neg_count);

    auto rng = std::mt19937(std::random_device{}());  // Random number generator

    int attempts = 0;
    constexpr int max_attempts_multiplier =
        10;  // Try harder to find negatives if needed

    // Loop until enough negatives are found or max attempts reached
    while (initial_negative_samples.size() < target_neg_count &&
           attempts < target_neg_count * max_attempts_multiplier) {
        // Iterate through images that have ground truth boxes
        for (const auto& [img_path_str, gt_boxes] : image_gt_map) {
            auto img = cv::imread(img_path_str);
            if (img.empty()) continue;  // Skip if image loading fails here too

            // Define distributions for random coordinates
            auto dist_x = std::uniform_int_distribution<int>(
                0, img.cols - standard_size.width);
            auto dist_y = std::uniform_int_distribution<int>(
                0, img.rows - standard_size.height);

            for (int i = 0; i < samples_per_image; ++i) {
                attempts++;
                int x = dist_x(rng);
                int y = dist_y(rng);
                cv::Rect random_rect(
                    x, y, standard_size.width, standard_size.height);

                // Calculate max IoU with ground truth boxes for this image
                auto max_iou = 0.0f;
                for (const auto& gt_box : gt_boxes) {
                    max_iou =
                        std::max(max_iou, CalculateIoU(random_rect, gt_box));
                }

                // If IoU is low enough, it's an initial negative
                if (max_iou < neg_iou_threshold) {
                    initial_negative_samples.push_back(img(random_rect));

                    if (initial_negative_samples.size() >= target_neg_count) {
                        goto negatives_found;  // Exit loops once target is
                                               // reached
                    }
                }
                if (attempts >= target_neg_count * max_attempts_multiplier) {
                    goto negatives_found;  // Exit if too many attempts
                }
            }
        }
        // If one pass wasn't enough, loop again over the images
    }

negatives_found:  // Label for goto statement
    std::println("Generated {} initial negative samples after {} attempts.",
                 initial_negative_samples.size(),
                 attempts);

    if (initial_negative_samples.size() <
        target_neg_count / 2) {  // Check if we got a reasonable number
        std::println(stderr,
                     "Warning: Could not generate enough initial negative "
                     "samples. Proceeding with {}.",
                     initial_negative_samples.size());
    }

    // --- Step 3: Build/Load BoVW Dictionary ---
    std::println("\n--- Step 3: Building/Loading BoVW Dictionary ---");
    // Instantiate BoVW extractor (using default dictionary size 1000, hessian
    // 400) You might want to adjust these parameters
    auto bovw_extractor = std::make_shared<FeatureExtractor>(1000, 4);

    // Option 1: Build vocabulary from positive samples
    auto vocab_built = false;
    if (std::filesystem::exists(vocab_path)) {
        std::println("Attempting to load existing vocabulary from: {}",
                     vocab_path);
        if (bovw_extractor->LoadVocabulary(vocab_path)) {
            vocab_built = true;
            std::println("Vocabulary loaded successfully.");
        } else {
            std::println(
                stderr,
                "Warning: Failed to load existing vocabulary. Rebuilding...");
        }
    }

    if (!vocab_built) {
        std::println("Building vocabulary from {} positive samples...",
                     positive_samples.size());
        if (bovw_extractor->BuildVocabulary(positive_samples)) {
            // std::cout << "Vocabulary built successfully." << std::endl;
            std::println("Vocabulary built successfully (Size: {} x {}).",
                         bovw_extractor->GetDictionarySize(),
                         positive_samples[0].total());
            // Save the built vocabulary for future use
            if (!bovw_extractor->SaveVocabulary(vocab_path)) {
                std::println(
                    stderr,
                    "Warning: Failed to save the built vocabulary to {}",
                    vocab_path);
            }
            vocab_built = true;
        } else {
            std::println(stderr,
                         "Error: Failed to build vocabulary. Cannot proceed.");
            return 1;
        }
    }

    if (!bovw_extractor->IsVocabularyReady()) {
        std::println(
            stderr,
            "Error: BoVW vocabulary is not ready after build/load attempt.");
        return 1;
    }

    std::println("\n--- Step 4: Extracting Features for Initial Training ---");
    cv::Mat positive_features;
    cv::Mat initial_negative_features;

    std::println("Extracting features for {} positive samples...",
                 positive_samples.size());
    try {
        positive_features = bovw_extractor->BatchExtract(positive_samples);
        if (positive_features.rows !=
            static_cast<long long>(positive_samples.size())) {
            throw std::runtime_error(
                "Positive feature matrix row count mismatch.");
        }
        std::println("Positive features extracted (Size: {} x {})",
                     positive_features.rows,
                     positive_features.cols);
    } catch (const std::exception& e) {
        std::println(
            stderr, "Error extracting positive features: {}", e.what());
        return 1;
    }

    if (!initial_negative_samples.empty()) {
        std::println("Extracting features for {} initial negative samples...",
                     initial_negative_samples.size());
        try {
            initial_negative_features =
                bovw_extractor->BatchExtract(initial_negative_samples);
            if (initial_negative_features.rows !=
                static_cast<long long>(initial_negative_samples.size())) {
                throw std::runtime_error(
                    "Initial negative feature matrix row count mismatch.");
            }
            std::println("Initial negative features extracted (Size: {} x {})",
                         initial_negative_features.rows,
                         initial_negative_features.cols);
        } catch (const std::exception& e) {
            std::println(stderr,
                         "Error extracting initial negative features: {}",
                         e.what());
            return 1;
        }
    } else {
        std::println(
            stderr,
            "Warning: No initial negative samples to extract features from.");
        return 1;
    }

    std::println(
        "\n--- Step 5: Initial SVM Training (using initial negatives) ---");
    // Prepare data for SVM
    auto num_pos_initial = positive_features.rows;
    auto num_neg_initial = initial_negative_features.rows;
    auto total_samples_initial = num_pos_initial + num_neg_initial;

    if (total_samples_initial == 0) {
        std::println(stderr,
                     "Error: No features available for initial SVM training.");
        return 1;
    }
    auto feature_dim = bovw_extractor->GetDictionarySize();
    auto initial_combined_features =
        cv::Mat(total_samples_initial, feature_dim, CV_32F);
    auto initial_combined_labels = cv::Mat(total_samples_initial, 1, CV_32S);
    initial_combined_labels.reserve(total_samples_initial);

    // Add positive features and labels (Label 1)

    initial_combined_features.rowRange(0, num_pos_initial) =
        positive_features;  // Copy positive features to the top
    initial_combined_labels.rowRange(0, num_pos_initial) = 1;
    std::println("Added {} positive samples to initial training data.",
                 num_pos_initial);

    // Add initial negative features and labels (Label 0)
    initial_combined_features.rowRange(num_pos_initial, total_samples_initial) =
        initial_negative_features;  // Copy negative features to the bottom
    initial_combined_labels.rowRange(num_pos_initial, total_samples_initial) =
        0;
    std::println("Added {} initial negative samples to initial training data.",
                 num_neg_initial);

    std::println("Initial combined training data: {} samples, {} features.",
                 initial_combined_features.rows,
                 initial_combined_features.cols);

    // Instantiate SVMClassifier (passing the BoVW extractor)
    // Use default SVM parameters for now
    SVMClassifier initial_svm_classifier(bovw_extractor);

    // Train initial SVM model
    std::println("Training initial SVM model...");
    initial_svm_classifier.Train(initial_combined_features,
                                 initial_combined_labels);

    // Save initial SVM model
    std::println("Saving initial SVM model to: {}", initial_svm_model_path);
    initial_svm_classifier.SaveModel(initial_svm_model_path);

    std::println("\n--- Step 6: Mining Hard Negatives (using initial SVM) ---");
    std::vector<cv::Mat> hard_negative_samples;

    // Check if the initial SVM model exists before trying to load/use it
    if (!std::filesystem::exists(initial_svm_model_path)) {
        std::println(stderr, "Error: Initial SVM model not found.");
        return 1;
    } else {
        std::println("Mining hard negatives using initial SVM...");
        auto mining_image_count = 0;
        for (const auto& [img_path_str, gt_boxes] : image_gt_map) {
            auto img = cv::imread(img_path_str);
            if (img.empty()) continue;

            try {
                // Detect objects using the initial SVM and lenient threshold
                auto detections = DetectObjects(img,
                                                initial_svm_classifier,
                                                standard_size,
                                                0,
                                                nms_threshold);
                for (const auto& detection : detections) {
                    // Calculate max IoU with ground truth boxes for this
                    // detection
                    float max_iou = 0.0f;
                    for (const auto& gt_box : gt_boxes) {
                        max_iou = std::max(max_iou,
                                           CalculateIoU(detection.box, gt_box));
                    }

                    // If IoU is low enough, it's a hard negative
                    if (max_iou < hard_neg_iou_threshold) {
                        // Crop the detected region
                        auto detected_rect = detection.box;
                        if (detected_rect.width > 0 &&
                            detected_rect.height > 0) {
                            cv::Mat hard_neg_patch = img(detected_rect);
                            if (!hard_neg_patch.empty()) {
                                // Resize to standard size
                                cv::Mat resized_hard_neg;
                                cv::resize(hard_neg_patch,
                                           resized_hard_neg,
                                           standard_size);
                                hard_negative_samples.push_back(
                                    resized_hard_neg);
                            }
                        }
                    }
                }
            } catch (const std::exception& e) {
                std::println(stderr,
                             " Error during detection on image {}: {}",
                             img_path_str,
                             e.what());
                // Continue to next image
            }

            mining_image_count++;
            if (mining_image_count % 50 == 0) {
                std::println("Processed {} images for hard negative mining...",
                             mining_image_count);
            }
        }
        std::println("Mining complete. Found {} hard negative samples.",
                     hard_negative_samples.size());
    }  // End of check for initial SVM model existence

    std::println(
        "\n--- Step 7: Extracting Features for Hard Negative Samples ---");
    cv::Mat hard_negative_features;
    if (!hard_negative_samples.empty()) {
        std::println("Extracting features for {} hard negative samples...",
                     hard_negative_samples.size());
        try {
            hard_negative_features =
                bovw_extractor->BatchExtract(hard_negative_samples);
            if (hard_negative_features.rows !=
                static_cast<long long>(hard_negative_samples.size())) {
                throw std::runtime_error(
                    "Hard negative feature matrix row count mismatch.");
            }
            std::println("Hard negative features extracted (Size: {} x {})",
                         hard_negative_features.rows,
                         hard_negative_features.cols);
        } catch (const std::exception& e) {
            std::println(stderr,
                         "Error extracting hard negative features: {}",
                         e.what());
            return 1;
        }
    } else {
        std::println(
            "Warning: No hard negative samples were mined or available to "
            "extract features from. Proceeding with empty feature matrix.");
        return 1;
    }

    std::println("\n--- Step 8: Final SVM Training  ---");
    // Prepare data for final SVM training (Positives + Hard Negatives)
    auto num_pos_final = positive_features.rows;  // Use original positives
    auto num_neg_hard = hard_negative_features.rows;
    auto total_samples_final = num_pos_final + num_neg_hard;

    cv::Mat final_combined_features =
        cv::Mat::zeros(total_samples_final, feature_dim, CV_32F);

    auto final_combined_labels = cv::Mat(total_samples_final, 1, CV_32S);

    // Add positive features and labels (Label 1)
    final_combined_features.rowRange(0, num_pos_final) = positive_features;
    final_combined_labels.rowRange(0, num_pos_final) = 1;
    std::println("Added {} positive samples to final training data.",
                 num_pos_final);

    // Add hard negative features and labels (Label 0)
    final_combined_features.rowRange(num_pos_final, total_samples_final) =
        hard_negative_features;
    final_combined_labels.rowRange(num_pos_final, total_samples_final) = 0;

    std::println("Added {} hard negative samples to final training data.",
                 num_neg_hard);

    std::println("Final combined training data: {} samples, {} features.",
                 final_combined_features.rows,
                 final_combined_features.cols);

    // Instantiate a new SVMClassifier for the final model
    SVMClassifier final_svm_classifier(bovw_extractor);

    // Train final SVM model
    std::println("Training final SVM model...");
    final_svm_classifier.Train(final_combined_features, final_combined_labels);

    // Save final SVM model
    std::println("Saving final SVM model to: {}", final_svm_model_path);
    final_svm_classifier.SaveModel(final_svm_model_path);

    std::println("\nHard Negative Mining Pipeline Completed Successfully!");
    std::println("Final model saved to: {}", final_svm_model_path);
    std::println("Vocabulary saved to/loaded from: {}", vocab_path);

    // Calculate mIOU and precision and recall on validation set
    std::println("\n--- Step 9: Evaluating on Validation Set ---");
    auto val_annotations = XmlParser::parseDirectory(val_annotation_path);
    std::println("Found {} validation annotation files.",
                 val_annotations.size());
    std::map<std::string, std::vector<cv::Rect>> val_image_gt_map;
    for (const auto& annotation : val_annotations) {
        const auto img_path =
            std::filesystem::path(val_image_path) / annotation.filename;
        std::vector<cv::Rect> current_gt_boxes;
        for (const auto& object : annotation.objects) {
            auto gt_box = object.toCvRect();
            current_gt_boxes.push_back(gt_box);
        }

        val_image_gt_map[img_path.string()] = current_gt_boxes;
    }
    auto mIoU = 0.0f;
    auto tp = 0;
    auto fp = 0;
    auto fn = 0;
    for (const auto& [path, objects] : val_image_gt_map) {
        auto img = cv::imread(path);
        auto detections = DetectObjects(img,
                                        final_svm_classifier,
                                        standard_size,
                                        score_threshold,
                                        nms_threshold);
        auto detected = std::vector<bool>(detections.size(), false);
        auto gt_detected = std::vector<bool>(objects.size(), false);
        for (size_t i = 0; i < detections.size(); ++i) {
            auto& detection = detections[i];
            auto max_iou = 0.0f;
            auto max_index = -1;
            for (size_t j = 0; j < objects.size(); ++j) {
                auto& gt_box = objects[j];
                auto iou = CalculateIoU(detection.box, gt_box);
                if (iou > max_iou) {
                    max_iou = iou;
                    max_index = static_cast<int>(j);
                }
            }
            if (max_iou > 0.5f && !gt_detected[max_index]) {
                tp++;
                mIoU += max_iou;
                detected[i] = true;
                gt_detected[max_index] = true;
            } else {
                fp++;
            }
        }
        for (size_t j = 0; j < objects.size(); ++j) {
            if (!gt_detected[j]) {
                fn++;
            }
        }
    }
    std::println("mIoU: {:.2f}, TP: {}, FP: {}, FN: {}, P: {:.2f}, R: {:.2f}",
                 mIoU / tp,
                 tp,
                 fp,
                 fn,
                 (1.0f * tp) / (tp + fp),
                 (1.0f * tp) / (tp + fn));

    return 0;
}
