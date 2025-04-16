module;

#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>

export module object_detection;

import std;

export import svm_classifier;
import thread_pool;

// Structure to hold detection results
export struct DetectionBox {
    cv::Rect box;
    float score;
};

// Function declarations

/**
 * @brief Helper function for NMS: Calculate Intersection over Union (IoU)
 * @param box1 First bounding box.
 * @param box2 Second bounding box.
 * @return float The IoU value.
 */
export auto CalculateIoU(const cv::Rect& box1, const cv::Rect& box2) -> float {
    auto intersection = box1 & box2;
    auto intersection_area = intersection.area();
    auto union_area = box1.area() + box2.area() - intersection_area;

    if (union_area <= 0) {
        return 0.0f;
    }
    return static_cast<float>(intersection_area) / union_area;
}

/**
 * @brief Function for Non-Maximum Suppression (NMS)
 * @param boxes Vector of detected boxes with scores.
 * @param nms_threshold Overlap threshold for suppression.
 * @return std::vector<DetectionBox> Filtered list of boxes after NMS.
 */
export auto NonMaximumSuppression(std::vector<DetectionBox>& boxes,
                                  float nms_threshold)
    -> std::vector<DetectionBox> {
    if (boxes.empty()) {
        return {};
    }

    // 1. Sort boxes by score descending
    std::sort(boxes.begin(),
              boxes.end(),
              [](const DetectionBox& a, const DetectionBox& b) {
                  return a.score > b.score;
              });

    auto picked_boxes = std::vector<DetectionBox>();
    auto suppressed = std::vector<bool>(boxes.size(), false);

    for (size_t i = 0; i < boxes.size(); ++i) {
        if (suppressed[i]) {
            continue;
        }
        picked_boxes.push_back(boxes[i]);
        suppressed[i] = true;  // Mark the current box as picked

        for (size_t j = i + 1; j < boxes.size(); ++j) {
            if (suppressed[j]) {
                continue;
            }
            float iou = CalculateIoU(boxes[i].box, boxes[j].box);
            if (iou > nms_threshold) {
                suppressed[j] = true;  // Suppress overlapping box
            }
        }
    }
    return picked_boxes;
}

/**
 * @brief Main detection function using sliding window and image pyramid.
 * @param image Input image (will be converted to grayscale if needed).
 * @param classifier Trained SVMClassifier instance (configured with HOG
 * extractor).
 * @param window_size The detection window size (must match HOG/SVM training).
 * @param score_threshold Minimum score from SVM to consider a detection (e.g.,
 * 0.5).
 * @param nms_threshold IoU threshold for Non-Maximum Suppression (e.g., 0.3).
 * @return std::vector<DetectionBox> List of final detected boxes after NMS.
 */
export auto DetectObjects(const cv::Mat& image,
                          SVMClassifier& classifier,
                          const cv::Size& window_size,
                          float score_threshold,
                          float nms_threshold = 0.5)
    -> std::vector<DetectionBox> {
    std::vector<DetectionBox> all_detections;
    cv::Mat original_image;
    image.copyTo(original_image);  // Keep original

    // Selective Search works on color images
    if (original_image.empty()) {
        std::println(stderr, "Error: Input image is empty.");
        return {};
    }
    if (original_image.channels() != 3) {
        // Convert grayscale to BGR if needed by Selective Search
        cv::cvtColor(original_image, original_image, cv::COLOR_GRAY2BGR);
    }

    // --- Selective Search ---
    cv::Ptr<cv::ximgproc::segmentation::SelectiveSearchSegmentation> ss =
        cv::ximgproc::segmentation::createSelectiveSearchSegmentation();
    ss->setBaseImage(original_image);
    ss->switchToSelectiveSearchQuality();  // Use quality mode

    std::vector<cv::Rect> rects;
    try {
        ss->process(rects);
    } catch (const cv::Exception& e) {
        std::println(
            stderr, "Error during Selective Search process: {}", e.what());
        return {};
    }
    std::println("Selective Search generated {} regions.", rects.size());
    auto pool = ThreadPool();
    auto detection_mutex = std::mutex();
    auto futures = std::vector<std::future<void>>();
    futures.reserve(rects.size());
    // --- Process Candidate Regions ---
    for (const auto& rect : rects) {
        futures.emplace_back(pool.enqueue([=,
                                           &classifier,
                                           &detection_mutex,
                                           &all_detections] {
            // Ensure rect is within image bounds (Selective Search can
            // sometimes propose slightly outside)
            cv::Rect valid_rect =
                rect & cv::Rect(0, 0, original_image.cols, original_image.rows);

            cv::Mat patch = original_image(valid_rect);
            cv::Mat resized_patch;

            // Resize the patch to the fixed window size expected by HOG/SVM
            cv::resize(patch, resized_patch, window_size);

            // Ensure resized patch is grayscale for HOG if needed
            if (resized_patch.channels() == 3) {
                cv::cvtColor(resized_patch, resized_patch, cv::COLOR_BGR2GRAY);
            }
            if (resized_patch.type() != CV_8U) {
                resized_patch.convertTo(resized_patch, CV_8U);
            }

            // --- Feature Extraction & Prediction ---
            // This assumes classifier's feature_extractor_ is set to HOG
            cv::Mat feature;
            try {
                feature =
                    classifier.Preprocess(resized_patch);  // Use resized patch
            } catch (const std::exception& e) {
                std::println(
                    stderr,
                    "Error during Preprocess for region [{}, {}, {}, {}]: {}",
                    valid_rect.x,
                    valid_rect.y,
                    valid_rect.width,
                    valid_rect.height,
                    e.what());
                return;
            }

            if (const auto score = classifier.Predict(feature);
                score > score_threshold &&
                score != -std::numeric_limits<float>::infinity()) {
                // Store the original region rectangle (valid_rect) and score
                std::lock_guard<std::mutex> lock(detection_mutex);
                all_detections.emplace_back(valid_rect, score);
            }
        }));
    }
    // Wait for all threads to finish
    for (auto& future : futures) {
        future.wait();
    }

    std::println("Found {} potential detections before NMS.",
                 all_detections.size());

    // --- Apply Non-Maximum Suppression ---
    auto final_detections =
        NonMaximumSuppression(all_detections, nms_threshold);
    std::println("Found {} final detections after NMS.",
                 final_detections.size());

    return final_detections;
}