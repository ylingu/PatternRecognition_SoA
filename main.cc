#include <opencv2/imgcodecs.hpp>

#include "feature_extraction.h"

cv::Mat visualizeHOGFeatures(const cv::Mat& img,
                             const Eigen::VectorXd& descriptors,
                             cv::Size win_size,
                             cv::Size cell_size,
                             int nbins) {
    cv::Mat vis;
    cv::resize(img, vis, win_size);
    cv::cvtColor(vis, vis, cv::COLOR_GRAY2BGR);

    int cells_x = win_size.width / cell_size.width;
    int cells_y = win_size.height / cell_size.height;

    // 遍历每个 cell
    for (int y = 0; y < cells_y; y++) {
        for (int x = 0; x < cells_x; x++) {
            int offset = (y * cells_x + x) * nbins;
            // cell 中心
            cv::Point cell_center(x * cell_size.width + cell_size.width / 2,
                                  y * cell_size.height + cell_size.height / 2);
            // 对每个 bin 绘制一个方向线段
            for (int bin = 0; bin < nbins; bin++) {
                // 计算 bin 对应角度（范围：[0, π]）
                float angle = bin * CV_PI / nbins;
                // 获取该 bin 的权重（幅值）
                float magnitude = descriptors[offset + bin];

                // 设定缩放系数以便可视化
                float scale = 10.0f;
                int dx = static_cast<int>(scale * magnitude * std::cos(angle));
                int dy = static_cast<int>(scale * magnitude * std::sin(angle));

                cv::Point pt1 = cell_center;
                cv::Point pt2 = cell_center + cv::Point(dx, dy);
                cv::arrowedLine(
                    vis, pt1, pt2, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
            }
        }
    }
    return vis;
}

auto main() -> int {
    cv::Mat img = cv::imread("Lenna.png");

    // Create an instance of HOGFeatureExtraction
    auto hog_extractor = HOGFeatureExtraction(cv::Size(64, 128));

    // Extract HOG features
    auto hog_features = hog_extractor.Extract(img);

    auto hog_vis = visualizeHOGFeatures(
        img, hog_features, cv::Size(64, 128), cv::Size(8, 8), 9);

    cv::imwrite("hog.png", hog_vis);

    return 0;
}
