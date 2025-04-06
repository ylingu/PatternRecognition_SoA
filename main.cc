#include "feature_extraction.h"

// 用于可视化LBP特征的函数
auto VisualizeLBP(const cv::Mat &lbp_image,
                  const cv::Mat &original_image,
                  bool use_color_mapping) -> cv::Mat {
    cv::Mat visual;

    // 找到LBP值的范围用于归一化
    double min_val, max_val;
    cv::minMaxLoc(lbp_image, &min_val, &max_val);

    // 创建8位归一化可视化图像
    cv::Mat lbp_normalized;
    lbp_image.convertTo(
        lbp_normalized, CV_8U, 255.0 / (max_val > 0 ? max_val : 1));

    if (use_color_mapping) {
        // 使用颜色映射以获得更好的可视化效果
        cv::applyColorMap(lbp_normalized, visual, cv::COLORMAP_JET);
    } else {
        cv::cvtColor(lbp_normalized, visual, cv::COLOR_GRAY2BGR);
    }

    // 如果提供了原始图像，创建叠加效果
    if (!original_image.empty()) {
        cv::Mat overlay;
        if (original_image.channels() == 1) {
            cv::cvtColor(original_image, overlay, cv::COLOR_GRAY2BGR);
        } else {
            overlay = original_image.clone();
        }

        // 如果需要，调整大小
        if (overlay.size() != visual.size()) {
            cv::resize(overlay, overlay, visual.size());
        }

        // 使用50%的alpha值混合
        cv::addWeighted(visual, 0.5, overlay, 0.5, 0, visual);
    }

    return visual;
}


auto main() -> int {
    // 读取图像
    cv::Mat img = cv::imread("Lenna.png");
    if (img.empty()) {
        std::cerr << "Error: Could not read the image." << std::endl;
        return -1;
    }

    // 定义LBP参数
    int radius = 1;
    int neighbors = 8;

    // 创建LBP特征提取器 - 三种类型，两种邻居数
    LBPFeatureExtraction circularLBP(radius, neighbors);
    LBPFeatureExtraction rotationLBP(
        radius, neighbors, LBPFeatureExtraction::LBPType::Rotation);
    LBPFeatureExtraction uniformLBP(
        radius, neighbors, LBPFeatureExtraction::LBPType::Uniform);

    // 转为灰度图用于处理
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    // 提取8邻居LBP特征
    auto circularLBPImage = circularLBP.GetLBPMat(gray);
    auto rotationLBPImage = rotationLBP.GetLBPMat(gray);
    auto uniformLBPImage = uniformLBP.GetLBPMat(gray);

    // 可视化LBP结果
    cv::Mat circularVis = VisualizeLBP(circularLBPImage, cv::Mat(), true);
    cv::Mat rotationVis = VisualizeLBP(rotationLBPImage, cv::Mat(), true);
    cv::Mat uniformVis = VisualizeLBP(uniformLBPImage, cv::Mat(), true);

    // 保存结果
    cv::imwrite("circular_lbp_8.png", circularVis);
    cv::imwrite("rotation_lbp_8.png", rotationVis);
    cv::imwrite("uniform_lbp_8.png", uniformVis);
    return 0;
}
