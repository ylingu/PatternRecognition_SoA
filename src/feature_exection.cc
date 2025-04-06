#include <bitset>
#include <opencv2/opencv.hpp>

#include "feature_extraction.h"

int FeatureExtraction::kLevel = 16;
auto FeatureExtraction::Compress(const cv::Mat &img) -> cv::Mat {
    cv::Mat res = cv::Mat::zeros(img.rows, img.cols, CV_8U);
    img.convertTo(res, CV_8U, (kLevel - 1) / 255.0);
    // *双重循环图像压缩，效率较低，但会发生四舍五入某些情况下分类准确率高于直接转换
    // for (int i = 0; i < img.rows; i++) {
    //     for (int j = 0; j < img.cols; j++) {
    //         res.at<uchar>(i, j) = img.at<uchar>(i, j) * (kLevel - 1) / 255;
    //     }
    // }
    return res;
}

auto GLCMFeatureExtraction::Extract(const cv::Mat &img) -> Eigen::VectorXd {
    Eigen::VectorXd feature = Eigen::VectorXd::Zero(angles_.size() * 4);
    for (auto it = angles_.begin(); it != angles_.end(); it++) {
        Eigen::MatrixXd glcm = Eigen::MatrixXd::Zero(kLevel, kLevel);
        for (int i = 0; i != img.rows; ++i) {
            for (int j = 0; j != img.cols; ++j) {
                if (round(j + d_ * cos(*it)) < img.cols &&
                    round(i + d_ * sin(*it)) < img.rows &&
                    round(j + d_ * cos(*it)) >= 0 &&
                    round(i + d_ * sin(*it)) >= 0) {
                    glcm(img.at<uchar>(i, j),
                         img.at<uchar>(
                             static_cast<int>(round(i + d_ * sin(*it))),
                             static_cast<int>(round(j + d_ * cos(*it)))))++;
                }
                if (round(j - d_ * cos(*it)) < img.cols &&
                    round(i - d_ * sin(*it)) < img.rows &&
                    round(j - d_ * cos(*it)) >= 0 &&
                    round(i - d_ * sin(*it)) >= 0) {
                    glcm(img.at<uchar>(i, j),
                         img.at<uchar>(
                             static_cast<int>(round(i - d_ * sin(*it))),
                             static_cast<int>(round(j - d_ * cos(*it)))))++;
                }
            }
        }
        glcm /= glcm.sum();
        double energy = glcm.cwiseProduct(glcm).sum();
        double entropy = -glcm.cwiseProduct(glcm.unaryExpr([&](double x) {
                                  return x == 0 ? 0 : log(x);
                              }))
                              .sum();
        double contrast = 0, homogeneity = 0;
        for (int i = 0; i < kLevel; i++) {
            for (int j = 0; j < kLevel; j++) {
                contrast += (i - j) * (i - j) * glcm(i, j);
                homogeneity += glcm(i, j) / (1 + (i - j) * (i - j));
            }
        }
        feature.segment(std::distance(angles_.begin(), it) * 4, 4) =
            Eigen::Vector4d(entropy, energy, contrast, homogeneity);
    }
    return feature;
}

auto GLCMFeatureExtraction::BatchExtract(const std::vector<cv::Mat> &data)
    -> Eigen::MatrixXd {
    Eigen::MatrixXd features =
        Eigen::MatrixXd::Zero(data.size(), angles_.size() * 4);
    for (auto it = data.begin(); it != data.end(); it++) {
        auto img = Compress(*it);
        auto feature = Extract(img);
        features.row(std::distance(data.begin(), it)) = feature;
    }
    return features;
}

auto GaborFeatureExtraction::Extract(const cv::Mat &img) -> Eigen::VectorXd {
    Eigen::VectorXd feature = Eigen::VectorXd::Zero(kernels_.size() * 4);
    cv::Mat dest;
    for (auto it = kernels_.begin(); it != kernels_.end(); ++it) {
        cv::filter2D(img, dest, CV_32F, *it);
        cv::Scalar mean, stddev;
        cv::meanStdDev(dest, mean, stddev);
        cv::Mat squared;
        cv::multiply(dest, dest, squared);
        double energy = cv::sum(squared)[0];
        cv::Mat hist;
        float range[] = {0, static_cast<float>(kLevel)};
        const float *hist_range[] = {range};
        cv::calcHist(
            &dest, 1, 0, cv::Mat(), hist, 1, &kLevel, hist_range, true, false);
        hist /= dest.total();
        cv::Mat log_p;
        cv::log(hist, log_p);
        double entropy = -cv::sum(hist.mul(log_p))[0];
        feature.segment(std::distance(kernels_.begin(), it) * 4, 4) =
            Eigen::Vector4d(mean[0], stddev[0], energy, entropy);
    }
    return feature;
}

auto GaborFeatureExtraction::BatchExtract(const std::vector<cv::Mat> &data)
    -> Eigen::MatrixXd {
    if (kernels_.empty()) {
        GetGaborKernel();
    }
    Eigen::MatrixXd features =
        Eigen::MatrixXd::Zero(data.size(), kernels_.size() * 4);
    for (auto it = data.begin(); it != data.end(); it++) {
        auto img = Compress(*it);
        auto feature = Extract(img);
        features.row(std::distance(data.begin(), it)) = feature;
    }
    return features;
}

auto HOGFeatureExtraction::Extract(const cv::Mat &img) -> Eigen::VectorXd {
    std::vector<float> descriptors;
    hog_.compute(img, descriptors);
    Eigen::Map<Eigen::VectorXf> ret(descriptors.data(), descriptors.size());
    return ret.cast<double>();
}

auto HOGFeatureExtraction::BatchExtract(const std::vector<cv::Mat> &data)
    -> Eigen::MatrixXd {
    Eigen::MatrixXd features;
    for (auto it = data.begin(); it != data.end(); it++) {
        auto img = Compress(*it);
        auto feature = Extract(img);
        if (features.size() == 0) {
            features = Eigen::MatrixXd::Zero(data.size(), feature.size());
        }
        features.row(std::distance(data.begin(), it)) = feature;
    }
    return features;
}

auto LBPFeatureExtraction::GetLBPMat(const cv::Mat &img) -> cv::Mat {
    // 使用32位整数Mat存储LBP码，避免高维LBP信息截断
    cv::Mat dest = cv::Mat::zeros(img.size(), CV_32S);
    // 对图像中的每个像素计算LBP值
    for (int i = 0; i != img.rows; ++i) {
        for (int j = 0; j != img.cols; ++j) {
            if (i < radius_ || i >= img.rows - radius_ || j < radius_ ||
                j >= img.cols - radius_) {
                continue;
            }

            double center = img.at<uchar>(i, j);
            std::bitset<32> code;  // 使用32位bitset存储LBP码

            // 计算基本LBP码
            for (int p = 0; p != neighbors_; ++p) {
                double theta = 2 * CV_PI * p / neighbors_;
                double x = j + radius_ * cos(theta),
                       y = i - radius_ * sin(theta);
                double val = BilinearInterpolation(img, x, y);

                if (val > center) {
                    code.set(p);
                }
            }

            uint32_t final_code = 0;

            // 根据LBP类型处理LBP码
            switch (lbp_type_) {
                case LBPType::Circular:
                    // 标准圆形LBP，直接使用完整码
                    final_code = static_cast<uint32_t>(code.to_ulong());
                    break;
                case LBPType::Rotation:
                    // 旋转不变LBP，找到循环移位最小值
                    {
                        uint32_t min_rot =
                            static_cast<uint32_t>(code.to_ulong());
                        for (int r = 1; r < neighbors_; r++) {
                            // 循环右移，保持位数为neighbors_
                            uint32_t rot_val =
                                ((code.to_ulong() >> r) |
                                 (code.to_ulong() << (neighbors_ - r))) &
                                ((1UL << neighbors_) - 1);
                            min_rot = std::min(min_rot, rot_val);
                        }
                        final_code = min_rot;
                    }
                    break;
                case LBPType::Uniform:
                    // 等价模式LBP处理
                    {
                        if (look_up_.find(code.to_ulong()) != look_up_.end()) {
                            final_code = look_up_[code.to_ulong()];
                        } else {
                            // 如果不是等价模式，使用最大值
                            final_code = look_up_.size();
                        }
                    }
                    break;
            }
            dest.at<int>(i, j) = static_cast<int>(final_code);
        }
    }
    return dest;
}


auto LBPFeatureExtraction::Extract(const cv::Mat &img) -> Eigen::VectorXd {
    CV_Assert(img.channels() == 1);

    // 计算直方图的大小
    int hist_size = 0;
    switch (lbp_type_) {
        case LBPType::Circular:
            hist_size = (1 << neighbors_);  // 2^neighbors
            break;
        case LBPType::Rotation:
            hist_size = 4116;  // 旋转不变LBP的直方图大小
            break;
        case LBPType::Uniform:
            hist_size =
                neighbors_ * (neighbors_ - 1) + 3;  // 等价模式LBP的直方图大小
            break;
    }
    // 计算LBP矩阵
    auto dest = GetLBPMat(img);
    auto dest_float = cv::Mat();
    dest.convertTo(dest_float, CV_32F);

    // 计算直方图
    cv::Mat hist;

    // 对于可控维度的LBP，直接使用OpenCV的calcHist
    int channels[] = {0};
    int histSize[] = {hist_size};
    float range[] = {0, static_cast<float>(hist_size)};
    const float *ranges[] = {range};
    cv::calcHist(&dest_float, 1, channels, cv::Mat(), hist, 1, &hist_size, ranges);

    // 归一化直方图
    hist /= dest.total();

    // 转换为Eigen::VectorXd并返回
    Eigen::VectorXd feature(hist_size);
    for (int i = 0; i < hist_size; i++) {
        feature(i) = hist.at<float>(i);
    }
    return feature;
}

auto LBPFeatureExtraction::BatchExtract(const std::vector<cv::Mat> &data)
    -> Eigen::MatrixXd {
    // 确定特征向量的维度
    int feature_dimension = 0;
    switch (lbp_type_) {
        case LBPType::Circular:
            feature_dimension = (1 << neighbors_);  // 2^neighbors
            break;
        case LBPType::Rotation:
            feature_dimension = 4116;  // 旋转不变LBP的直方图大小
            break;
        case LBPType::Uniform:
            feature_dimension =
                neighbors_ * (neighbors_ - 1) + 3;  // 等价模式LBP的直方图大小
            break;
    }

    Eigen::MatrixXd features(data.size(), feature_dimension);

    for (auto it = data.begin(); it != data.end(); it++) {
        auto img = Compress(*it);
        auto feature = Extract(img);
        features.row(std::distance(data.begin(), it)) = feature;
    }

    return features;
}