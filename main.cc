#include "feature_extraction.h"
#include <iostream>
#include <opencv2/opencv.hpp>

// HOG可视化函数，模仿提供的样例代码
cv::Mat get_hogdescriptor_visual_image(const cv::Mat& origImg,
                                      const Eigen::VectorXd& descriptorValues,
                                      cv::Size winSize,
                                      cv::Size cellSize,
                                      int scaleFactor,
                                      double viz_factor) {
    cv::Mat visual_image;
    cv::resize(origImg, visual_image, 
               cv::Size(origImg.cols * scaleFactor, origImg.rows * scaleFactor));
    
    // 如果是灰度图，转为彩色图
    if (visual_image.channels() == 1) {
        cv::cvtColor(visual_image, visual_image, cv::COLOR_GRAY2BGR);
    }

    int gradientBinSize = 9;  // 梯度方向bin数量
    // 将180°分为9个bin，每个bin有多少弧度
    float radRangeForOneBin = CV_PI / (float)gradientBinSize;

    // 准备数据结构：每个单元格有9个方向/梯度强度
    int cells_in_x_dir = winSize.width / cellSize.width;
    int cells_in_y_dir = winSize.height / cellSize.height;
    int totalnrofcells = cells_in_x_dir * cells_in_y_dir;
    
    // 为梯度强度分配内存
    float*** gradientStrengths = new float**[cells_in_y_dir];
    int** cellUpdateCounter = new int*[cells_in_y_dir];
    
    for (int y = 0; y < cells_in_y_dir; y++) {
        gradientStrengths[y] = new float*[cells_in_x_dir];
        cellUpdateCounter[y] = new int[cells_in_x_dir];
        
        for (int x = 0; x < cells_in_x_dir; x++) {
            gradientStrengths[y][x] = new float[gradientBinSize];
            cellUpdateCounter[y][x] = 0;
            
            for (int bin = 0; bin < gradientBinSize; bin++)
                gradientStrengths[y][x][bin] = 0.0;
        }
    }

    // 块数 = 单元格数 - 1
    // 因为每个单元格上都有一个新块（重叠块！）除了最后一个
    int blocks_in_x_dir = cells_in_x_dir - 1;
    int blocks_in_y_dir = cells_in_y_dir - 1;

    // 计算每个单元格的梯度强度
    int descriptorDataIdx = 0;

    for (int blockx = 0; blockx < blocks_in_x_dir; blockx++) {
        for (int blocky = 0; blocky < blocks_in_y_dir; blocky++) {
            // 每个块包含4个单元格...
            for (int cellNr = 0; cellNr < 4; cellNr++) {
                // 计算相应的单元格编号
                int cellx = blockx;
                int celly = blocky;
                
                if (cellNr == 1) celly++;
                if (cellNr == 2) cellx++;
                if (cellNr == 3) {
                    cellx++;
                    celly++;
                }
                
                for (int bin = 0; bin < gradientBinSize; bin++) {
                    float gradientStrength = 0;
                    if (descriptorDataIdx < descriptorValues.size()) {
                        gradientStrength = descriptorValues[descriptorDataIdx];
                    }
                    descriptorDataIdx++;
                    gradientStrengths[celly][cellx][bin] += gradientStrength;
                } // 对所有bins
                
                // 注意：重叠块会导致这个和被多次更新！
                // 因此我们跟踪单元格被更新的次数，
                // 以计算平均梯度强度
                cellUpdateCounter[celly][cellx]++;
            } // 对所有单元格
        } // 对所有块的x位置
    } // 对所有块的y位置

    // 计算平均梯度强度
    for (int celly = 0; celly < cells_in_y_dir; celly++) {
        for (int cellx = 0; cellx < cells_in_x_dir; cellx++) {
            float NrUpdatesForThisCell = (float)cellUpdateCounter[celly][cellx];
            
            // 计算每个梯度bin方向的平均梯度强度
            if (NrUpdatesForThisCell > 0) {
                for (int bin = 0; bin < gradientBinSize; bin++) {
                    gradientStrengths[celly][cellx][bin] /= NrUpdatesForThisCell;
                }
            }
        }
    }

    std::cout << "descriptorDataIdx = " << descriptorDataIdx << std::endl;

    // 绘制单元格
    for (int celly = 0; celly < cells_in_y_dir; celly++) {
        for (int cellx = 0; cellx < cells_in_x_dir; cellx++) {
            int drawX = cellx * cellSize.width;
            int drawY = celly * cellSize.height;
            
            int mx = drawX + cellSize.width / 2;
            int my = drawY + cellSize.height / 2;
            
            // 绘制单元格边界
            cv::rectangle(visual_image,
                        cv::Point(drawX * scaleFactor, drawY * scaleFactor),
                        cv::Point((drawX + cellSize.width) * scaleFactor,
                                (drawY + cellSize.height) * scaleFactor),
                        cv::Scalar(100, 100, 100),
                        1);
            
            // 在每个单元格中绘制9个梯度强度方向
            for (int bin = 0; bin < gradientBinSize; bin++) {
                float currentGradStrength = gradientStrengths[celly][cellx][bin];
                
                if (currentGradStrength == 0)
                    continue;
                
                float currRad = bin * radRangeForOneBin + radRangeForOneBin / 2;
                
                float dirVecX = std::cos(currRad);
                float dirVecY = std::sin(currRad);
                float maxVecLen = cellSize.width / 2;
                float scale = viz_factor; // 只是一个可视化比例，以便更好地看到线条
                
                // 计算线条坐标
                float x1 = mx - dirVecX * currentGradStrength * maxVecLen * scale;
                float y1 = my - dirVecY * currentGradStrength * maxVecLen * scale;
                float x2 = mx + dirVecX * currentGradStrength * maxVecLen * scale;
                float y2 = my + dirVecY * currentGradStrength * maxVecLen * scale;
                
                // 绘制梯度可视化
                cv::line(visual_image,
                       cv::Point(x1 * scaleFactor, y1 * scaleFactor),
                       cv::Point(x2 * scaleFactor, y2 * scaleFactor),
                       cv::Scalar(0, 0, 255),
                       1);
            } // 对所有bins
        } // 对所有cellx
    } // 对所有celly

    for (int y = 0; y < cells_in_y_dir; y++) {
        for (int x = 0; x < cells_in_x_dir; x++) {
            delete[] gradientStrengths[y][x];
        }
        delete[] gradientStrengths[y];
        delete[] cellUpdateCounter[y];
    }
    delete[] gradientStrengths;
    delete[] cellUpdateCounter;
    
    return visual_image;
}

auto main() -> int {
    // 读取图像
    cv::Mat img = cv::imread("Lenna.png", cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "Error: Could not read the image." << std::endl;
        return 1;
    }

    // 创建HOG特征提取器
    auto hog_extractor = HOGFeatureExtraction(cv::Size(512, 512));
    
    // 提取HOG特征
    auto hog_features = hog_extractor.Extract(img);
    
    std::cout << "HOG特征维度: " << hog_features.size() << std::endl;
    
    // 创建空白背景用于HOG可视化
    cv::Mat background = cv::Mat::zeros(img.size(), CV_8UC1);
    
    // 在空白背景上可视化HOG特征
    cv::Mat background_hog = get_hogdescriptor_visual_image(
        background, hog_features, 
        cv::Size(512, 512), cv::Size(8, 8), 
        1, 2.0);
    
    cv::imwrite("hog_visualization_background.png", background_hog);
    
    // 在原始图像上可视化HOG特征
    cv::Mat img_hog = get_hogdescriptor_visual_image(
        img, hog_features, 
        cv::Size(512, 512), cv::Size(8, 8), 
        1, 2.0);
    
    cv::imwrite("hog_visualization_image.png", img_hog);
    
    std::cout << "HOG特征可视化已保存为 hog_visualization_background.png 和 hog_visualization_image.png" << std::endl;
    
    return 0;
}
