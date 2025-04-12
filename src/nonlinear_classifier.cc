#include "nonlinear_classifier.h"

#include <Eigen/Dense>
#include <cmath>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <optional>
#include <utility>
#include <vector>

#include "threadpool.h"
#include "utils.h"

auto KNNClassifier::Fit(
    const std::vector<std::pair<Eigen::VectorXd, int>> &train_data) -> void {
    // 寻找方差最大的维度
    double max_var = 0;
    int split = 0;
    for (int i = 0; i < train_data[0].first.size(); i++) {
        double mean = 0;
        for (int j = 0; j < train_data.size(); j++) {
            mean += train_data[j].first(i);
        }
        mean /= train_data.size();
        double var = 0;
        for (int j = 0; j < train_data.size(); j++) {
            var += (train_data[j].first(i) - mean) *
                   (train_data[j].first(i) - mean);
        }
        if (var > max_var) {
            max_var = var;
            split = i;
        }
    }
    // 递归构建kd树
    std::function<void(std::unique_ptr<KNNTreeNode<Eigen::VectorXd>> &,
                       const std::vector<std::pair<Eigen::VectorXd, int>> &,
                       int)>
        build = [&](std::unique_ptr<KNNTreeNode<Eigen::VectorXd>> &current,
                    const std::vector<std::pair<Eigen::VectorXd, int>> &data,
                    int split) {
            if (data.size() == 0) {
                return;
            }
            // 寻找中位数
            auto sorted_data = data;
            int mid = sorted_data.size() / 2;
            std::sort(sorted_data.begin(),
                      sorted_data.end(),
                      [split](const std::pair<Eigen::VectorXd, int> &a,
                              const std::pair<Eigen::VectorXd, int> &b) {
                          return a.first(split) < b.first(split);
                      });
            current = std::make_unique<KNNTreeNode<Eigen::VectorXd>>(
                sorted_data[mid].first, split, sorted_data[mid].second);
            std::vector<std::pair<Eigen::VectorXd, int>> left_data(
                sorted_data.begin(), sorted_data.begin() + mid);
            std::vector<std::pair<Eigen::VectorXd, int>> right_data(
                sorted_data.begin() + mid + 1, sorted_data.end());
            build(current->left, left_data, (split + 1) % data[0].first.size());
            build(
                current->right, right_data, (split + 1) % data[0].first.size());
        };
    build(root_, train_data, split);
}

auto KNNClassifier::Predict(const std::vector<Eigen::VectorXd> &test_data) const
    -> std::vector<int> {
    std::vector<int> result;
    // 递归搜索kd树
    std::function<void(const std::unique_ptr<KNNTreeNode<Eigen::VectorXd>> &,
                       const Eigen::VectorXd &,
                       std::vector<std::pair<double, int>> &,
                       int)>
        search =
            [&](const std::unique_ptr<KNNTreeNode<Eigen::VectorXd>> &current,
                const Eigen::VectorXd &data,
                std::vector<std::pair<double, int>> &heap,
                int k) {
                if (current == nullptr) {
                    return;
                }
                auto dist = (current->data - data).norm();
                if (heap.size() < k) {
                    heap.push_back({dist, current->label});
                    std::push_heap(heap.begin(),
                                   heap.end(),
                                   std::greater<std::pair<double, int>>());
                } else {
                    if (dist < heap.front().first) {
                        std::pop_heap(heap.begin(),
                                      heap.end(),
                                      std::greater<std::pair<double, int>>());
                        heap.pop_back();
                        heap.push_back({dist, current->label});
                        std::push_heap(heap.begin(),
                                       heap.end(),
                                       std::greater<std::pair<double, int>>());
                    }
                }
                auto split_dist = std::abs(data(current->split) -
                                           current->data(current->split));
                if (heap.size() < k || split_dist < heap.front().first) {
                    if (data(current->split) < current->data(current->split)) {
                        search(current->left, data, heap, k);
                        if (heap.size() < k ||
                            split_dist < heap.front().first) {
                            search(current->right, data, heap, k);
                        }
                    } else {
                        search(current->right, data, heap, k);
                        if (heap.size() < k ||
                            split_dist < heap.front().first) {
                            search(current->left, data, heap, k);
                        }
                    }
                }
            };
    for (int i = 0; i < test_data.size(); i++) {
        std::vector<std::pair<double, int>> heap;
        search(root_, test_data[i], heap, k_);
        std::map<int, int> count;
        for (int j = 0; j < heap.size(); j++) {
            count[heap[j].second]++;
        }
        int max_count = 0;
        int max_label = 0;
        for (auto &pair : count) {
            if (pair.second > max_count) {
                max_count = pair.second;
                max_label = pair.first;
            }
        }
        result.push_back(max_label);
    }
    return result;
}

int SVMClassifier::kEpochs = 1000;
double SVMClassifier::kEpsilon = 0.0001;

auto SVMClassifier::Train(const cv::Mat &train_data,
                          const std::vector<int> &train_label,
                          const cv::Mat &validate_data,
                          const std::vector<int> &validate_label,
                          const std::vector<double> &c_range,
                          const std::vector<double> &gamma_range) -> void {
    auto train_label_mat = cv::Mat(train_label, true);
    auto best_param = std::pair<double, double>(0, 0);
    auto best_accuracy = 0.0;
    ThreadPool pool(std::thread::hardware_concurrency());
    std::vector<std::future<void>> futures;
    std::mutex mtx;
    for (auto c : c_range) {
        for (auto gamma : gamma_range) {
            futures.emplace_back(pool.Enqueue([&, c, gamma]() {
                cv::Ptr<cv::ml::SVM> temp;
                temp = cv::ml::SVM::create();
                temp->setType(cv::ml::SVM::C_SVC);
                temp->setKernel(cv::ml::SVM::RBF);
                temp->setC(c);
                temp->setGamma(gamma);
                temp->setTermCriteria(cv::TermCriteria(
                    cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS,
                    kEpochs,
                    kEpsilon));
                temp->train(train_data, cv::ml::ROW_SAMPLE, train_label_mat);
                auto result = Predict(validate_data, temp);
                auto accuracy = CalcAccuracy(result, validate_label);
                std::lock_guard<std::mutex> lock(mtx);
                if (accuracy >= best_accuracy) {
                    best_accuracy = accuracy;
                    best_param = {c, gamma};
                    svm_ = temp;
                }
            }));
        }
    }
    for (auto &future : futures) {
        future.get();
    }
}

auto SVMClassifier::Predict(const cv::Mat &test_data_mat,
                            const std::optional<cv::Ptr<cv::ml::SVM>> &svm)
    -> std::vector<int> {
    cv::Mat result;
    if (svm.has_value()) {
        svm.value()->predict(test_data_mat, result);
    } else {
        svm_->predict(test_data_mat, result);
    }
    return std::vector<int>(result.begin<float>(), result.end<float>());
}

auto SVMClassifier::PredictRaw(const cv::Mat &test_data_mat)
    -> std::vector<float> {
    assert(svm_ != nullptr);
    cv::Mat result;
    svm_->predict(test_data_mat, result, cv::ml::StatModel::RAW_OUTPUT);
    if (result.empty() || result.type() != CV_32F ||
        result.rows != test_data_mat.rows || result.cols != 1) {
        throw std::runtime_error("SVM prediction failed.");
    }
    return std::vector<float>(result.begin<float>(), result.end<float>());
}


auto DecisionTree::CalcEntropy(const std::vector<std::string> &labels)
    -> double {
    std::map<std::string, int> count;
    for (auto &label : labels) {
        count[label]++;
    }
    double entropy = 0;
    for (auto &pair : count) {
        double p = static_cast<double>(pair.second) / labels.size();
        entropy -= p > 0 ? p * std::log2(p) : 0;
    }
    return entropy;
}

auto DecisionTree::Build(std::string csv_filepath) -> void {
    CSV data(csv_filepath);
    std::function<void(CSV &, std::unique_ptr<DecisionTreeNode> &)> BuildTree =
        [&](CSV &data, std::unique_ptr<DecisionTreeNode> &current) {
            if (data.GetRowCount() <= 1) {
                return;
            }
            // 如果所有数据都属于同一类别
            bool same_label = true;
            for (int i = 1; i < data.GetRowCount(); i++) {
                if (data.GetCell(i, data.GetColumnCount() - 1) !=
                    data.GetCell(1, data.GetColumnCount() - 1)) {
                    same_label = false;
                    break;
                }
            }
            if (same_label) {
                current = std::make_unique<DecisionTreeNode>(
                    "", data.GetCell(1, data.GetColumnCount() - 1));
                return;
            }
            // 如果没有属性可用
            if (data.GetColumnCount() == 1) {
                int max_count = 0;
                std::string max_label;
                std::map<std::string, int> count;
                for (int i = 1; i < data.GetRowCount(); i++) {
                    count[data.GetCell(i, data.GetColumnCount() - 1)]++;
                }
                for (auto &pair : count) {
                    if (pair.second > max_count) {
                        max_count = pair.second;
                        max_label = pair.first;
                    }
                }
                current = std::make_unique<DecisionTreeNode>("", max_label);
                return;
            }
            // 寻找最佳属性，使用信息增益比(C4.5算法)
            double max_gain_ratio = 0;
            std::string best_attribute;
            auto attributes = data.GetColumnNames();
            attributes.pop_back();
            double entropy = CalcEntropy(data.GetColumn(attributes.size()));
            for (auto &attribute : attributes) {
                double gain = entropy;
                std::map<std::string, std::vector<std::string>> split_data;
                for (int i = 0; i < data.GetRowCount() - 1; i++) {
                    split_data[data.GetCell(i, attribute)].push_back(
                        data.GetCell(i, attributes.size()));
                }
                for (auto &pair : split_data) {
                    double sub_entropy = CalcEntropy(pair.second) *
                                         pair.second.size() /
                                         (data.GetRowCount() - 1);
                    gain -= sub_entropy;
                }
                double split_info = CalcEntropy(data.GetColumn(attribute));
                double gain_ratio = gain / split_info;
                if (gain_ratio > max_gain_ratio) {
                    max_gain_ratio = gain_ratio;
                    best_attribute = attribute;
                }
            }
            // 递归构建决策树
            current = std::make_unique<DecisionTreeNode>(best_attribute, "");
            std::map<std::string, std::vector<std::vector<std::string>>>
                split_data;
            auto column = data.GetColumn(best_attribute);
            data.RemoveColumn(best_attribute);
            auto header = data.GetColumnNames();
            for (int i = 0; i < data.GetRowCount() - 1; i++) {
                if (split_data.find(column[i]) == split_data.end()) {
                    split_data[column[i]] =
                        std::vector<std::vector<std::string>>{header};
                }
                split_data[column[i]].push_back(data.GetRow(i));
            }
            for (auto &pair : split_data) {
                auto child = std::make_unique<DecisionTreeNode>("", "");
                CSV child_data(pair.second);
                BuildTree(child_data, child);
                current->children[pair.first] = std::move(child);
            }
        };
    BuildTree(data, root_);
}

auto DecisionTree::Predict(const CSV &test_data) const
    -> std::vector<std::string> {
    std::vector<std::string> result;
    std::function<void(const CSV &,
                       std::vector<std::string> &,
                       const std::unique_ptr<DecisionTreeNode> &)>
        PredictTree = [&](const CSV &test_data,
                          std::vector<std::string> &result,
                          const std::unique_ptr<DecisionTreeNode> &current) {
            if (current->children.size() == 0) {
                result.push_back(current->label);
                return;
            }
            auto attribute = current->attribute;
            auto index = test_data.GetColumnIndex(attribute);
            for (int i = 1; i < test_data.GetRowCount(); i++) {
                auto value = test_data.GetCell(i, index);
                if (current->children.find(value) != current->children.end()) {
                    PredictTree(test_data, result, current->children.at(value));
                } else {
                    result.push_back(current->label);
                }
            }
        };
    PredictTree(test_data, result, root_);
    return result;
}