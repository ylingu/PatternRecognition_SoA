module;

#include <filesystem>
#include <print>
#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include <pugixml.hpp>


export module xml_parser;


// 导出基本数据结构
export struct BoundingBox {
    std::string label;           // 对象类别标签
    int xmin, ymin, xmax, ymax;  // 边界框坐标
    float score;                 // 置信度分数 (用于检测结果)

    BoundingBox() : xmin(0), ymin(0), xmax(0), ymax(0), score(0.0f) {}

    BoundingBox(const std::string& _label,
                int _xmin,
                int _ymin,
                int _xmax,
                int _ymax,
                float _score = 1.0f)
        : label(_label),
          xmin(_xmin),
          ymin(_ymin),
          xmax(_xmax),
          ymax(_ymax),
          score(_score) {}

    // 获取边界框宽度
    int width() const { return xmax - xmin; }

    // 获取边界框高度
    int height() const { return ymax - ymin; }

    // 获取边界框面积
    int area() const { return width() * height(); }

    // 转换为OpenCV的矩形格式
    cv::Rect toCvRect() const {
        return cv::Rect(xmin, ymin, width(), height());
    }
};

export struct AnnotationData {
    std::string filename;              // 图像文件名
    int width, height, depth;          // 图像尺寸信息
    std::vector<BoundingBox> objects;  // 图像中的所有对象
};

// 导出XML解析器类
export class XmlParser {
public:
    // 解析单个XML文件
    static AnnotationData parseXml(const std::string& filepath) {
        AnnotationData result;
        pugi::xml_document doc;

        // 加载XML文件
        pugi::xml_parse_result xml_result = doc.load_file(filepath.c_str());
        if (!xml_result) {
            std::println("XML解析错误: {} 在文件 {}", xml_result.description(), filepath);
            return result;  // 返回空结果
        }

        // 获取根节点
        pugi::xml_node root = doc.child("annotation");
        if (!root) {
            std::println("找不到annotation节点: {}", filepath);
            return result;
        }

        // 解析基本信息
        result.filename = root.child("filename").text().get();

        // 解析图像尺寸
        pugi::xml_node size_node = root.child("size");
        if (size_node) {
            result.width = size_node.child("width").text().as_int();
            result.height = size_node.child("height").text().as_int();
            result.depth = size_node.child("depth").text().as_int();
        }

        // 解析所有对象
        for (pugi::xml_node object_node = root.child("object"); object_node;
             object_node = object_node.next_sibling("object")) {
            BoundingBox box;
            box.label = object_node.child("name").text().get();

            // 获取边界框信息
            pugi::xml_node bbox_node = object_node.child("bndbox");
            if (bbox_node) {
                box.xmin = bbox_node.child("xmin").text().as_int();
                box.ymin = bbox_node.child("ymin").text().as_int();
                box.xmax = bbox_node.child("xmax").text().as_int();
                box.ymax = bbox_node.child("ymax").text().as_int();
            }

            // 添加到结果中
            result.objects.push_back(box);
        }

        return result;
    }

    // 解析目录中的所有XML文件
    static std::vector<AnnotationData> parseDirectory(
        const std::string& directory) {
        std::vector<AnnotationData> results;

        try {
            // 遍历目录中的所有XML文件
            for (const auto& entry :
                 std::filesystem::directory_iterator(directory)) {
                if (entry.path().extension() == ".xml") {
                    
                    AnnotationData data = parseXml(entry.path().string());
                    if (!data.filename.empty()) {
                        results.push_back(data);
                    }
                }
            }
        } catch (const std::filesystem::filesystem_error& e) {
            std::println("文件系统错误: {}", e.what());
        }

        return results;
    }

    // 将注释数据保存为XML文件
    static bool saveToXml(const AnnotationData& data,
                          const std::string& filepath) {
        pugi::xml_document doc;

        // 创建根节点
        pugi::xml_node root = doc.append_child("annotation");

        // 添加基本信息
        root.append_child("folder").text().set("VOC2007");
        root.append_child("filename").text().set(data.filename.c_str());

        // 添加source信息
        pugi::xml_node source = root.append_child("source");
        source.append_child("database").text().set("My Database");
        source.append_child("annotation").text().set("VOC2007");
        source.append_child("image").text().set("flickr");
        source.append_child("flickrid").text().set("NULL");

        // 添加owner信息
        pugi::xml_node owner = root.append_child("owner");
        owner.append_child("flickrid").text().set("NULL");
        owner.append_child("name").text().set("author");

        // 添加图像尺寸信息
        pugi::xml_node size = root.append_child("size");
        size.append_child("width").text().set(data.width);
        size.append_child("height").text().set(data.height);
        size.append_child("depth").text().set(data.depth);

        root.append_child("segmented").text().set(0);

        // 添加所有对象
        for (const auto& box : data.objects) {
            pugi::xml_node object = root.append_child("object");
            object.append_child("name").text().set(box.label.c_str());
            object.append_child("pose").text().set("Unspecified");
            object.append_child("truncated").text().set(0);
            object.append_child("difficult").text().set(0);

            pugi::xml_node bndbox = object.append_child("bndbox");
            bndbox.append_child("xmin").text().set(box.xmin);
            bndbox.append_child("ymin").text().set(box.ymin);
            bndbox.append_child("xmax").text().set(box.xmax);
            bndbox.append_child("ymax").text().set(box.ymax);
        }

        // 保存到文件
        return doc.save_file(filepath.c_str());
    }
};