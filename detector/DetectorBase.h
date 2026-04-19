#pragma once
#include <QImage>
#include <QRectF>
#include <QString>
#include <QVector>

// =============================================================================
//  DetectionResult  —  目标检测器输出的单个检测框
//
//  字段顺序与 PoseDetector.cpp 中的聚合初始化保持一致：
//    DetectionResult{bbox, classId, confidence, label}
// =============================================================================
struct DetectionResult {
    QRectF  bbox;            // 检测框（原图像素坐标）
    int     classId{0};      // 类别 ID（0 = person）
    float   confidence{0.f}; // 置信度 [0, 1]
    QString label;           // 类别名称
};

// =============================================================================
//  DetectorBase  —  目标检测器抽象基类
// =============================================================================
class DetectorBase {
public:
    virtual ~DetectorBase() = default;

    // 加载模型（ONNX / TRT engine 等），成功返回 true
    virtual bool loadModel(const QString& modelPath) = 0;

    // 对单帧推理，返回所有检测结果
    virtual QVector<DetectionResult> detect(const QImage& frame) = 0;

    // 返回后端名称，如 "ONNX" / "TensorRT"
    virtual QString backendName() const = 0;
};
