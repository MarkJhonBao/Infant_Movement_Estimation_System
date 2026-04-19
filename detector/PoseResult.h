#pragma once
#include <QPointF>
#include <QRectF>
#include <QVector>
#include <QString>

// ─────────────────────────────────────────────────────────────────────────────
//  COCO-17 骨架常量
// ─────────────────────────────────────────────────────────────────────────────
namespace COCOSkeleton {

constexpr int NumKeypoints = 17;
constexpr float DefaultVisibilityThreshold  = 0.3f;
constexpr float DefaultPoseScoreThreshold   = 0.3f;

// COCO keypoint indices
enum KptIdx {
    Nose=0, LeftEye, RightEye, LeftEar, RightEar,
    LeftShoulder, RightShoulder,
    LeftElbow, RightElbow,
    LeftWrist, RightWrist,
    LeftHip, RightHip,
    LeftKnee, RightKnee,
    LeftAnkle, RightAnkle
};

// 骨架连接对 (src, dst)
inline const QVector<QPair<int,int>>& limbs() {
    static const QVector<QPair<int,int>> L = {
        {Nose,       LeftEye},   {Nose,        RightEye},
        {LeftEye,    LeftEar},   {RightEye,    RightEar},
        {LeftShoulder,  RightShoulder},
        {LeftShoulder,  LeftElbow},   {LeftElbow,   LeftWrist},
        {RightShoulder, RightElbow},  {RightElbow,  RightWrist},
        {LeftShoulder,  LeftHip},     {RightShoulder, RightHip},
        {LeftHip,    RightHip},
        {LeftHip,    LeftKnee},   {LeftKnee,    LeftAnkle},
        {RightHip,   RightKnee}, {RightKnee,   RightAnkle}
    };
    return L;
}

inline const char* kptName(int idx) {
    static const char* names[] = {
        "Nose","LeftEye","RightEye","LeftEar","RightEar",
        "LeftShoulder","RightShoulder",
        "LeftElbow","RightElbow",
        "LeftWrist","RightWrist",
        "LeftHip","RightHip",
        "LeftKnee","RightKnee",
        "LeftAnkle","RightAnkle"
    };
    return (idx >= 0 && idx < NumKeypoints) ? names[idx] : "Unknown";
}

} // namespace COCOSkeleton

// ─────────────────────────────────────────────────────────────────────────────
//  Keypoint — 单个关键点（原始图像坐标，归一化或像素均可，由调用方决定）
// ─────────────────────────────────────────────────────────────────────────────
struct Keypoint {
    float x{0.f};          // 像素 x（原图坐标系）
    float y{0.f};          // 像素 y
    float score{0.f};      // heatmap 置信度 [0,1]
    bool  visible{false};  // score > visibilityThreshold

    QPointF toPointF() const { return {x, y}; }
};

// ─────────────────────────────────────────────────────────────────────────────
//  PoseResult — 一个人的完整姿态
// ─────────────────────────────────────────────────────────────────────────────
struct PoseResult {
    QVector<Keypoint> keypoints;   // 长度 = COCOSkeleton::NumKeypoints
    float             poseScore{0.f}; // 所有可见关键点置信度均值
    QRectF            boundingBox;    // 可选：来自上游目标检测框

    bool isValid() const {
        return poseScore > 0.f && !keypoints.isEmpty();
    }
};
