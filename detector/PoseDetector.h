#pragma once
#include "PoseResult.h"
#include "DetectorBase.h"   // for DetectionResult (bounding boxes from obj-det)

#include <QObject>
#include <QImage>
#include <QVector>
#include <memory>
#include <string>
#include <vector>

#ifdef HAVE_TENSORRT
#  include <NvInfer.h>
#  include <NvOnnxParser.h>
#  include <cuda_runtime_api.h>
#endif

// ─────────────────────────────────────────────────────────────────────────────
//  PoseDetector
//
//  封装 RTMPose / mmpose (COCO-17 top-down) ONNX 模型，通过 TensorRT 加速推理。
//
//  推理流水线
//
//  Stub 模式
//  ─────────
//  未定义 HAVE_TENSORRT 时返回合成关键点，供 UI 开发 / 非 GPU 机器测试使用。
// ─────────────────────────────────────────────────────────────────────────────
class PoseDetector : public QObject {
    Q_OBJECT
public:
    explicit PoseDetector(QObject* parent = nullptr);
    ~PoseDetector() override;

    // ── 模型加载 ─────────────────────────────────────────────────────────────
    bool loadModel (const QString& onnxPath);     // 从 ONNX 构建引擎（慢，首次）
    bool loadEngine(const QString& enginePath);   // 加载序列化引擎（快，后续）
    bool saveEngine(const QString& enginePath);   // 将当前引擎序列化到磁盘

    // ── 推理接口 ─────────────────────────────────────────────────────────────
    QVector<PoseResult> detect(const QImage& frame);   // 全图单人
    QVector<PoseResult> detectWithBoxes(              // top-down 多人
        const QImage& frame,
        const QVector<DetectionResult>& boxes);

    // ── 配置 ─────────────────────────────────────────────────────────────────
    void setInputSize(int w, int h)         { m_inputW = w; m_inputH = h; }
    void setFP16(bool enable)               { m_fp16   = enable; }
    void setVisibilityThreshold(float t)    { m_visThresh = t; }
    void setPoseScoreThreshold(float t)     { m_poseScoreThresh = t; }
    void setNumKeypoints(int n)             { m_numKpts = n; }

    int     inputWidth()  const { return m_inputW; }
    int     inputHeight() const { return m_inputH; }
    bool    isLoaded()    const { return m_loaded; }
    QString backendName() const;

private:
    // ── 内部辅助 ─────────────────────────────────────────────────────────────
    struct SimCCPeak { float x, y, score; };
    struct CropInfo { QRectF roiPixels; };

    // 将 frame 中的 roi 区域 letterbox 缩放写入 float CHW 缓冲
    CropInfo prepareInput(const QImage& src,
                          const QRectF&  roi,
                          std::vector<float>& outCHW) const;

    // 对单个 heatmap 通道做 soft-argmax，返回 (x, y, score)
    SimCCPeak decodeSimCCPeak(const float* simccX, const float* simccY) const;

    // 执行一次前向推理，输出写入 m_cpuOutput
    bool runInference(const std::vector<float>& inputCHW);

    // 将 heatmap 输出转换为原图关键点
    QVector<Keypoint> simccToKeypoints(const CropInfo& crop,
                                       int imgW, int imgH) const;

    // ── 状态 ─────────────────────────────────────────────────────────────────
    bool  m_loaded{false};
    int   m_inputW{288};  // 模型实际输入宽度（TopDownAffine image_size=[192,256] 已被覆盖）
    int   m_inputH{384};  // 模型实际输入高度  input_size=(288,384) W×H
    int   m_numKpts{COCOSkeleton::NumKeypoints};
    bool  m_fp16{true};
    float m_visThresh{COCOSkeleton::DefaultVisibilityThreshold};
    float m_poseScoreThresh{COCOSkeleton::DefaultPoseScoreThreshold};

    // SimCC 输出维度（所有后端/Stub 共用）
    int   m_simccXDim{0};
    int   m_simccYDim{0};

#ifdef HAVE_TENSORRT
    struct TRTLogger : public nvinfer1::ILogger {
        void log(Severity s, const char* msg) noexcept override;
    } m_logger;

    std::unique_ptr<nvinfer1::IRuntime>          m_runtime;
    std::unique_ptr<nvinfer1::ICudaEngine>       m_engine;
    std::unique_ptr<nvinfer1::IExecutionContext> m_context;

    std::string m_inputTensorName;
    std::string m_outputTensorNameX;
    std::string m_outputTensorNameY;

    void*  m_gpuInput {nullptr};
    void*  m_gpuOutputX{nullptr};
    void*  m_gpuOutputY{nullptr};
    int    m_outputSizeX{0};
    int    m_outputSizeY{0};
    int    m_outputSize{0};  // 输出张量总 float 数
    int    m_hmW{0};         // heatmap 宽 (= inputW/4)
    int    m_hmH{0};         // heatmap 高 (= inputH/4)

    std::vector<float> m_cpuOutputX;
    std::vector<float> m_cpuOutputY;
    std::vector<float> m_cpuOutput;

    bool buildFromOnnx(const QString& onnxPath);
    bool inspectEngineTensors();
    bool allocateBuffers();
#elif defined(HAVE_ONNXRUNTIME)
    Ort::Env m_onnxEnv;
    std::unique_ptr<Ort::Session> m_onnxSession{nullptr};
    // 用 std::string 持有名称内存，避免 AllocatedStringPtr::release() 后悬空指针
    std::vector<std::string>     m_onnxInputNamesBuf;
    std::vector<std::string>     m_onnxOutputNamesBuf;
    std::vector<const char*>     m_onnxInputNames;    // 指向上面两个 buf
    std::vector<const char*>     m_onnxOutputNames;
    std::vector<int64_t>         m_onnxInputShape;    // 固定形状（动态轴替换为实际值）
#else
    // stub 缓冲
    std::vector<float> m_cpuOutput;
    int m_hmW{48};
    int m_hmH{64};
    int m_outputSize{0};
#endif
};
