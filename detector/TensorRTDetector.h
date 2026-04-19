#pragma once
#include "DetectorBase.h"
#include <memory>

#ifdef HAVE_TENSORRT
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#endif

// ─────────────────────────────────────────────────────────────────────────────
//  TensorRT YOLOv8 detector
//  Parses an ONNX model via NvOnnxParser and runs FP16 inference on GPU.
//  Falls back to a stub when TensorRT / CUDA is not available.
// ─────────────────────────────────────────────────────────────────────────────
class TensorRTDetector : public DetectorBase {
public:
    TensorRTDetector();
    ~TensorRTDetector() override;

    bool loadModel(const QString& modelPath) override;
    QVector<DetectionResult> detect(const QImage& frame) override;
    QString backendName() const override { return QStringLiteral("TensorRT"); }

    void setInputSize(int w, int h) { m_inputW = w; m_inputH = h; }
    void setFP16(bool enable)       { m_fp16 = enable; }

    // Optionally load a serialised engine cache to speed up startup
    bool loadEngine(const QString& enginePath);
    bool saveEngine(const QString& enginePath);

    void setLabels(const QStringList& labels) { m_labels = labels; }
    const QStringList& labels() const         { return m_labels; }

private:
    int         m_inputW{640};
    int         m_inputH{640};
    bool        m_fp16{true};
    QStringList m_labels;

    // 已有成员...
    bool  m_loaded{false};
    float m_confThreshold{0.35f};
    float m_nmsThreshold{0.45f};

#ifdef HAVE_TENSORRT
    struct TRTLogger : public nvinfer1::ILogger {
        void log(Severity s, const char* msg) noexcept override;
    } m_logger;

    std::unique_ptr<nvinfer1::IRuntime>        m_runtime;
    std::unique_ptr<nvinfer1::ICudaEngine>     m_engine;
    std::unique_ptr<nvinfer1::IExecutionContext> m_context;
    void* m_gpuInput{nullptr};
    void* m_gpuOutput{nullptr};
    std::vector<float> m_cpuOutput;
    int m_outputSize{0};

    QVector<DetectionResult> postProcess(int srcW, int srcH);
#endif
};
