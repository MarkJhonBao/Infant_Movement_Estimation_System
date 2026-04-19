#pragma once
#include "DetectorBase.h"
#include <memory>

#ifdef HAVE_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

// ─────────────────────────────────────────────────────────────────────────────
//  ONNX Runtime YOLOv8 detector
//  Falls back to a stub implementation when ONNX Runtime is not linked.
// ─────────────────────────────────────────────────────────────────────────────
class OnnxDetector : public DetectorBase {
public:
    OnnxDetector();
    ~OnnxDetector() override;

    bool loadModel(const QString& modelPath) override;
    QVector<DetectionResult> detect(const QImage& frame) override;
    QString backendName() const override { return QStringLiteral("ONNX Runtime"); }

    // Input tensor size (default 640×640 for YOLOv8)
    void setInputSize(int w, int h) { m_inputW = w; m_inputH = h; }

    // Provide class names from a labels file (one name per line)
    bool loadLabels(const QString& labelsPath);
    const QStringList& labels() const { return m_labels; }

private:
    QVector<DetectionResult> postProcess(
        const float* output, int rows, int cols,
        int srcW, int srcH);

    int         m_inputW{640};
    int         m_inputH{640};
    QStringList m_labels;

    // 已有成员...
    bool  m_loaded{false};
    float m_confThreshold{0.35f};
    float m_nmsThreshold{0.45f};

#ifdef HAVE_ONNXRUNTIME
    Ort::Env                           m_env;
    std::unique_ptr<Ort::Session>      m_session;
    Ort::SessionOptions                m_sessionOptions;
    std::vector<const char*>           m_inputNames;
    std::vector<const char*>           m_outputNames;
    std::vector<std::string>           m_inputNameStrings;
    std::vector<std::string>           m_outputNameStrings;
#endif
};
