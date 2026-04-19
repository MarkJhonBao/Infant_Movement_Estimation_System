#include "OnnxDetector.h"
#include <QFile>
#include <QTextStream>
#include <QDebug>
#include <algorithm>
#include <cmath>

// ─── Utility: non-maximum suppression ────────────────────────────────────────
static float iou(const QRectF& a, const QRectF& b) {
    QRectF inter = a.intersected(b);
    if (inter.isEmpty()) return 0.f;
    float interArea = static_cast<float>(inter.width() * inter.height());
    float unionArea = static_cast<float>(a.width()*a.height() + b.width()*b.height()) - interArea;
    return unionArea > 0 ? interArea / unionArea : 0.f;
}

static QVector<DetectionResult> nms(QVector<DetectionResult> dets, float iouThresh) {
    std::sort(dets.begin(), dets.end(),
              [](const DetectionResult& a, const DetectionResult& b){
                  return a.confidence > b.confidence;
              });
    QVector<DetectionResult> result;
    QVector<bool> suppressed(dets.size(), false);
    for (int i = 0; i < dets.size(); ++i) {
        if (suppressed[i]) continue;
        result.push_back(dets[i]);
        for (int j = i+1; j < dets.size(); ++j) {
            if (!suppressed[j] && dets[i].classId == dets[j].classId)
                if (iou(dets[i].bbox, dets[j].bbox) > iouThresh)
                    suppressed[j] = true;
        }
    }
    return result;
}

// ─── Constructor / Destructor ─────────────────────────────────────────────────
OnnxDetector::OnnxDetector()
#ifdef HAVE_ONNXRUNTIME
    : m_env(ORT_LOGGING_LEVEL_WARNING, "HospitalDashboard")
#endif
{}

OnnxDetector::~OnnxDetector() = default;

// ─── loadLabels ───────────────────────────────────────────────────────────────
bool OnnxDetector::loadLabels(const QString& labelsPath) {
    QFile f(labelsPath);
    if (!f.open(QIODevice::ReadOnly | QIODevice::Text)) return false;
    QTextStream in(&f);
    m_labels.clear();
    while (!in.atEnd())
        m_labels.append(in.readLine().trimmed());
    return !m_labels.isEmpty();
}

// ─── loadModel ────────────────────────────────────────────────────────────────
bool OnnxDetector::loadModel(const QString& modelPath) {
#ifdef HAVE_ONNXRUNTIME
    try {
        m_sessionOptions.SetIntraOpNumThreads(4);
        m_sessionOptions.SetGraphOptimizationLevel(
            GraphOptimizationLevel::ORT_ENABLE_ALL);

#ifdef _WIN32
        m_session = std::make_unique<Ort::Session>(
            m_env, modelPath.toStdWString().c_str(), m_sessionOptions);
#else
        m_session = std::make_unique<Ort::Session>(
            m_env, modelPath.toStdString().c_str(), m_sessionOptions);
#endif

        Ort::AllocatorWithDefaultOptions allocator;
        size_t numInputs  = m_session->GetInputCount();
        size_t numOutputs = m_session->GetOutputCount();

        m_inputNameStrings.clear();
        m_outputNameStrings.clear();
        m_inputNames.clear();
        m_outputNames.clear();

        for (size_t i = 0; i < numInputs; ++i) {
            auto name = m_session->GetInputNameAllocated(i, allocator);
            m_inputNameStrings.push_back(std::string(name.get()));
        }
        for (size_t i = 0; i < numOutputs; ++i) {
            auto name = m_session->GetOutputNameAllocated(i, allocator);
            m_outputNameStrings.push_back(std::string(name.get()));
        }
        for (auto& s : m_inputNameStrings)  m_inputNames.push_back(s.c_str());
        for (auto& s : m_outputNameStrings) m_outputNames.push_back(s.c_str());

        m_loaded = true;
        qInfo() << "[OnnxDetector] Model loaded:" << modelPath;
        return true;
    } catch (const Ort::Exception& e) {
        qWarning() << "[OnnxDetector] Load failed:" << e.what();
        m_loaded = false;
        return false;
    }
#else
    Q_UNUSED(modelPath)
    qWarning() << "[OnnxDetector] Built without ONNX Runtime — using stub.";
    m_loaded = true;   // stub: pretend loaded
    return true;
#endif
}

// ─── detect ──────────────────────────────────────────────────────────────────
QVector<DetectionResult> OnnxDetector::detect(const QImage& frame) {
#ifdef HAVE_ONNXRUNTIME
    if (!m_session) return {};

    // Pre-process: resize + RGB float32 CHW normalised to [0,1]
    QImage rgb = frame.scaled(m_inputW, m_inputH, Qt::IgnoreAspectRatio,
                              Qt::SmoothTransformation)
                      .convertToFormat(QImage::Format_RGB888);

    std::vector<float> inputTensor(3 * m_inputW * m_inputH);
    for (int c = 0; c < 3; ++c)
        for (int h = 0; h < m_inputH; ++h)
            for (int w = 0; w < m_inputW; ++w)
                inputTensor[c*m_inputH*m_inputW + h*m_inputW + w] =
                    rgb.constBits()[h*m_inputW*3 + w*3 + c] / 255.f;

    std::vector<int64_t> inputShape = {1, 3, m_inputH, m_inputW};
    Ort::MemoryInfo memInfo =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value inputOrtTensor = Ort::Value::CreateTensor<float>(
        memInfo, inputTensor.data(), inputTensor.size(),
        inputShape.data(), inputShape.size());

    auto outputs = m_session->Run(
        Ort::RunOptions{nullptr},
        m_inputNames.data(),  &inputOrtTensor, 1,
        m_outputNames.data(), m_outputNames.size());

    // YOLOv8 output: [1, num_classes+4, num_boxes]
    auto& out = outputs[0];
    auto  shape = out.GetTensorTypeAndShapeInfo().GetShape();
    int   rows = static_cast<int>(shape[1]);   // num_classes+4
    int   cols = static_cast<int>(shape[2]);   // num_boxes
    const float* data = out.GetTensorData<float>();

    return nms(postProcess(data, rows, cols, frame.width(), frame.height()),
               m_nmsThreshold);
#else
    // ── STUB: generate synthetic detections for UI testing ──────────────────
    Q_UNUSED(frame)
    static int tick = 0; ++tick;
    QVector<DetectionResult> results;
    if (tick % 3 == 0) {
        DetectionResult d;
        d.classId   = 0;
        d.label     = m_labels.isEmpty() ? "医疗耗材" : m_labels[0];
        d.confidence = 0.91f;
        d.bbox      = QRectF(0.15, 0.20, 0.35, 0.45);
        results.push_back(d);
    }
    if (tick % 5 == 0) {
        DetectionResult d;
        d.classId   = 1;
        d.label     = m_labels.size() > 1 ? m_labels[1] : "注射器";
        d.confidence = 0.83f;
        d.bbox      = QRectF(0.55, 0.30, 0.28, 0.38);
        results.push_back(d);
    }
    return results;
#endif
}

// ─── postProcess ─────────────────────────────────────────────────────────────
QVector<DetectionResult> OnnxDetector::postProcess(
    const float* data, int rows, int cols, int srcW, int srcH)
{
    // rows = 4 + num_classes, cols = num_proposals
    int numClasses = rows - 4;
    QVector<DetectionResult> results;

    for (int col = 0; col < cols; ++col) {
        // cx, cy, w, h (normalised to input size)
        float cx = data[0*cols + col];
        float cy = data[1*cols + col];
        float bw = data[2*cols + col];
        float bh = data[3*cols + col];

        // Find best class
        int   bestCls  = 0;
        float bestConf = 0.f;
        for (int c = 0; c < numClasses; ++c) {
            float conf = data[(4+c)*cols + col];
            if (conf > bestConf) { bestConf = conf; bestCls = c; }
        }
        if (bestConf < m_confThreshold) continue;

        // Convert to normalised [0,1] box (x,y = top-left)
        float x1 = (cx - bw/2.f) / static_cast<float>(m_inputW);
        float y1 = (cy - bh/2.f) / static_cast<float>(m_inputH);
        float nw  = bw / static_cast<float>(m_inputW);
        float nh  = bh / static_cast<float>(m_inputH);

        DetectionResult det;
        det.bbox       = QRectF(qBound(0.f,x1,1.f), qBound(0.f,y1,1.f),
                                qBound(0.f,nw,1.f),  qBound(0.f,nh,1.f));
        det.classId    = bestCls;
        det.confidence = bestConf;
        det.label      = (bestCls < m_labels.size())
                          ? m_labels[bestCls]
                          : QString("class_%1").arg(bestCls);
        results.push_back(det);
    }
    Q_UNUSED(srcW) Q_UNUSED(srcH)
    return nms(results, m_nmsThreshold);
}
