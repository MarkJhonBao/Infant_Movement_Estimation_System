#include "TensorRTDetector.h"
#include <QDebug>
#include <QFile>

// ─── iou helper (shared logic, duplicated here to avoid linking issues) ───────
static float trtIou(const QRectF& a, const QRectF& b) {
    QRectF inter = a.intersected(b);
    if (inter.isEmpty()) return 0.f;
    float interArea = static_cast<float>(inter.width() * inter.height());
    float unionArea = static_cast<float>(
        a.width()*a.height() + b.width()*b.height()) - interArea;
    return unionArea > 0 ? interArea / unionArea : 0.f;
}

static QVector<DetectionResult> trtNms(QVector<DetectionResult> dets, float iouThresh) {
    std::sort(dets.begin(), dets.end(),
              [](const DetectionResult& a, const DetectionResult& b){
                  return a.confidence > b.confidence; });
    QVector<DetectionResult> result;
    QVector<bool> sup(dets.size(), false);
    for (int i = 0; i < dets.size(); ++i) {
        if (sup[i]) continue;
        result.push_back(dets[i]);
        for (int j = i+1; j < dets.size(); ++j)
            if (!sup[j] && dets[i].classId == dets[j].classId)
                if (trtIou(dets[i].bbox, dets[j].bbox) > iouThresh)
                    sup[j] = true;
    }
    return result;
}

// ═════════════════════════════════════════════════════════════════════════════
#ifdef HAVE_TENSORRT
// ─── TRTLogger ───────────────────────────────────────────────────────────────
void TensorRTDetector::TRTLogger::log(Severity s, const char* msg) noexcept {
    if (s <= Severity::kWARNING)
        qWarning() << "[TensorRT]" << msg;
}

// ─── Constructor / Destructor ─────────────────────────────────────────────────
TensorRTDetector::TensorRTDetector() {
    m_runtime.reset(nvinfer1::createInferRuntime(m_logger));
}

TensorRTDetector::~TensorRTDetector() {
    if (m_gpuInput)  cudaFree(m_gpuInput);
    if (m_gpuOutput) cudaFree(m_gpuOutput);
}

// ─── loadModel (build engine from ONNX) ──────────────────────────────────────
bool TensorRTDetector::loadModel(const QString& modelPath) {
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(
        nvinfer1::createInferBuilder(m_logger));
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(
        builder->createNetworkV2(
            1U << static_cast<uint32_t>(
                nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)));
    auto parser = std::unique_ptr<nvonnxparser::IParser>(
        nvonnxparser::createParser(*network, m_logger));

    if (!parser->parseFromFile(modelPath.toStdString().c_str(),
                               static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        qWarning() << "[TensorRT] ONNX parse failed:" << modelPath;
        return false;
    }

    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(
        builder->createBuilderConfig());
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30);  // 1 GB
    if (m_fp16 && builder->platformHasFastFp16())
        config->setFlag(nvinfer1::BuilderFlag::kFP16);

    m_engine.reset(builder->buildEngineWithConfig(*network, *config));
    if (!m_engine) { qWarning() << "[TensorRT] Engine build failed"; return false; }
    m_context.reset(m_engine->createExecutionContext());

    // Allocate GPU buffers
    int inSize  = 3 * m_inputH * m_inputW;
    int numCls  = network->getOutput(0)->getDimensions().d[1] - 4;
    m_outputSize = (numCls + 4) * 8400;  // default 8400 anchors YOLOv8

    cudaMalloc(&m_gpuInput,  inSize       * sizeof(float));
    cudaMalloc(&m_gpuOutput, m_outputSize * sizeof(float));
    m_cpuOutput.resize(m_outputSize);

    m_loaded = true;
    qInfo() << "[TensorRT] Engine ready, FP16=" << m_fp16;
    return true;
}

// ─── loadEngine / saveEngine ─────────────────────────────────────────────────
bool TensorRTDetector::loadEngine(const QString& enginePath) {
    QFile f(enginePath);
    if (!f.open(QIODevice::ReadOnly)) return false;
    QByteArray blob = f.readAll();
    m_engine.reset(m_runtime->deserializeCudaEngine(blob.constData(), blob.size()));
    if (!m_engine) return false;
    m_context.reset(m_engine->createExecutionContext());

    int inSize  = 3 * m_inputH * m_inputW;
    cudaMalloc(&m_gpuInput,  inSize       * sizeof(float));
    cudaMalloc(&m_gpuOutput, m_outputSize * sizeof(float));
    m_cpuOutput.resize(m_outputSize);

    m_loaded = true;
    return true;
}

bool TensorRTDetector::saveEngine(const QString& enginePath) {
    if (!m_engine) return false;
    auto serialised = std::unique_ptr<nvinfer1::IHostMemory>(
        m_engine->serialize());
    QFile f(enginePath);
    if (!f.open(QIODevice::WriteOnly)) return false;
    f.write(static_cast<const char*>(serialised->data()),
            static_cast<qint64>(serialised->size()));
    return true;
}

// ─── detect ──────────────────────────────────────────────────────────────────
QVector<DetectionResult> TensorRTDetector::detect(const QImage& frame) {
    if (!m_context) return {};

    // Pre-process
    QImage rgb = frame.scaled(m_inputW, m_inputH,
                              Qt::IgnoreAspectRatio, Qt::SmoothTransformation)
                      .convertToFormat(QImage::Format_RGB888);

    std::vector<float> host(3 * m_inputW * m_inputH);
    for (int c = 0; c < 3; ++c)
        for (int h = 0; h < m_inputH; ++h)
            for (int w = 0; w < m_inputW; ++w)
                host[c*m_inputH*m_inputW + h*m_inputW + w] =
                    rgb.constBits()[h*m_inputW*3 + w*3 + c] / 255.f;

    cudaMemcpy(m_gpuInput, host.data(), host.size()*sizeof(float),
               cudaMemcpyHostToDevice);

    void* bindings[] = { m_gpuInput, m_gpuOutput };
    m_context->executeV2(bindings);

    cudaMemcpy(m_cpuOutput.data(), m_gpuOutput,
               m_outputSize*sizeof(float), cudaMemcpyDeviceToHost);

    return trtNms(postProcess(frame.width(), frame.height()), m_nmsThreshold);
}

// ─── postProcess ─────────────────────────────────────────────────────────────
QVector<DetectionResult> TensorRTDetector::postProcess(int /*srcW*/, int /*srcH*/) {
    // YOLOv8 output layout: [1, (4+cls), 8400]
    int numCls  = m_outputSize / 8400 - 4;
    int numBoxes= 8400;
    QVector<DetectionResult> results;

    for (int col = 0; col < numBoxes; ++col) {
        float cx = m_cpuOutput[0*numBoxes + col];
        float cy = m_cpuOutput[1*numBoxes + col];
        float bw = m_cpuOutput[2*numBoxes + col];
        float bh = m_cpuOutput[3*numBoxes + col];

        int   bestCls  = 0;
        float bestConf = 0.f;
        for (int c = 0; c < numCls; ++c) {
            float conf = m_cpuOutput[(4+c)*numBoxes + col];
            if (conf > bestConf) { bestConf = conf; bestCls = c; }
        }
        if (bestConf < m_confThreshold) continue;

        float x1 = (cx - bw/2.f) / m_inputW;
        float y1 = (cy - bh/2.f) / m_inputH;
        float nw = bw / m_inputW;
        float nh = bh / m_inputH;

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
    return results;
}

// ═════════════════════════════════════════════════════════════════════════════
#else  // stub when TensorRT not available
// ─────────────────────────────────────────────────────────────────────────────
TensorRTDetector::TensorRTDetector() = default;
TensorRTDetector::~TensorRTDetector() = default;

bool TensorRTDetector::loadModel(const QString& modelPath) {
    Q_UNUSED(modelPath)
    qWarning() << "[TensorRTDetector] Built without TensorRT — using stub.";
    m_loaded = true;
    return true;
}

bool TensorRTDetector::loadEngine(const QString&) { return false; }
bool TensorRTDetector::saveEngine(const QString&) { return false; }

QVector<DetectionResult> TensorRTDetector::detect(const QImage& frame) {
    Q_UNUSED(frame)
    static int tick = 0; ++tick;
    QVector<DetectionResult> results;
    if (tick % 4 == 0) {
        DetectionResult d;
        d.classId   = 0;
        d.label     = m_labels.isEmpty() ? "手术耗材" : m_labels[0];
        d.confidence= 0.95f;
        d.bbox      = QRectF(0.20, 0.15, 0.40, 0.50);
        results.push_back(d);
    }
    return results;
}
#endif
