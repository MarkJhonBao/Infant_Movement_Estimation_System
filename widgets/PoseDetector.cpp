#include "PoseDetector.h"
#include <QDebug>
#include <QFile>
#include <QRect>
#include <algorithm>
#include <cmath>
#include <numeric>

#ifdef HAVE_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

namespace {

constexpr float kBBoxPadding = 1.25f;
constexpr float kPixelStd = 200.0f;
constexpr float kChannelMean[3] = {123.675f, 116.28f, 103.53f};
constexpr float kChannelStd[3] = {58.395f, 57.12f, 57.375f};

// RTMPose ImageNet normalization (BGR order in MMPose convention)
constexpr float kRtmposeMean[3] = {123.675f, 116.28f, 103.53f};
constexpr float kRtmposeStd[3]  = {58.395f, 57.12f, 57.375f};

QRectF adjustedPoseRoi(const QRectF& roi, int inputW, int inputH, int imgW, int imgH) {
    QRectF safe = roi.normalized();
    float cx = safe.center().x();
    float cy = safe.center().y();
    float w = std::max(1.0f, (float)safe.width());
    float h = std::max(1.0f, (float)safe.height());

    const float aspect = static_cast<float>(inputW) / static_cast<float>(inputH);
    if (w > aspect * h) {
        h = w / aspect;
    } else {
        w = h * aspect;
    }

    const float scaleX = (w / kPixelStd) * kBBoxPadding;
    const float scaleY = (h / kPixelStd) * kBBoxPadding;
    w = scaleX * kPixelStd;
    h = scaleY * kPixelStd;
    // 确保ROI不超过原始图像边界
    float newX = std::max(0.0f, cx - w * 0.5f);
    float newY = std::max(0.0f, cy - h * 0.5f);
    // 如果调整后的ROI超出原始图像边界，则调整宽度和高度
    w = std::min(w, static_cast<float>(imgW) - newX);
    h = std::min(h, static_cast<float>(imgH) - newY);
    return QRectF(newX, newY, w, h);
}

inline float sampleChannel(const QImage& img, float x, float y, int c) {
    const float px = std::clamp(x, 0.0f, static_cast<float>(img.width() - 1));
    const float py = std::clamp(y, 0.0f, static_cast<float>(img.height() - 1));
    const int x0 = static_cast<int>(std::floor(px));
    const int y0 = static_cast<int>(std::floor(py));
    const int x1 = std::min(x0 + 1, img.width() - 1);
    const int y1 = std::min(y0 + 1, img.height() - 1);
    const float dx = px - x0;
    const float dy = py - y0;

    const uchar* row0 = img.constScanLine(y0);
    const uchar* row1 = img.constScanLine(y1);

    const float v00 = row0[x0 * 3 + c];
    const float v01 = row0[x1 * 3 + c];
    const float v10 = row1[x0 * 3 + c];
    const float v11 = row1[x1 * 3 + c];

    const float top = v00 + (v01 - v00) * dx;
    const float bottom = v10 + (v11 - v10) * dx;
    return top + (bottom - top) * dy;
}

} // namespace

// ═════════════════════════════════════════════════════════════════════════════
#ifdef HAVE_TENSORRT
// ─────────────────────────────────────────────────────────────────────────────

// ── TRTLogger ─────────────────────────────────────────────────────────────────
void PoseDetector::TRTLogger::log(Severity s, const char* msg) noexcept {
    if (s <= Severity::kWARNING)
        qWarning() << "[PoseDetector/TRT]" << msg;
}

// ── Constructor / Destructor ──────────────────────────────────────────────────
PoseDetector::PoseDetector(QObject* parent) : QObject(parent)
#ifdef HAVE_TENSORRT
    , m_runtime(nvinfer1::createInferRuntime(m_logger))
#endif
#ifdef HAVE_ONNXRUNTIME
    , m_onnxEnv(ORT_LOGGING_LEVEL_WARNING, "PoseDetector")
#endif
{}

PoseDetector::~PoseDetector() {
#ifdef HAVE_TENSORRT
    if (m_gpuInput)  cudaFree(m_gpuInput);
    if (m_gpuOutputX) cudaFree(m_gpuOutputX);
    if (m_gpuOutputY) cudaFree(m_gpuOutputY);
#endif
}

QString PoseDetector::backendName() const {
#ifdef HAVE_TENSORRT
    return QStringLiteral("TensorRT-Pose");
#elif defined(HAVE_ONNXRUNTIME)
    return QStringLiteral("ONNXRuntime-Pose");
#else
    return QStringLiteral("Stub-Pose");
#endif
}

// ── loadModel ─────────────────────────────────────────────────────────────────
bool PoseDetector::loadModel(const QString& onnxPath) {
#ifdef HAVE_TENSORRT
    qDebug() << "[PoseDetector] 尝试TensorRT加载:" << onnxPath;
    if (buildFromOnnx(onnxPath)) {
        qDebug() << "[PoseDetector] TensorRT引擎构建成功";
        return true;
    }
#endif

#ifdef HAVE_ONNXRUNTIME
    qDebug() << "[PoseDetector] 尝试ONNXRuntime加载:" << onnxPath;
    try {
        Ort::SessionOptions sessionOptions;
        sessionOptions.SetIntraOpNumThreads(1);
        sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
        
        m_onnxSession = std::make_unique<Ort::Session>(m_onnxEnv, 
            onnxPath.toStdString().c_str(), sessionOptions);
        
        Ort::AllocatorWithDefaultOptions allocator;
        size_t numInputs = m_onnxSession->GetInputCount();
        
        // 获取输入信息
        for (size_t i = 0; i < numInputs; ++i) {
            auto inputName = m_onnxSession->GetInputNameAllocated(i, allocator);
            m_onnxInputNames.emplace_back(inputName.release());
        }
        
        // 获取输出信息 (simcc_x, simcc_y)
        size_t numOutputs = m_onnxSession->GetOutputCount();
        for (size_t i = 0; i < numOutputs; ++i) {
            auto outputName = m_onnxSession->GetOutputNameAllocated(i, allocator);
            m_onnxOutputNames.emplace_back(outputName.release());
        }
        
        // 检查输出张量形状 [1,17,simcc_dim]
        auto inputShapeInfo = m_onnxSession->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
        m_onnxInputShape = inputShapeInfo.GetShape();  // 保存用于推理
        
        // 从第一个输出推断 simcc 维度
        auto outputShapeInfo = m_onnxSession->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo();
        auto outputShape = outputShapeInfo.GetShape();
        m_numKpts = outputShape[1];
        m_simccXDim = outputShape[2];  // RTMPose: simcc_x/y 形状 [1,K,192]
        m_simccYDim = m_simccXDim;
        
        m_cpuOutputX.resize(1 * m_numKpts * m_simccXDim);
        m_cpuOutputY.resize(1 * m_numKpts * m_simccYDim);
        
        m_loaded = true;
        qDebug() << "[PoseDetector] ONNXRuntime成功加载:"
                 << "输入形状" << m_onnxInputShape.size()
                 << "关键点数" << m_numKpts << "SimCC维度" << m_simccXDim;
        return true;
    } catch (const Ort::Exception& e) {
        qWarning() << "[PoseDetector] ONNXRuntime加载失败:" << e.what();
    }
#endif
    
    qWarning() << "[PoseDetector] 所有后端失败，fallback到Stub模式";
    return false;
}

bool PoseDetector::buildFromOnnx(const QString& onnxPath) {
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(
        nvinfer1::createInferBuilder(m_logger));
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(
        builder->createNetworkV2(
            1U << static_cast<uint32_t>(
                nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)));
    auto parser = std::unique_ptr<nvonnxparser::IParser>(
        nvonnxparser::createParser(*network, m_logger));

    if (!parser->parseFromFile(onnxPath.toStdString().c_str(),
                               (int)nvinfer1::ILogger::Severity::kWARNING)) {
        qWarning() << "[PoseDetector] ONNX parse failed:" << onnxPath;
        return false;
    }

    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(
        builder->createBuilderConfig());
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30); // 1 GB
    if (m_fp16 && builder->platformHasFastFp16())
        config->setFlag(nvinfer1::BuilderFlag::kFP16);

    // The downloaded RTMPose ONNX uses dynamic axes, so TensorRT 10 requires
    // an explicit optimization profile even though we run a fixed input size.
    auto* profile = builder->createOptimizationProfile();
    auto* inputTensor = network->getInput(0);
    if (!profile) {
        qWarning() << "[PoseDetector] Failed to create optimization profile";
        return false;
    }
    if (!inputTensor) {
        qWarning() << "[PoseDetector] Missing network input";
        return false;
    }
    const char* inputName = inputTensor->getName();
    nvinfer1::Dims4 inputDims{1, 3, m_inputH, m_inputW};
    if (!profile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMIN, inputDims) ||
        !profile->setDimensions(inputName, nvinfer1::OptProfileSelector::kOPT, inputDims) ||
        !profile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMAX, inputDims)) {
        qWarning() << "[PoseDetector] Failed to set optimization profile";
        return false;
    }
    config->addOptimizationProfile(profile);

    m_engine.reset(builder->buildEngineWithConfig(*network, *config));
    if (!m_engine) {
        qWarning() << "[PoseDetector] Engine build failed";
        return false;
    }
    m_context.reset(m_engine->createExecutionContext());

    // 推导 heatmap 尺寸：输出形状 [1, numKpts, hmH, hmW]
    if (!inspectEngineTensors()) return false;
    return allocateBuffers();
}

// ── loadEngine / saveEngine ───────────────────────────────────────────────────
bool PoseDetector::loadEngine(const QString& enginePath) {
    QFile f(enginePath);
    if (!f.open(QIODevice::ReadOnly)) return false;
    QByteArray blob = f.readAll();

    m_engine.reset(m_runtime->deserializeCudaEngine(blob.constData(), blob.size()));
    if (!m_engine) return false;
    m_context.reset(m_engine->createExecutionContext());

    // 从引擎查询输出维度
    const char* outputName = nullptr;
    for (int i = 0; i < m_engine->getNbIOTensors(); ++i) {
        const char* tensorName = m_engine->getIOTensorName(i);
        if (m_engine->getTensorIOMode(tensorName) == nvinfer1::TensorIOMode::kOUTPUT) {
            outputName = tensorName;
            break;
        }
    }
    if (!outputName) return false;

    if (!inspectEngineTensors()) return false;
    return allocateBuffers();
}

bool PoseDetector::inspectEngineTensors() {
    if (!m_engine) return false;

    struct TensorInfo {
        std::string name;
        nvinfer1::Dims dims;
    };

    std::vector<TensorInfo> outputs;
    for (int i = 0; i < m_engine->getNbIOTensors(); ++i) {
        const char* tensorName = m_engine->getIOTensorName(i);
        if (!tensorName) continue;

        if (m_engine->getTensorIOMode(tensorName) == nvinfer1::TensorIOMode::kINPUT) {
            m_inputTensorName = tensorName;
        } else {
            outputs.push_back({tensorName, m_engine->getTensorShape(tensorName)});
        }
    }

    if (m_inputTensorName.empty() || outputs.size() < 2) {
        qWarning() << "[PoseDetector] Unexpected tensor layout in engine";
        return false;
    }

    auto productOf = [](const nvinfer1::Dims& dims) {
        int total = 1;
        for (int i = 0; i < dims.nbDims; ++i) {
            if (dims.d[i] <= 0) return 0;
            total *= dims.d[i];
        }
        return total;
    };

    int idxX = -1;
    int idxY = -1;
    for (int i = 0; i < static_cast<int>(outputs.size()); ++i) {
        QString name = QString::fromStdString(outputs[i].name).toLower();
        if (idxX < 0 && name.contains("simcc_x")) idxX = i;
        if (idxY < 0 && name.contains("simcc_y")) idxY = i;
    }
    if (idxX < 0 || idxY < 0) {
        idxX = outputs[0].dims.d[outputs[0].dims.nbDims - 1]
             <= outputs[1].dims.d[outputs[1].dims.nbDims - 1] ? 0 : 1;
        idxY = idxX == 0 ? 1 : 0;
    }

    const auto& outX = outputs[idxX];
    const auto& outY = outputs[idxY];
    if (outX.dims.nbDims < 3 || outY.dims.nbDims < 3) {
        qWarning() << "[PoseDetector] SimCC output rank is invalid";
        return false;
    }

    m_outputTensorNameX = outX.name;
    m_outputTensorNameY = outY.name;
    m_numKpts = outX.dims.d[outX.dims.nbDims - 2];
    m_simccXDim = outX.dims.d[outX.dims.nbDims - 1];
    m_simccYDim = outY.dims.d[outY.dims.nbDims - 1];
    m_outputSizeX = productOf(outX.dims);
    m_outputSizeY = productOf(outY.dims);

    if (m_numKpts <= 0 || m_simccXDim <= 0 || m_simccYDim <= 0 ||
        m_outputSizeX <= 0 || m_outputSizeY <= 0) {
        qWarning() << "[PoseDetector] SimCC tensor dimensions are invalid";
        return false;
    }
    return true;
}

bool PoseDetector::saveEngine(const QString& enginePath) {
    if (!m_engine) return false;
    auto serialised = std::unique_ptr<nvinfer1::IHostMemory>(m_engine->serialize());
    QFile f(enginePath);
    if (!f.open(QIODevice::WriteOnly)) return false;
    f.write((const char*)serialised->data(), (qint64)serialised->size());
    return true;
}

bool PoseDetector::allocateBuffers() {
    int inSize = 3 * m_inputH * m_inputW;
    cudaMalloc(&m_gpuInput,   inSize * sizeof(float));
    cudaMalloc(&m_gpuOutputX, m_outputSizeX * sizeof(float));
    cudaMalloc(&m_gpuOutputY, m_outputSizeY * sizeof(float));
    m_cpuOutputX.resize(m_outputSizeX);
    m_cpuOutputY.resize(m_outputSizeY);
    m_loaded = true;
    qInfo() << "[PoseDetector] Ready. Input:" << m_inputW << "×" << m_inputH
            << "SimCC:" << m_simccXDim << "x" << m_simccYDim
            << "Keypoints:" << m_numKpts << "FP16:" << m_fp16;
    return true;
}

// ── runInference ──────────────────────────────────────────────────────────────
bool PoseDetector::runInference(const std::vector<float>& inputCHW) {
#ifdef HAVE_TENSORRT
    if (!m_context) return false;
    cudaMemcpy(m_gpuInput, inputCHW.data(),
               inputCHW.size() * sizeof(float), cudaMemcpyHostToDevice);
    nvinfer1::Dims4 inputDims{1, 3, m_inputH, m_inputW};
    bool ok = m_context->setInputShape(m_inputTensorName.c_str(), inputDims)
           && m_context->setTensorAddress(m_inputTensorName.c_str(), m_gpuInput)
           && m_context->setTensorAddress(m_outputTensorNameX.c_str(), m_gpuOutputX)
           && m_context->setTensorAddress(m_outputTensorNameY.c_str(), m_gpuOutputY)
           && m_context->enqueueV3(0);
    if (ok) {
        cudaMemcpy(m_cpuOutputX.data(), m_gpuOutputX,
                   m_outputSizeX * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(m_cpuOutputY.data(), m_gpuOutputY,
                   m_outputSizeY * sizeof(float), cudaMemcpyDeviceToHost);
    }
    return ok;
#elif defined(HAVE_ONNXRUNTIME)
    if (!m_onnxSession) return false;
    
    try {
        Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault);
        
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            memoryInfo, inputCHW.data(), inputCHW.size(),
            m_onnxInputShape.data(), m_onnxInputShape.size());
        
        Ort::RunOptions runOptions;
        std::vector<Ort::Value> outputs = m_onnxSession->Run(
            runOptions,
            m_onnxInputNames.data(), &inputTensor, 1,
            m_onnxOutputNames.data(), m_onnxOutputNames.size());
        
        // 假设输出顺序：simcc_x [0], simcc_y [1]
        float* simccX = outputs[0].GetTensorMutableData<float>();
        float* simccY = outputs[1].GetTensorMutableData<float>();
        
        std::copy(simccX, simccX + m_cpuOutputX.size(), m_cpuOutputX.data());
        std::copy(simccY, simccY + m_cpuOutputY.size(), m_cpuOutputY.data());
        
        return true;
    } catch (const Ort::Exception& e) {
        qWarning() << "[PoseDetector] ONNX推理失败:" << e.what();
        return false;
    }
#else
    return true;  // stub
#endif
}

// ── TopDownAffine ────────────────────────────────────────────────────────────
// 实现与模型训练时相同的TopDownAffine预处理
PoseDetector::CropInfo PoseDetector::prepareInput(
        const QImage& src, const QRectF& roi,
        std::vector<float>& outCHW) const
{
    QImage rgb = src.convertToFormat(QImage::Format_RGB888);
    QRectF cropRoi = adjustedPoseRoi(roi, m_inputW, m_inputH, rgb.width(), rgb.height());
    if (cropRoi.width() < 2.f || cropRoi.height() < 2.f) {
        cropRoi = QRectF(0, 0, std::max(1, rgb.width()), std::max(1, rgb.height()));
    }

    outCHW.resize(3 * m_inputW * m_inputH);
    float centerX = cropRoi.x() + cropRoi.width() * 0.5f;
    float centerY = cropRoi.y() + cropRoi.height() * 0.5f;
    float scaleX = cropRoi.width() / static_cast<float>(m_inputW);
    float scaleY = cropRoi.height() / static_cast<float>(m_inputH);
    
    for (int y = 0; y < m_inputH; ++y) {
        for (int x = 0; x < m_inputW; ++x) {
            float srcX = centerX + (x - m_inputW * 0.5f) * scaleX;
            float srcY = centerY + (y - m_inputH * 0.5f) * scaleY;
            
            for (int c = 0; c < 3; ++c) {
                float pix = sampleChannel(rgb, srcX, srcY, 2-c);  // BGR order
                outCHW[c * m_inputH * m_inputW + y * m_inputW + x] = (pix - kRtmposeMean[c]) / kRtmposeStd[c];
            }
        }
    }

    return { cropRoi };
}

// ── softArgmax ────────────────────────────────────────────────────────────────
PoseDetector::SimCCPeak PoseDetector::decodeSimCCPeak(
        const float* simccX, const float* simccY) const
{
    const int idxX = static_cast<int>(std::max_element(simccX, simccX + m_simccXDim) - simccX);
    const int idxY = static_cast<int>(std::max_element(simccY, simccY + m_simccYDim) - simccY);
    const float scoreX = 1.f / (1.f + std::exp(-simccX[idxX]));
    const float scoreY = 1.f / (1.f + std::exp(-simccY[idxY]));
    // 根据SimCC论文，使用split_ratio将索引转换为输入图像坐标
    // 此处使用实际的SimCC维度与输入维度的比例
    // RTMPose SimCC 官方解码逻辑: split_ratio=2 时坐标为 idx / 2
    return {
        static_cast<float>(idxX) / 2.0f,
        static_cast<float>(idxY) / 2.0f,
        0.5f * (scoreX + scoreY)
    };
}

// ── heatmapsToKeypoints ───────────────────────────────────────────────────────
QVector<Keypoint> PoseDetector::simccToKeypoints(
        const CropInfo& crop,
        int imgW, int imgH) const
{
    Q_UNUSED(imgW) Q_UNUSED(imgH)
    QVector<Keypoint> kpts;
    kpts.reserve(m_numKpts);

    for (int k = 0; k < m_numKpts; ++k) {
        const float* simccX = m_cpuOutputX.data() + k * m_simccXDim;
        const float* simccY = m_cpuOutputY.data() + k * m_simccYDim;
        auto [inputX, inputY, score] = decodeSimCCPeak(simccX, simccY);

        // 确保与prepareInput完全对称的坐标映射
        // 在prepareInput中：srcX = centerX + (x - m_inputW * 0.5f) * scaleX
        // 所以逆向映射：origX = centerX + (inputX - m_inputW * 0.5f) * scaleX
        float cropCenterX = crop.roiPixels.x() + crop.roiPixels.width() * 0.5f;
        float cropCenterY = crop.roiPixels.y() + crop.roiPixels.height() * 0.5f;
        
        // 使用与prepareInput完全相同的缩放比例计算
        float scaleX = crop.roiPixels.width() / static_cast<float>(m_inputW);
        float scaleY = crop.roiPixels.height() / static_cast<float>(m_inputH);
        
        // 完全对称的逆变换
        const float ix = cropCenterX + (inputX - m_inputW * 0.5f) * scaleX;
        const float iy = cropCenterY + (inputY - m_inputH * 0.5f) * scaleY;

        Keypoint kp;
        kp.x       = ix;
        kp.y       = iy;
        kp.score   = score;
        kp.visible = (score >= m_visThresh);
        kpts.push_back(kp);
    }
    return kpts;
}

// ── detect (全图单人) ─────────────────────────────────────────────────────────
QVector<PoseResult> PoseDetector::detect(const QImage& frame) {
#ifdef HAVE_TENSORRT
    if (!m_context) return {};
#elif defined(HAVE_ONNXRUNTIME)
    if (!m_onnxSession) return {};
#endif
    
    QRectF fullRoi(0, 0, frame.width(), frame.height());
    DetectionResult dr;
    dr.bbox       = fullRoi;
    dr.classId    = 0;
    dr.confidence = 1.f;
    dr.label      = "person";
    return detectWithBoxes(frame, { dr });
}

// ── detectWithBoxes (top-down 多人) ──────────────────────────────────────────
QVector<PoseResult> PoseDetector::detectWithBoxes(
        const QImage& frame,
        const QVector<DetectionResult>& boxes)
{
#ifdef HAVE_TENSORRT
    if (!m_context) return {};
#elif defined(HAVE_ONNXRUNTIME)
    if (!m_onnxSession) return {};
#endif

    QVector<PoseResult> results;
    results.reserve(boxes.size());

    for (const auto& box : boxes) {
        std::vector<float> inputCHW;
        auto crop = prepareInput(frame, box.bbox, inputCHW);

        if (!runInference(inputCHW)) continue;

        auto kpts = simccToKeypoints(crop, frame.width(), frame.height());

        float scoreSum = 0.f;
        int   visCount = 0;
        for (const auto& kp : kpts)
            if (kp.visible) { scoreSum += kp.score; ++visCount; }

        PoseResult pr;
        pr.keypoints   = kpts;
        pr.poseScore   = visCount > 0 ? scoreSum / visCount : 0.f;
        pr.boundingBox = box.bbox;

        if (pr.poseScore >= m_poseScoreThresh)
            results.push_back(pr);
    }
    return results;
}

// ═════════════════════════════════════════════════════════════════════════════
#else  // ─── STUB：无 TensorRT 时的合成数据 ────────────────────────────────────
// ─────────────────────────────────────────────────────────────────────────────

PoseDetector::PoseDetector(QObject* parent) : QObject(parent) {}
PoseDetector::~PoseDetector() = default;

QString PoseDetector::backendName() const { return QStringLiteral("Stub-Pose"); }

bool PoseDetector::loadModel(const QString& path) {
    Q_UNUSED(path)
    qWarning() << "[PoseDetector] Stub 模式（无 TensorRT）";
    // 根据模型配置，SimCC输出维度是输入的2倍
    m_simccXDim = m_inputW * 2;
    m_simccYDim = m_inputH * 2;
    m_loaded = true;
    return true;
}

bool PoseDetector::loadEngine(const QString&) { return false; }
bool PoseDetector::saveEngine(const QString&) { return false; }

// 生成 T 形站立姿态的合成关键点
static QVector<Keypoint> makeSyntheticPose(const QRectF& roi, float visThresh)
{
    static const float nx[COCOSkeleton::NumKeypoints] = {
        0.50f, 0.45f, 0.55f, 0.40f, 0.60f,
        0.35f, 0.65f,
        0.30f, 0.70f,
        0.25f, 0.75f,
        0.40f, 0.60f,
        0.40f, 0.60f,
        0.40f, 0.60f
    };
    static const float ny[COCOSkeleton::NumKeypoints] = {
        0.10f, 0.12f, 0.12f, 0.15f, 0.15f,
        0.25f, 0.25f,
        0.42f, 0.42f,
        0.55f, 0.55f,
        0.55f, 0.55f,
        0.72f, 0.72f,
        0.90f, 0.90f
    };

    static int tick = 0; ++tick;
    float swing = 0.02f * std::sin(tick * 0.05f);

    QVector<Keypoint> kpts;
    kpts.reserve(COCOSkeleton::NumKeypoints);
    for (int i = 0; i < COCOSkeleton::NumKeypoints; ++i) {
        Keypoint kp;
        kp.x       = roi.x() + (nx[i] + swing) * roi.width();
        kp.y       = roi.y() + ny[i] * roi.height();
        kp.score   = 0.92f - i * 0.01f;
        kp.visible = kp.score >= visThresh;
        kpts.push_back(kp);
    }
    return kpts;
}

bool PoseDetector::runInference(const std::vector<float>&) { return true; }

QVector<PoseResult> PoseDetector::detect(const QImage& frame) {
    // Bug 2 修复：原版在 Stub 分支内部嵌套了 #ifdef HAVE_TENSORRT，
    // 该宏在 Stub 编译单元中未定义，导致预处理后逻辑矛盾：
    //   - #ifdef HAVE_TENSORRT 块被整体跳过
    //   - 但 #else 块里的 frame.isNull() 检查不足以阻止后续合成逻辑
    // 实际后果是：当 frame 非空时合成姿态路径会被执行，但 Stub 分支
    // 外层的 PoseDetectorManager::init() 在 loadModel() 失败时直接
    // return false，导致 m_ready=false，submitFrame() 提前退出，
    // detect() 根本不会被调用。
    //
    // 修复方案：
    //   1. PoseDetectorManager::init() 不因 loadModel() 失败而中止（见 .h）
    //   2. 此处直接生成合成姿态，不再有任何条件门控
    if (frame.isNull()) return {};

    // 生成合成姿态用于UI测试
    QRectF roi(frame.width()  * 0.15, frame.height() * 0.05,
               frame.width()  * 0.70, frame.height() * 0.90);
    DetectionResult dr;
    dr.bbox       = roi;
    dr.classId    = 0;
    dr.confidence = 0.99f;
    dr.label      = "person";
    return detectWithBoxes(frame, { dr });
}

QVector<PoseResult> PoseDetector::detectWithBoxes(
        const QImage& frame,
        const QVector<DetectionResult>& boxes)
{
    Q_UNUSED(frame)
    QVector<PoseResult> results;
    for (const auto& box : boxes) {
        auto kpts = makeSyntheticPose(box.bbox, m_visThresh);
        float scoreSum = 0.f; int cnt = 0;
        for (const auto& k : kpts) if (k.visible) { scoreSum += k.score; ++cnt; }
        PoseResult pr;
        pr.keypoints  = kpts;
        pr.poseScore  = cnt > 0 ? scoreSum / cnt : 0.f;
        
        // 从实际检测到的关节点重新计算真实包围盒，不使用输入box
        float minX = 1e9f, maxX = -1e9f;
        float minY = 1e9f, maxY = -1e9f;
        int visibleCount = 0;
        
        for (const auto& kp : kpts) {
            if (kp.visible) {
                minX = std::min(minX, kp.x);
                maxX = std::max(maxX, kp.x);
                minY = std::min(minY, kp.y);
                maxY = std::max(maxY, kp.y);
                visibleCount++;
            }
        }
        
        if (visibleCount >= 3) {
            // 至少3个可见关节点时才计算真实包围盒
            float margin = 0.1f * std::max(maxX - minX, maxY - minY);
            pr.boundingBox = QRectF(
                minX - margin, minY - margin,
                maxX - minX + 2 * margin,
                maxY - minY + 2 * margin
            );
        } else {
            // 可见点不足时回退使用原始检测框
            pr.boundingBox = box.bbox;
        }
        
        results.push_back(pr);
    }
    return results;
}

// stub 模式下这些方法保持编译不报错
PoseDetector::CropInfo PoseDetector::prepareInput(
        const QImage&, const QRectF&, std::vector<float>&) const
{ return { QRectF() }; }

PoseDetector::SimCCPeak PoseDetector::decodeSimCCPeak(
        const float*, const float*) const
{ return {0.f, 0.f, 0.f}; }

QVector<Keypoint> PoseDetector::simccToKeypoints(
        const CropInfo&, int, int) const
{ return {}; }

#endif // HAVE_TENSORRT