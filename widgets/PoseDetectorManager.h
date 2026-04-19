#pragma once
#include "PoseDetector.h"
#include "DetectorBase.h"   // DetectionResult
#include <QObject>
#include <QThread>
#include <QMutex>
#include <QImage>
#include <memory>

// ─────────────────────────────────────────────────────────────────────────────
//  PoseWorker — 后台线程执行姿态推理 with proper frame drop & priority
// ─────────────────────────────────────────────────────────────────────────────
class PoseWorker : public QObject {
    Q_OBJECT
public:
    explicit PoseWorker(PoseDetector* det, QObject* parent = nullptr)
        : QObject(parent), m_detector(det) {}

public slots:
    void processFrame(const QImage& frame) {
        if (m_busy.testAndSetAcquire(false, true)) {
            // 添加空指针检查
            if (m_detector) {
                auto results = m_detector->detect(frame);
                m_busy.storeRelease(false);
                emit poseReady(results);
            } else {
                m_busy.storeRelease(false);
            }
        }
    }

    void processFrameWithId(const QImage& frame, quint64 frameId) {
        // Lock-free frame drop: if busy, immediately drop this frame
        if (m_busy.testAndSetAcquire(false, true)) {
            // 添加空指针检查
            if (m_detector) {
                auto results = m_detector->detect(frame);
                m_busy.storeRelease(false);
                emit poseReadyWithId(frameId, results);
            } else {
                m_busy.storeRelease(false);
            }
        }
    }

    void processFrameWithBoxes(const QImage& frame,
                               const QVector<DetectionResult>& boxes) {
        if (m_busy.testAndSetAcquire(false, true)) {
            // 添加空指针检查
            if (m_detector) {
                auto results = m_detector->detectWithBoxes(frame, boxes);
                m_busy.storeRelease(false);
                emit poseReady(results);
            } else {
                m_busy.storeRelease(false);
            }
        }
    }

    void stop() {
        m_running = false;
    }

signals:
    void poseReady(const QVector<PoseResult>& results);
    void poseReadyWithId(quint64 frameId, const QVector<PoseResult>& results);

private:
    PoseDetector* m_detector{nullptr};
    QAtomicInteger<bool> m_busy{false};
    std::atomic<bool> m_running{true};
};

// ─────────────────────────────────────────────────────────────────────────────
//  PoseDetectorManager — 拥有 PoseDetector + 工作线程
//
//  典型接线（top-down 联动目标检测）：
//
//    DetectorManager*     objDet  = ...;   // 已有目标检测器
//    PoseDetectorManager* poseMgr = new PoseDetectorManager(this);
//    poseMgr->init("rtmpose.onnx", "rtmpose.engine");
//
//    connect(objDet,  &DetectorManager::detectionReady,
//            poseMgr, [=](const QVector<DetectionResult>& dets){
//                poseMgr->submitFrameWithBoxes(currentFrame, dets);
//            });
//    connect(poseMgr, &PoseDetectorManager::poseReady,
//            overlay, &PoseOverlayWidget::updatePoses);
//
//  单人模式（无目标检测器）：
//
//    connect(frameTimer, &QTimer::timeout, [=]{
//        poseMgr->submitFrame(currentFrame);
//    });
// ─────────────────────────────────────────────────────────────────────────────
class PoseDetectorManager : public QObject {
    Q_OBJECT
public:
    explicit PoseDetectorManager(QObject* parent = nullptr) : QObject(parent) {}

    ~PoseDetectorManager() override {
        // 正确的线程关闭顺序：
        // 1. 先将 m_ready 置 false，阻止任何新的 emit _runPose* 信号入队
        // 2. 停止工作线程事件循环（quit），等待当前任务执行完毕（wait）
        // 3. QThread::finished 信号已连接 m_worker->deleteLater，
        //    wait() 返回意味着线程已退出，deleteLater 已被处理
        // 4. m_detector 最后析构（unique_ptr 自动），此时 worker 已不再访问它
        m_ready = false;
        if (m_thread && m_thread->isRunning()) {
            m_thread->quit();
            if (!m_thread->wait(5000)) {          // 最多等 5 秒
                qWarning() << "[PoseDetectorManager] 工作线程未能在 5s 内退出，强制终止";
                m_thread->terminate();
                m_thread->wait();
            }
        }
        // m_thread parent 为 nullptr（见 init），由我们手动 delete
        delete m_thread;
        m_thread = nullptr;
        // m_worker 由 QThread::finished→deleteLater 负责销毁，
        // wait() 返回后已析构完毕，此处只需置空指针
        m_worker = nullptr;
        // m_detector（unique_ptr）在此之后自动析构，安全
    }

    // ── 配置（须在 init 前调用）─────────────────────────────────────────────
    void setInputSize(int w, int h)        { m_inputW = w; m_inputH = h; }
    void setFP16(bool enable)              { m_fp16 = enable; }
    void setVisibilityThreshold(float t)   { m_visThresh = t; }
    void setPoseScoreThreshold(float t)    { m_poseScoreThresh = t; }

    // ── init ─────────────────────────────────────────────────────────────────
    bool init(const QString& onnxOrEnginePath,
              const QString& engineCachePath = {})
    {
        // ── 若已有线程在运行（热重载），先安全关闭旧线程 ──────────────────
        if (m_thread && m_thread->isRunning()) {
            m_ready = false;
            m_thread->quit();
            if (!m_thread->wait(5000)) {
                m_thread->terminate();
                m_thread->wait();
            }
            delete m_thread;
            m_thread = nullptr;
            m_worker = nullptr;   // 已由 finished→deleteLater 销毁
        }

        // ── 创建推理器 ────────────────────────────────────────────────────
        m_detector = std::make_unique<PoseDetector>();
        m_detector->setInputSize(m_inputW, m_inputH);
        m_detector->setFP16(m_fp16);
        m_detector->setVisibilityThreshold(m_visThresh);
        m_detector->setPoseScoreThreshold(m_poseScoreThresh);

        bool ok = false;
        const bool isOnnx = onnxOrEnginePath.endsWith(".onnx", Qt::CaseInsensitive);
        if (isOnnx) {
            ok = m_detector->loadModel(onnxOrEnginePath);
            if (ok && !engineCachePath.isEmpty())
                m_detector->saveEngine(engineCachePath);
        } else if (!onnxOrEnginePath.isEmpty()) {
            ok = m_detector->loadEngine(onnxOrEnginePath);
        }

        if (!ok) {
            if (!onnxOrEnginePath.isEmpty())
                qWarning() << "[PoseDetectorManager] 模型加载失败，进入 Stub 合成模式:"
                           << onnxOrEnginePath;
            else
                qInfo() << "[PoseDetectorManager] 无模型路径，以 Stub 模式启动";
        }

        // ── 创建工作线程 ──────────────────────────────────────────────────
        // 关键：m_thread parent 设为 nullptr，由析构函数手动管理生命周期，
        // 避免 Qt parent-child 树在主线程析构时以错误顺序销毁线程对象。
        m_thread = new QThread(nullptr);
        m_thread->setPriority(QThread::HighPriority);

        m_worker = new PoseWorker(m_detector.get());
        // moveToThread 前不能有 parent，m_worker 已无 parent（默认 nullptr）
        m_worker->moveToThread(m_thread);

        // 线程结束时销毁 worker（在线程的事件循环中执行 deleteLater，线程安全）
        connect(m_thread, &QThread::finished,
                m_worker, &QObject::deleteLater,
                Qt::DirectConnection);

        connect(this,     &PoseDetectorManager::_runPose,
                m_worker, &PoseWorker::processFrame,
                Qt::QueuedConnection);
        connect(this,     &PoseDetectorManager::_runPoseWithId,
                m_worker, &PoseWorker::processFrameWithId,
                Qt::QueuedConnection);
        connect(this,     &PoseDetectorManager::_runPoseWithBoxes,
                m_worker, &PoseWorker::processFrameWithBoxes,
                Qt::QueuedConnection);
        connect(m_worker, &PoseWorker::poseReady,
                this,     &PoseDetectorManager::poseReady,
                Qt::QueuedConnection);
        connect(m_worker, &PoseWorker::poseReadyWithId,
                this,     &PoseDetectorManager::poseReadyWithId,
                Qt::QueuedConnection);

        m_thread->start();
        m_ready = true;
        qInfo() << "[PoseDetectorManager] 就绪. 后端:" << m_detector->backendName();
        emit modelUpdated(m_detector->backendName());
        return true;
    }

    bool    isReady()     const { return m_ready; }
    QString backendName() const { return m_detector ? m_detector->backendName() : "None"; }

public slots:
    void submitFrame(const QImage& frame) {
        if (m_ready) emit _runPose(frame);
    }
    void submitFrame(const QImage& frame, quint64 frameId) {
        if (m_ready) emit _runPoseWithId(frame, frameId);
    }
    void submitFrameWithBoxes(const QImage& frame,
                              const QVector<DetectionResult>& boxes) {
        if (m_ready) emit _runPoseWithBoxes(frame, boxes);
    }

signals:
    void poseReady(const QVector<PoseResult>& results);
    void poseReadyWithId(quint64 frameId, const QVector<PoseResult>& results);
    void modelUpdated(const QString& backendName);

    void _runPose(const QImage& frame);
    void _runPoseWithId(const QImage& frame, quint64 frameId);
    void _runPoseWithBoxes(const QImage& frame,
                           const QVector<DetectionResult>& boxes);

private:
    std::unique_ptr<PoseDetector> m_detector;
    PoseWorker* m_worker{nullptr};
    QThread*    m_thread{nullptr};
    bool        m_ready{false};

    int   m_inputW{288};
    int   m_inputH{384};
    bool  m_fp16{false};          // Stub/ONNX 模式默认关闭 FP16
    float m_visThresh{0.3f};
    float m_poseScoreThresh{0.3f};
};