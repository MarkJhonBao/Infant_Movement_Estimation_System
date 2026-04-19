#pragma once
#include "DetectorBase.h"   // DetectionResult, DetectorBase
#include <QObject>
#include <QThread>
#include <QMutex>
#include <QImage>
#include <QString>
#include <QVector>
#include <memory>

// =============================================================================
//  DetectorWorker  —  在后台线程执行目标检测（drop-frame 策略）
// =============================================================================
class DetectorWorker : public QObject {
    Q_OBJECT
public:
    explicit DetectorWorker(DetectorBase* det, QObject* parent = nullptr);

public slots:
    void processFrame(const QImage& frame);

signals:
    void resultsReady(const QVector<DetectionResult>& results);

private:
    DetectorBase* m_detector{nullptr};
    QMutex        m_mutex;
    bool          m_busy{false};
};

// =============================================================================
//  DetectorManager  —  拥有 DetectorBase 实现 + 工作线程
//
//  典型用法：
//    DetectorManager* mgr = new DetectorManager(this);
//    mgr->setBackend(DetectorManager::Backend::TensorRT);
//    mgr->init("yolo.onnx", "coco.txt");
//    connect(mgr, &DetectorManager::detectionReady, this, &MyClass::onDetections);
//    // 每帧调用：
//    mgr->submitFrame(frame);
// =============================================================================
class DetectorManager : public QObject {
    Q_OBJECT
public:
    enum class Backend { ONNX, TensorRT };

    explicit DetectorManager(QObject* parent = nullptr);
    ~DetectorManager() override;

    // ── 配置（须在 init 前调用）
    void setBackend(Backend b) { m_backend = b; }

    // ── 初始化：加载模型、启动工作线程
    bool init(const QString& modelPath, const QString& labelsPath = {});

    // ── 提交帧（若工作线程繁忙则丢弃该帧）
    void submitFrame(const QImage& frame);

    bool    isReady()     const { return m_ready; }
    QString backendName() const;

signals:
    // 检测结果就绪（跨线程，已通过 QueuedConnection 投递到调用线程）
    void detectionReady(const QVector<DetectionResult>& results);

    // 内部跨线程信号（前缀 _ 表示内部使用）
    void _runDetection(const QImage& frame);

private:
    std::unique_ptr<DetectorBase> m_detector;
    DetectorWorker* m_worker{nullptr};
    QThread*        m_thread{nullptr};
    bool            m_ready{false};
    Backend         m_backend{Backend::ONNX};
};
