#pragma once
#include <QWidget>
#include <QImage>
#include <QTimer>
#include <QVector>
#include <QThread>
#include <QAtomicInteger>
#include <QMutex>
#include <QWaitCondition>
#include <QSize>
#include <atomic>
#include <utility>

// ── 姿态结果（替换原 DetectorBase.h 中的 DetectionResult）─────────────────
#include "../detector/PoseResult.h"

// Lock-free ring buffer for frame queue
template<typename T, int SIZE>
class RingBuffer {
public:
    RingBuffer() : m_head(0), m_tail(0) {}

    bool tryPush(const T& value) {
        int nextHead = (m_head.loadAcquire() + 1) % SIZE;
        if (nextHead == m_tail.loadAcquire()) return false;
        m_buffer[m_head] = value;
        m_head.storeRelease(nextHead);
        return true;
    }

    bool tryPop(T& value) {
        int tail = m_tail.loadAcquire();
        if (tail == m_head.loadAcquire()) return false;
        value = m_buffer[tail];
        m_tail.storeRelease((tail + 1) % SIZE);
        return true;
    }

    int count() const {
        int head = m_head.loadAcquire();
        int tail = m_tail.loadAcquire();
        return (head >= tail) ? head - tail : (SIZE - tail + head);
    }

    bool isEmpty() const { return m_head.loadAcquire() == m_tail.loadAcquire(); }
    bool isFull()  const { return (m_head.loadAcquire() + 1) % SIZE == m_tail.loadAcquire(); }

private:
    QAtomicInteger<int> m_head;
    QAtomicInteger<int> m_tail;
    T m_buffer[SIZE];
};

// ─────────────────────────────────────────────────────────────────────────────
//  FrameScalingWorker
//
//  【修复 Bug 1】：原版继承 QObject + moveToThread，然后通过 invokeMethod
//  (QueuedConnection) 调用 processQueue()。processQueue() 内部使用
//  QWaitCondition::wait() 永久阻塞线程事件循环，导致 queueFrame() 槽
//  （同样是 QueuedConnection）永远无法被调度 → m_scaledFrame 始终为空
//  → 视频帧永远不显示。
//
//  修复方案：改为继承 QThread，在 run() 中执行阻塞循环，彻底脱离
//  Qt 事件循环，queueFrame() 改为普通线程安全函数（加互斥锁直接调用）。
// ─────────────────────────────────────────────────────────────────────────────
class FrameScalingWorker : public QThread {
    Q_OBJECT
public:
    explicit FrameScalingWorker(QObject* parent = nullptr);
    ~FrameScalingWorker() override;

    // 线程安全：可从任意线程直接调用（内部加锁）
    void setTargetSize(QSize size);
    void queueFrame(const QImage& frame, quint64 frameId);
    void stop();

signals:
    void scaledFrameReady(const QImage& original, const QImage& scaled,
                          QRect videoRect, quint64 frameId, qint64 timestamp);

protected:
    // 阻塞循环在独立线程 run() 中运行，不占用任何事件循环
    void run() override;

private:
    QSize   m_targetSize;
    QMutex  m_mutex;
    QWaitCondition m_wakeCondition;
    std::atomic<bool> m_running{true};
    RingBuffer<std::pair<QImage, quint64>, 4> m_frameQueue;
};

// ─────────────────────────────────────────────────────────────────────────────
//  DetectionDisplayWidget
//  中心面板：实时摄像头/视频帧 + 骨架姿态叠加渲染
// ─────────────────────────────────────────────────────────────────────────────
class DetectionDisplayWidget : public QWidget {
    Q_OBJECT
public:
    explicit DetectionDisplayWidget(QWidget* parent = nullptr);
    ~DetectionDisplayWidget() override;

    // ── 数据推送 ──────────────────────────────────────────────────────────────
    void setFrame(const QImage& img);

    // 推送姿态结果（替换原 setDetections）
    void setPoses(const QVector<PoseResult>& poses);

    // ── HUD 配置 ──────────────────────────────────────────────────────────────
    void setBackendName(const QString& name) { m_backendName = name; update(); }
    void setFps(double fps)                  { m_fps = fps;          update(); }

    // 替换原 setDetCount
    void setPoseCount(int n)                 { m_poseCount = n;      update(); }

    // ── 性能监控 ──────────────────────────────────────────────────────────────
    void setProcessingDelay(int ms)          { m_processingDelay = ms; update(); }
    void setCacheSize(int size)              { m_cacheSize = size; update(); }

    // ── 骨架样式（运行时可调）────────────────────────────────────────────────
    void setLimbWidth(float w)    { m_limbWidth  = w; }
    void setKptRadius(float r)    { m_kptRadius  = r; }
    void setShowLabels(bool show) { m_showLabels = show; update(); }
    void setShowBbox(bool show)   { m_showBbox   = show; update(); }

    // ── 新增：关键点坐标叠加面板 ──────────────────────────────────────────
    void setShowKeypointCoords(bool show) { m_showKeypointCoords = show; update(); }

    // ── 新增：关节速度面板（左下角）──────────────────────────────────────
    void setShowVelocityPanel(bool show) { m_showVelocityPanel = show; update(); }

    // ── 辅助方法 ─────────────────────────────────────────────────────────────
    bool   isFrameNull()    const { return m_frame.isNull(); }
    QImage currentFrame()   const { return m_frame; }

signals:
    // 约 30 fps 对外发帧（供 PoseDetectorManager 消费）
    void frameReady(const QImage& frame, quint64 frameId);

protected:
    void paintEvent(QPaintEvent*) override;
    void resizeEvent(QResizeEvent*) override;
    void showEvent(QShowEvent*) override;

private slots:
    void onScaledFrameReady(const QImage& original, const QImage& scaled,
                             QRect videoRect, quint64 frameId, qint64 timestamp);
    void onTimer();

public slots:
    // 接收带帧ID的姿态结果
    void setPosesWithId(quint64 frameId, const QVector<PoseResult>& poses);

private:
    // ── 骨架绘制辅助 ─────────────────────────────────────────────────────────
    void drawSkeleton(QPainter& p,
                      const PoseResult& pose,
                      float ox, float oy,
                      float scaleX, float scaleY,
                      int personIdx) const;

    static QColor limbColor(int srcIdx, int dstIdx);
    static QColor kptColor(int idx);

    // ── 帧-姿态对齐机制 ─────────────────────────────────────────────────────
    struct PoseEntry {
        quint64 frameId;
        QVector<PoseResult> poses;
        qint64 receiveTime;
    };

    // ── 数据成员 ─────────────────────────────────────────────────────────────
    QImage              m_frame;
    QImage              m_scaledFrame;
    QRect               m_videoRect;
    float               m_scaleX{1.0f};
    float               m_scaleY{1.0f};
    QVector<PoseResult> m_poses;
    quint64             m_currentFrameId{0};
    qint64              m_currentFrameTime{0};  // 帧发出时间戳
    qint64              m_lastRenderTime{0};
    qint64              m_renderLatency{0};

    // 姿态结果滑动缓存窗口（保留最近 16 帧）
    QVector<PoseEntry>  m_poseCache;

    QString m_backendName{"Stub-Pose"};
    double  m_fps{0.0};
    int     m_poseCount{0};
    int     m_processingDelay{0};
    int     m_cacheSize{0};
    int     m_droppedFrames{0};

    // 【修复 Bug 3】：原版使用 static 局部变量，跨帧共享状态造成跳帧逻辑
    // 状态污染。改为类成员变量，生命周期与对象绑定。
    int     m_consecutiveDrops{0};

    QTimer  m_timer;
    int     m_frameCount{0};
    qint64  m_lastMs{0};

    // 骨架样式
    float m_limbWidth {3.0f};
    float m_kptRadius {5.0f};
    bool  m_showLabels{false};
    bool  m_showBbox  {true};
    bool  m_showKeypointCoords{true};   // 新增：显示关键点坐标叠加面板
    bool  m_showVelocityPanel{true};    // 新增：显示左下角关节速度面板

    // ── 关节速度估算（左下角面板）────────────────────────────────────────
    struct KeypointVelocity {
        float vx{0.f};          // x方向速度 (像素/秒)
        float vy{0.f};          // y方向速度 (像素/秒)
        float speed{0.f};       // 合速度 (像素/秒)
        bool  valid{false};
    };
    // 每个人的17个关键点速度
    QVector<QVector<KeypointVelocity>> m_velocities;

    // 前一帧姿态和时间（用于差分计算速度）
    struct PrevPoseEntry {
        QVector<Keypoint> keypoints;
        qint64 timestamp{0};
    };
    QVector<PrevPoseEntry> m_prevPoses;  // 每个人一组

    // 【修复 Bug 1】：去掉 m_scalingThread（QThread* 外挂），
    // FrameScalingWorker 本身即线程，直接持有即可。
    FrameScalingWorker* m_scalingWorker{nullptr};
};
