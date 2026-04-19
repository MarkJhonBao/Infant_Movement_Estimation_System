#include "DetectionDisplayWidget.h"
#include "DashboardTheme.h"
#include <QPainter>
#include <QDateTime>
#include <QFontMetrics>
#include <QElapsedTimer>
#include <algorithm>
#include <cmath>
#include <QShowEvent>

// ─────────────────────────────────────────────────────────────────────────────
//  FrameScalingWorker 实现
//
//  【修复 Bug 1】架构说明：
//  继承 QThread，将阻塞循环放在 run() 中独立运行。
//  queueFrame() 是普通线程安全函数（加互斥锁），可从主线程直接调用，
//  调用后通过 wakeOne() 唤醒 run() 中的等待，无需经过任何事件循环。
//  scaledFrameReady 信号通过 Qt::QueuedConnection 跨线程投递回主线程。
// ─────────────────────────────────────────────────────────────────────────────
FrameScalingWorker::FrameScalingWorker(QObject* parent)
    : QThread(parent)
    , m_targetSize(QSize(0, 0))
    , m_running(true)
{}

FrameScalingWorker::~FrameScalingWorker() {
    stop();
    wait();   // 等待 run() 退出，防止析构竞争
}

void FrameScalingWorker::setTargetSize(QSize size) {
    QMutexLocker lk(&m_mutex);
    m_targetSize = size;
}

// 【修复 Bug 1】：原版此函数是 slot（QueuedConnection），由于 processQueue()
// 阻塞了线程事件循环，此 slot 永远无法被调度，wakeOne() 也永远不会执行。
// 修复后改为普通成员函数，直接加锁操作共享数据再调用 wakeOne()，
// 与 run() 中的 wait() 正确配合。
void FrameScalingWorker::queueFrame(const QImage& frame, quint64 frameId) {
    QMutexLocker lk(&m_mutex);
    if (m_frameQueue.isFull()) {
        // 队列满时丢弃最旧帧，为新帧腾出空间
        std::pair<QImage, quint64> dummy;
        m_frameQueue.tryPop(dummy);
    }
    m_frameQueue.tryPush({frame, frameId});
    m_wakeCondition.wakeOne();   // 唤醒 run() 中的 wait()
}

void FrameScalingWorker::stop() {
    QMutexLocker lk(&m_mutex);
    m_running = false;
    m_wakeCondition.wakeAll();   // 确保 run() 中的 wait() 能够退出
}

// 【修复 Bug 1】：原版 processQueue() 通过 invokeMethod(QueuedConnection)
// 投递到线程事件循环，其内部 wait() 永久阻塞事件循环，造成死锁。
// 修复后改为 run() 重写，由 QThread 内部机制独立调度，完全脱离事件循环。
void FrameScalingWorker::run() {
    while (m_running) {
        std::pair<QImage, quint64> entry;

        {
            QMutexLocker lk(&m_mutex);
            // 若队列为空则等待，wakeOne()/wakeAll() 会唤醒此处
            while (m_frameQueue.isEmpty() && m_running)
                m_wakeCondition.wait(&m_mutex);

            if (!m_running) break;

            if (!m_frameQueue.tryPop(entry)) continue;
        }

        // ── 以下在锁外执行（耗时的缩放操作不占用互斥锁）──────────────────
        QSize size;
        {
            QMutexLocker lk(&m_mutex);
            size = m_targetSize;
        }

        if (size.isEmpty() || entry.first.isNull()) continue;

        qint64 timestamp = QDateTime::currentMSecsSinceEpoch();

        QImage scaled = entry.first.scaled(size, Qt::KeepAspectRatio,
                                           Qt::SmoothTransformation);
        int ox = (size.width()  - scaled.width())  / 2;
        int oy = (size.height() - scaled.height()) / 2;
        QRect videoRect(ox, oy, scaled.width(), scaled.height());

        // Qt::QueuedConnection 自动跨线程投递回主线程（在连接处指定）
        emit scaledFrameReady(entry.first, scaled, videoRect,
                              entry.second, timestamp);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  DetectionDisplayWidget 实现
// ─────────────────────────────────────────────────────────────────────────────
DetectionDisplayWidget::DetectionDisplayWidget(QWidget* parent)
    : QWidget(parent)
{
    setMinimumSize(400, 300);
    setAttribute(Qt::WA_OpaquePaintEvent);
    setAttribute(Qt::WA_NoSystemBackground);

    // 初始化成员变量
    m_frame              = QImage();
    m_scaledFrame        = QImage();
    m_videoRect          = QRect();
    m_scaleX             = 1.0f;
    m_scaleY             = 1.0f;
    m_currentFrameId     = 0;
    m_frameCount         = 0;
    m_droppedFrames      = 0;
    m_consecutiveDrops   = 0;   // 【修复 Bug 3】使用成员变量
    m_renderLatency      = 0;
    m_lastRenderTime     = 0;
    m_currentFrameTime   = 0;
    m_fps                = 0.0;
    m_poseCount          = 0;
    m_limbWidth          = 2.0f;
    m_kptRadius          = 3.0f;
    m_showBbox           = true;
    m_showLabels         = false;
    m_cacheSize          = 0;

    // 【修复 Bug 1】：创建并启动 FrameScalingWorker（现为 QThread 子类）
    // 不再需要 moveToThread / invokeMethod / QTimer::singleShot 等样板代码
    m_scalingWorker = new FrameScalingWorker(this);

    // scaledFrameReady 从工作线程发出，Qt::QueuedConnection 投递回主线程
    connect(m_scalingWorker, &FrameScalingWorker::scaledFrameReady,
            this,            &DetectionDisplayWidget::onScaledFrameReady,
            Qt::QueuedConnection);

    m_scalingWorker->setTargetSize(size());
    m_scalingWorker->start();   // 启动独立线程，run() 开始阻塞等待

    // 33ms ≈ 30fps UI 刷新定时器
    connect(&m_timer, &QTimer::timeout,
            this,     &DetectionDisplayWidget::onTimer);
    m_timer.start(33);
    m_lastMs = QDateTime::currentMSecsSinceEpoch();
}

DetectionDisplayWidget::~DetectionDisplayWidget() {
    m_scalingWorker->stop();
    m_scalingWorker->wait();
    // m_scalingWorker 父对象是 this，Qt 会自动 delete，无需手动释放
}

// ─── setFrame ─────────────────────────────────────────────────────────────────
void DetectionDisplayWidget::setFrame(const QImage& img) {
    qint64 now = QDateTime::currentMSecsSinceEpoch();

    // 1. 空帧直接跳过
    if (img.isNull()) {
        m_droppedFrames++;
        return;
    }

    // 2. 渲染延迟过高且帧间隔太短时跳帧
    // Bug 4 修复：原版阈值 renderLatency>50ms && 间隔<20ms 过于激进，
    // 在普通 PC 上 paintEvent 稍慢时就会持续丢帧，导致画面长时间黑屏。
    // 放宽至 100ms / 10ms，仅在真正过载时才丢帧。
    if (m_renderLatency > 100 && (now - m_lastRenderTime) < 10) {
        m_droppedFrames++;
        return;
    }

    // 3. 连续丢帧保护：超过 10 帧强制显示一帧避免画面完全卡死
    // 【修复 Bug 3】：原版使用 static 局部变量（跨调用共享状态）
    // 修复后使用成员变量 m_consecutiveDrops，生命周期与对象绑定
    if (m_renderLatency > 60 && (now - m_lastRenderTime) < 15) {
        if (m_consecutiveDrops < 10) {
            m_droppedFrames++;
            m_consecutiveDrops++;
            return;
        } else {
            m_consecutiveDrops = 0;  // 强制显示一帧
        }
    } else {
        m_consecutiveDrops = 0;
    }

    m_currentFrameId++;
    m_currentFrameTime = now;
    ++m_frameCount;

    // FPS 计算（每 30 帧更新一次）
    if (m_frameCount % 30 == 0) {
        qint64 dt = now - m_lastMs;
        if (dt > 0) m_fps = 30000.0 / dt;
        m_lastMs = now;
    }

    // 发送帧到姿态检测器（信号，保持原有接口）
    emit frameReady(img, m_currentFrameId);

    // 【修复 Bug 1】：原版通过 emit frameForScaling(...)（QueuedConnection）
    // 投递到已被阻塞的线程事件循环，帧永远无法送达缩放队列。
    // 修复后直接调用线程安全函数，内部加锁 + wakeOne()，立即唤醒 run()。
    m_scalingWorker->queueFrame(img, m_currentFrameId);
}

void DetectionDisplayWidget::onScaledFrameReady(const QImage& original,
                                                  const QImage& scaled,
                                                  QRect         videoRect,
                                                  quint64       frameId,
                                                  qint64        /*timestamp*/)
{
    // 丢弃严重滞后的帧（落后当前帧超过 5 帧）
    if (frameId + 5 < m_currentFrameId) {
        qDebug() << "丢弃延迟帧 - 帧ID:" << frameId << "当前帧:" << m_currentFrameId;
        return;
    }

    m_frame       = original;
    m_scaledFrame = scaled;
    m_videoRect   = videoRect;

    if (!original.isNull() && original.width() > 0 && original.height() > 0) {
        m_scaleX = (float)scaled.width()  / original.width();
        m_scaleY = (float)scaled.height() / original.height();
    } else {
        m_scaleX = 1.0f;
        m_scaleY = 1.0f;
    }

    // Bug 1 修复：原版用严格 frameId 匹配查找 m_poseCache。
    // 帧缩放（FrameScalingWorker）和姿态推理（PoseWorker）是两条独立的
    // 异步链路，推理结果通常晚于缩放结果到达主线程，导致查找时缓存为空，
    // 骨架始终无法绘制。
    //
    // 修复方案：优先查找精确匹配；找不到时退化为"使用最新一帧姿态"。
    // 这样在姿态结果稍滞后（1-3帧）的正常情况下依然能显示骨架，
    // 视觉上仅有极小的时序偏差，用户不可感知。
    bool foundExact = false;
    for (const auto& entry : m_poseCache) {
        if (entry.frameId == frameId) {
            m_poses     = entry.poses;
            m_poseCount = entry.poses.size();
            if (m_currentFrameTime > 0) {
                qint64 delay = entry.receiveTime - m_currentFrameTime;
                setProcessingDelay(static_cast<int>(delay));
            }
            setCacheSize(m_poseCache.size());
            foundExact = true;
            break;
        }
    }
    // 精确帧未命中时，使用缓存中最新一帧姿态（prepend 保证 first() 最新）
    if (!foundExact && !m_poseCache.isEmpty()) {
        const auto& latest = m_poseCache.first();
        m_poses     = latest.poses;
        m_poseCount = latest.poses.size();
        setCacheSize(m_poseCache.size());
    }

    qint64 renderStart = QDateTime::currentMSecsSinceEpoch();
    update();
    m_renderLatency  = QDateTime::currentMSecsSinceEpoch() - renderStart;
    m_lastRenderTime = renderStart;
}

// ─── setPoses ─────────────────────────────────────────────────────────────────
void DetectionDisplayWidget::setPoses(const QVector<PoseResult>& poses) {
    m_poses     = poses;
    m_poseCount = poses.size();
    update();
}

void DetectionDisplayWidget::setPosesWithId(quint64 frameId,
                                             const QVector<PoseResult>& poses)
{
    qint64 currentTime = QDateTime::currentMSecsSinceEpoch();

    PoseEntry entry;
    entry.frameId     = frameId;
    entry.poses       = poses;
    entry.receiveTime = currentTime;

    m_poseCache.prepend(entry);

    // 保留最近 16 帧 / 5 秒内的缓存
    while (m_poseCache.size() > 16 ||
           (!m_poseCache.isEmpty() &&
            (currentTime - m_poseCache.last().receiveTime) > 5000)) {
        m_poseCache.pop_back();
    }

    // ── 关节速度估算 ──────────────────────────────────────────────────────
    // 对每个检测到的人，与上一帧对应人进行差分，计算各关键点速度（像素/秒）
    m_velocities.resize(poses.size());

    for (int pi = 0; pi < poses.size(); ++pi) {
        const PoseResult& cur = poses[pi];
        m_velocities[pi].resize(COCOSkeleton::NumKeypoints);

        if (pi < m_prevPoses.size() && m_prevPoses[pi].timestamp > 0) {
            const PrevPoseEntry& prev = m_prevPoses[pi];
            double dtSec = (currentTime - prev.timestamp) / 1000.0;

            if (dtSec > 0.01 && dtSec < 2.0) {  // 帧间隔合理范围：10ms ~ 2s
                for (int ki = 0; ki < qMin(cur.keypoints.size(),
                                           (int)prev.keypoints.size()); ++ki) {
                    const Keypoint& ck = cur.keypoints[ki];
                    const Keypoint& pk = prev.keypoints[ki];
                    if (ck.visible && pk.visible) {
                        float vx = (ck.x - pk.x) / (float)dtSec;
                        float vy = (ck.y - pk.y) / (float)dtSec;
                        m_velocities[pi][ki].vx    = vx;
                        m_velocities[pi][ki].vy    = vy;
                        m_velocities[pi][ki].speed = std::sqrt(vx * vx + vy * vy);
                        m_velocities[pi][ki].valid = true;
                    } else {
                        m_velocities[pi][ki] = KeypointVelocity{};
                    }
                }
            } else {
                // 时间差异常，清零
                for (auto& v : m_velocities[pi]) v = KeypointVelocity{};
            }
        } else {
            // 无上一帧记录，全部无效
            for (auto& v : m_velocities[pi]) v = KeypointVelocity{};
        }
    }

    // 更新上一帧快照
    m_prevPoses.resize(poses.size());
    for (int pi = 0; pi < poses.size(); ++pi) {
        m_prevPoses[pi].keypoints  = poses[pi].keypoints;
        m_prevPoses[pi].timestamp  = currentTime;
    }

    // 若恰好是当前显示帧，立即更新
    if (frameId == m_currentFrameId) {
        m_poses     = poses;
        m_poseCount = poses.size();
        setCacheSize(m_poseCache.size());
        update();
    }
}

void DetectionDisplayWidget::resizeEvent(QResizeEvent* e) {
    QWidget::resizeEvent(e);
    // setTargetSize 内部加锁，线程安全，可直接调用
    m_scalingWorker->setTargetSize(size());
}

void DetectionDisplayWidget::showEvent(QShowEvent* e) {
    QWidget::showEvent(e);
    m_scalingWorker->setTargetSize(size());
}

void DetectionDisplayWidget::onTimer() {
    if (!m_scaledFrame.isNull())
        update();
}

// ─────────────────────────────────────────────────────────────────────────────
//  骨架颜色方案
// ─────────────────────────────────────────────────────────────────────────────
QColor DetectionDisplayWidget::limbColor(int s, int d) {
    using K = COCOSkeleton::KptIdx;
    bool head  = (s <= K::RightEar || d <= K::RightEar);
    bool left  = (s == K::LeftShoulder  || s == K::LeftElbow   ||
                  s == K::LeftWrist     || s == K::LeftHip      ||
                  s == K::LeftKnee      || s == K::LeftAnkle    ||
                  d == K::LeftShoulder  || d == K::LeftElbow    ||
                  d == K::LeftWrist     || d == K::LeftHip      ||
                  d == K::LeftKnee      || d == K::LeftAnkle);
    bool right = (s == K::RightShoulder || s == K::RightElbow  ||
                  s == K::RightWrist    || s == K::RightHip     ||
                  s == K::RightKnee     || s == K::RightAnkle   ||
                  d == K::RightShoulder || d == K::RightElbow   ||
                  d == K::RightWrist    || d == K::RightHip     ||
                  d == K::RightKnee     || d == K::RightAnkle);

    if (head)  return QColor(0xff, 0xdc, 0x1e);   // 黄 - 头部
    if (left)  return QColor(0x1e, 0x90, 0xff);   // 蓝 - 左侧
    if (right) return QColor(0xff, 0x50, 0x50);   // 红 - 右侧
    return         QColor(0x32, 0xcd, 0x32);      // 绿 - 躯干
}

QColor DetectionDisplayWidget::kptColor(int idx) {
    using K = COCOSkeleton::KptIdx;
    if (idx <= K::RightEar) return QColor(0xff, 0xdc, 0x1e);

    static const bool isLeft[] = {
        false,false,false,false,false,
        true,false,true,false,true,false,
        true,false,true,false,true,false
    };
    static const bool isRight[] = {
        false,false,false,false,false,
        false,true,false,true,false,true,
        false,true,false,true,false,true
    };
    if (idx < COCOSkeleton::NumKeypoints && isLeft[idx])  return QColor(0x1e, 0x90, 0xff);
    if (idx < COCOSkeleton::NumKeypoints && isRight[idx]) return QColor(0xff, 0x50, 0x50);
    return QColor(0xb4, 0xb4, 0xb4);
}

// ─── drawSkeleton ─────────────────────────────────────────────────────────────
void DetectionDisplayWidget::drawSkeleton(QPainter& p,
                                           const PoseResult& pose,
                                           float ox, float oy,
                                           float sx, float sy,
                                           int personIdx) const
{
    const auto& kpts = pose.keypoints;
    if (kpts.isEmpty()) return;

    auto toW = [&](const Keypoint& kp) -> QPointF {
        return { ox + kp.x * sx, oy + kp.y * sy };
    };

    // ── 1. 边界框（可选）────────────────────────────────────────────────────
    if (m_showBbox && !pose.boundingBox.isNull()) {
        QRectF br(ox + pose.boundingBox.x()      * sx,
                  oy + pose.boundingBox.y()      * sy,
                       pose.boundingBox.width()  * sx,
                       pose.boundingBox.height() * sy);

        p.setPen(QPen(QColor(0x00, 0xe5, 0xff, 0xb0), 1.5, Qt::DashLine));
        p.setBrush(Qt::NoBrush);
        p.drawRoundedRect(br, 4, 4);

        int cl = 12;
        int bx = (int)br.x(), by = (int)br.y();
        int bw = (int)br.width(), bh = (int)br.height();
        p.setPen(QPen(QColor(0x00, 0xe5, 0xff), 2));
        p.drawLine(bx,    by,    bx+cl, by);    p.drawLine(bx,    by,    bx,    by+cl);
        p.drawLine(bx+bw, by,    bx+bw-cl, by); p.drawLine(bx+bw, by,    bx+bw, by+cl);
        p.drawLine(bx,    by+bh, bx+cl, by+bh); p.drawLine(bx,    by+bh, bx,    by+bh-cl);
        p.drawLine(bx+bw, by+bh, bx+bw-cl, by+bh); p.drawLine(bx+bw, by+bh, bx+bw, by+bh-cl);

        p.setFont(Theme::bodyFont(9));
        QString tag = QString("P%1  %2%")
                          .arg(personIdx + 1)
                          .arg((int)(pose.poseScore * 100));
        QFontMetrics fm(p.font());
        int tw   = fm.horizontalAdvance(tag) + 12;
        int tagY = qMax((int)br.y() - 18, 0);
        p.setPen(Qt::NoPen);
        p.setBrush(QColor(0x00, 0xe5, 0xff));
        p.drawRect(bx, tagY, tw, 18);
        p.setPen(Qt::black);
        p.drawText(QRect(bx, tagY, tw, 18), Qt::AlignCenter, tag);
    }

    // ── 2. 骨架连线 ─────────────────────────────────────────────────────────
    for (const auto& [s, d] : COCOSkeleton::limbs()) {
        if (s >= kpts.size() || d >= kpts.size()) continue;
        const auto& ks = kpts[s];
        const auto& kd = kpts[d];
        if (!ks.visible || !kd.visible) continue;

        QColor col = limbColor(s, d);
        float avgScore = (ks.score + kd.score) * 0.5f;
        col.setAlpha((int)(std::clamp(avgScore, 0.f, 1.f) * 220.f) + 35);

        p.setPen(QPen(col, m_limbWidth, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
        p.drawLine(toW(ks), toW(kd));
    }

    // ── 3. 关键点 ───────────────────────────────────────────────────────────
    for (int i = 0; i < kpts.size(); ++i) {
        const auto& kp = kpts[i];
        if (!kp.visible) continue;

        QColor col = kptColor(i);
        float  r   = m_kptRadius * (0.7f + 0.5f * kp.score);

        QColor glow = col; glow.setAlpha(60);
        p.setPen(Qt::NoPen);
        p.setBrush(glow);
        p.drawEllipse(toW(kp), (double)(r * 1.8), (double)(r * 1.8));

        p.setPen(QPen(Qt::white, 1.0));
        p.setBrush(col);
        p.drawEllipse(toW(kp), (double)r, (double)r);

        if (m_showLabels) {
            p.setPen(Qt::white);
            p.setFont(Theme::bodyFont(8));
            p.drawText(toW(kp) + QPointF(r + 2, 4),
                       COCOSkeleton::kptName(i));
        }
    }
}

// ─── paintEvent ───────────────────────────────────────────────────────────────
void DetectionDisplayWidget::paintEvent(QPaintEvent*) {
    QElapsedTimer timer;
    timer.start();

    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing);
    p.setRenderHint(QPainter::SmoothPixmapTransform);

    int W = width(), H = height();

    // 背景
    p.fillRect(rect(), QColor(0x02, 0x08, 0x20));

    // 视频帧（后台线程已完成缩放）
    if (!m_scaledFrame.isNull()) {
        p.drawImage(m_videoRect, m_scaledFrame);

        // 叠加骨架
        for (int i = 0; i < m_poses.size(); ++i)
            drawSkeleton(p, m_poses[i],
                         (float)m_videoRect.x(), (float)m_videoRect.y(),
                         m_scaleX, m_scaleY, i);
    } else {
        // 无视频信号占位文字
        p.setFont(Theme::sectionFont(14));
        p.setPen(Theme::CyanDim);
        p.drawText(rect(), Qt::AlignCenter,
                   QStringLiteral("等待视频信号..."));
    }

    // 面板边框
    p.setPen(QPen(Theme::Cyan, 1));
    p.setBrush(Qt::NoBrush);
    p.drawRect(rect().adjusted(0, 0, -1, -1));

    // 角装饰
    auto corner = [&](int x, int y, int dx, int dy) {
        p.setPen(QPen(Theme::Cyan, 2));
        p.drawLine(x, y, x + dx * 20, y);
        p.drawLine(x, y, x, y + dy * 20);
    };
    corner(1,     1,      1,  1);
    corner(W - 2, 1,     -1,  1);
    corner(1,     H - 2,  1, -1);
    corner(W - 2, H - 2, -1, -1);

    // HUD：后端标签（左上角）
    {
        QString badge = QString("◈ %1").arg(m_backendName);
        p.setFont(Theme::bodyFont(8));
        QFontMetrics fm(p.font());
        int bw = fm.horizontalAdvance(badge) + 12;
        p.setPen(Qt::NoPen);
        p.setBrush(QColor(0x00, 0xe5, 0xff, 0xaa));
        p.drawRoundedRect(QRect(6, 6, bw, 18), 3, 3);
        p.setPen(Qt::black);
        p.drawText(QRect(6, 6, bw, 18), Qt::AlignCenter, badge);
    }

    // HUD：FPS（右上角）
    {
        QString fpsStr = QString("FPS: %1 | Render: %2ms")
                             .arg(m_fps, 0, 'f', 1).arg(timer.elapsed());
        p.setFont(Theme::bodyFont(8));
        QFontMetrics fm(p.font());
        int fw = fm.horizontalAdvance(fpsStr) + 12;
        p.setPen(Qt::NoPen);
        p.setBrush(QColor(0, 0, 0, 0x99));
        p.drawRoundedRect(QRect(W - fw - 6, 6, fw, 18), 3, 3);
        p.setPen(Theme::Green);
        p.drawText(QRect(W - fw - 6, 6, fw, 18), Qt::AlignCenter, fpsStr);
    }

    // HUD：姿态计数（左下角 - 紧贴底部单行）
    {
        QString dStr = QString("姿态识别: %1 人 | 丢帧: %2")
                           .arg(m_poseCount).arg(m_droppedFrames);
        p.setFont(Theme::bodyFont(8));
        QFontMetrics fm(p.font());
        int dw = fm.horizontalAdvance(dStr) + 12;
        p.setPen(Qt::NoPen);
        p.setBrush(QColor(0, 0, 0, 0x99));
        p.drawRoundedRect(QRect(6, H - 26, dw, 18), 3, 3);
        p.setPen(Theme::Yellow);
        p.drawText(QRect(6, H - 26, dw, 18), Qt::AlignCenter, dStr);
    }

    // ── 新增：关节速度面板（左下角半透明滚动列表）──────────────────────────
    // 显示当前帧每个人主要关节点的运动速度（像素/秒），颜色随速度变化
    do {
        if (!m_showVelocityPanel || m_poses.isEmpty() || m_velocities.isEmpty()) break;

        // 安全检查：m_velocities 可能与 m_poses 在帧缓存路径中不同步
        const int displayPerson = 0;
        if (displayPerson >= m_velocities.size()) break;

        struct VelEntry { int ki; float speed; float vx; float vy; };
        QVector<VelEntry> topVels;

        {
            const auto& vels = m_velocities[displayPerson];
            int velSz = qMin(vels.size(), (int)COCOSkeleton::NumKeypoints);
            for (int ki = 0; ki < velSz; ++ki) {
                if (vels[ki].valid && vels[ki].speed > 0.5f) {
                    VelEntry ve;
                    ve.ki    = ki;
                    ve.speed = vels[ki].speed;
                    ve.vx    = vels[ki].vx;
                    ve.vy    = vels[ki].vy;
                    topVels.append(ve);
                }
            }
            std::sort(topVels.begin(), topVels.end(),
                      [](const VelEntry& a, const VelEntry& b){ return a.speed > b.speed; });
            if (topVels.size() > 8) topVels.resize(8);
        }

        const int vPanelW  = 210;
        const int vRowH    = 15;
        const int vPadX    = 7;
        const int vPadY    = 4;
        const int vTitleH  = 18;
        const int vRows    = topVels.size();
        const int vPanelH  = vTitleH + vPadY * 2 + vRows * vRowH + 4;
        const int vPanelX  = 6;
        const int vPanelY  = H - 32 - vPanelH;

        // 固定列宽（防止负数宽度导致崩溃）
        const int nameColW = 72;
        const int barColW  = 70;
        const int barColX  = vPanelX + vPadX + nameColW + 2;
        const int valColX  = barColX + barColW + 4;
        const int valColW  = (vPanelX + vPanelW - vPadX) - valColX;

        if (valColW <= 0) break;  // 面板太窄则不绘制

        if (vPanelY > 60 && vRows > 0) {
            // 半透明背景
            p.setPen(QPen(QColor(0x1e, 0x90, 0xff, 0x70), 1));
            p.setBrush(QColor(0x02, 0x08, 0x20, 0xd0));
            p.drawRoundedRect(QRect(vPanelX, vPanelY, vPanelW, vPanelH), 4, 4);

            // 标题
            p.setFont(Theme::sectionFont(8));
            p.setPen(QColor(0x1e, 0x90, 0xff));
            QString vTitle = m_poses.size() > 1
                ? QString("◈ 关节速度  P1/%1人").arg(m_poses.size())
                : QString("◈ 关节速度估算  (px/s)");
            p.drawText(QRect(vPanelX + vPadX, vPanelY + vPadY,
                             vPanelW - vPadX * 2, vTitleH),
                       Qt::AlignLeft | Qt::AlignVCenter, vTitle);

            // 分隔线
            p.setPen(QPen(QColor(0x1a, 0x3a, 0x6a), 1));
            p.drawLine(vPanelX + 4, vPanelY + vTitleH + vPadY,
                       vPanelX + vPanelW - 4, vPanelY + vTitleH + vPadY);

            // 每行：关键点名 | 速度条 | 数值
            int rowY = vPanelY + vTitleH + vPadY + 4;
            float maxSpeed = topVels.isEmpty() ? 1.f : qMax(topVels.first().speed, 1.f);

            for (const auto& ve : topVels) {
                // 关键点名称（左/右颜色区分）
                p.setFont(Theme::bodyFont(7));
                p.setPen(kptColor(ve.ki));
                p.drawText(QRect(vPanelX + vPadX, rowY, nameColW, vRowH),
                           Qt::AlignLeft | Qt::AlignVCenter,
                           QString(COCOSkeleton::kptName(ve.ki)));

                // 速度条背景
                int barH = 6;
                int barY = rowY + (vRowH - barH) / 2;
                float ratio = std::clamp(ve.speed / (maxSpeed * 1.1f), 0.f, 1.f);
                int fillW = qMax(0, (int)(ratio * barColW));

                p.setPen(Qt::NoPen);
                p.setBrush(QColor(0x1a, 0x2a, 0x4a));
                p.drawRoundedRect(QRect(barColX, barY, barColW, barH), 2, 2);

                // 速度条填充（低绿→中黄→高红）
                QColor barColor;
                if (ratio < 0.33f)      barColor = QColor(0x32, 0xcd, 0x32);
                else if (ratio < 0.66f) barColor = QColor(0xff, 0xdc, 0x1e);
                else                    barColor = QColor(0xff, 0x50, 0x50);
                if (fillW > 0) {
                    p.setBrush(barColor);
                    p.drawRoundedRect(QRect(barColX, barY, fillW, barH), 2, 2);
                }

                // 速度数值 + 方向箭头
                QString arrow;
                if (std::abs(ve.vx) >= std::abs(ve.vy))
                    arrow = (ve.vx >= 0.f) ? "→" : "←";
                else
                    arrow = (ve.vy >= 0.f) ? "↓" : "↑";
                QString valStr = QString("%1 %2").arg((int)ve.speed).arg(arrow);

                p.setPen(barColor);
                p.drawText(QRect(valColX, rowY, valColW, vRowH),
                           Qt::AlignLeft | Qt::AlignVCenter, valStr);

                rowY += vRowH;
            }
        }

        // 无有效速度数据时显示提示
        if (vRows == 0) {
            p.setFont(Theme::bodyFont(7));
            QString hint = QStringLiteral("关节静止 / 等待速度数据...");
            int hw = QFontMetrics(p.font()).horizontalAdvance(hint) + 12;
            int hy = H - 52;
            if (hy > 60 && hw > 0) {
                p.setPen(Qt::NoPen);
                p.setBrush(QColor(0, 0, 0, 0x80));
                p.drawRoundedRect(QRect(6, hy, hw, 16), 3, 3);
                p.setPen(Theme::TextMuted);
                p.drawText(QRect(6, hy, hw, 16), Qt::AlignCenter, hint);
            }
        }
    } while (false);

    // HUD：处理延迟（右下角）
    if (m_processingDelay > 0) {
        QString delayStr = QString("延迟: %1ms | 缩放: %2ms")
                               .arg(m_processingDelay).arg(m_renderLatency);
        p.setFont(Theme::bodyFont(8));
        QFontMetrics fm(p.font());
        int dw = fm.horizontalAdvance(delayStr) + 12;
        QColor delayColor = m_processingDelay < 30 ? Theme::Green  :
                            m_processingDelay < 60 ? Theme::Yellow : Theme::Red;
        p.setPen(Qt::NoPen);
        p.setBrush(QColor(0, 0, 0, 0x99));
        p.drawRoundedRect(QRect(W - dw - 6, H - 26, dw, 18), 3, 3);
        p.setPen(delayColor);
        p.drawText(QRect(W - dw - 26, H - 26, dw, 18), Qt::AlignCenter, delayStr);
    }

    // ── 新增：关键点坐标叠加面板 ──────────────────────────────────────────────
    // 将检测到的每个人的可见关键点(名称 + 像素坐标 + 置信度)绘制在视频右侧半透明面板中
    if (m_showKeypointCoords && !m_poses.isEmpty()) {
        // 面板参数
        const int panelW   = 175;
        const int rowH     = 14;
        const int padX     = 6;
        const int padY     = 4;
        const int titleH   = 18;

        // 对每个检测到的人分别绘制一个坐标列表面板
        int panelY = 32;   // 从顶部 32px 开始，避免与 FPS HUD 重叠
        const int maxPersons = qMin(m_poses.size(), 2);  // 最多同时显示 2 人的面板

        for (int pi = 0; pi < maxPersons; ++pi) {
            const PoseResult& pose = m_poses[pi];
            if (pose.keypoints.isEmpty()) continue;

            // 统计本人可见关键点数
            int visibleCount = 0;
            for (const auto& kp : pose.keypoints)
                if (kp.visible) ++visibleCount;
            if (visibleCount == 0) continue;

            int panelH = titleH + padY * 2 + visibleCount * rowH + 4;
            int panelX = W - panelW - 6;

            // 半透明背景
            p.setPen(QPen(QColor(0x00, 0xe5, 0xff, 0x80), 1));
            p.setBrush(QColor(0x02, 0x08, 0x20, 0xcc));
            p.drawRoundedRect(QRect(panelX, panelY, panelW, panelH), 4, 4);

            // 标题行
            p.setFont(Theme::sectionFont(8));
            p.setPen(QColor(0x00, 0xe5, 0xff));
            QString title = QString("◈ P%1  置信度 %2%")
                                .arg(pi + 1)
                                .arg((int)(pose.poseScore * 100));
            p.drawText(QRect(panelX + padX, panelY + padY, panelW - padX * 2, titleH),
                       Qt::AlignLeft | Qt::AlignVCenter, title);

            // 分隔线
            p.setPen(QPen(QColor(0x1a, 0x3a, 0x6a), 1));
            p.drawLine(panelX + 4, panelY + titleH + padY,
                       panelX + panelW - 4, panelY + titleH + padY);

            // 每个可见关键点一行：名称  (x, y)  conf%
            int rowY = panelY + titleH + padY + 4;
            p.setFont(Theme::bodyFont(7));

            for (int ki = 0; ki < pose.keypoints.size(); ++ki) {
                const Keypoint& kp = pose.keypoints[ki];
                if (!kp.visible) continue;

                // 关键点序号和名称（带左侧/右侧颜色区分）
                QColor nameColor;
                const char* name = COCOSkeleton::kptName(ki);
                // 左侧 = 蓝色，右侧 = 红色，中轴 = 灰色
                static const bool isLeft[] = {
                    false, true, false, true, false,
                    true, false, true, false, true, false,
                    true, false, true, false, true, false
                };
                static const bool isRight[] = {
                    false, false, true, false, true,
                    false, true, false, true, false, true,
                    false, true, false, true, false, true
                };
                if (ki < COCOSkeleton::NumKeypoints && isLeft[ki])
                    nameColor = QColor(0x4d, 0xb8, 0xff);
                else if (ki < COCOSkeleton::NumKeypoints && isRight[ki])
                    nameColor = QColor(0xff, 0x80, 0x80);
                else
                    nameColor = QColor(0xaa, 0xcc, 0xdd);

                // 名称列
                p.setPen(nameColor);
                QString nameStr = QString("%1").arg(name);
                p.drawText(QRect(panelX + padX, rowY, 78, rowH),
                           Qt::AlignLeft | Qt::AlignVCenter, nameStr);

                // 坐标 + 置信度列
                p.setPen(Theme::TextSecondary);
                // 将原图像素坐标映射回视频帧坐标（关键点存储的是原图坐标）
                QString coordStr = QString("(%1,%2) %3%")
                    .arg((int)kp.x)
                    .arg((int)kp.y)
                    .arg((int)(kp.score * 100));
                p.drawText(QRect(panelX + padX + 78, rowY, panelW - padX * 2 - 78, rowH),
                           Qt::AlignLeft | Qt::AlignVCenter, coordStr);

                rowY += rowH;
            }

            panelY += panelH + 6;  // 下一个人的面板位置
        }

        // 若检测到超过 2 人，显示省略提示
        if (m_poses.size() > 2) {
            p.setFont(Theme::bodyFont(7));
            p.setPen(Theme::TextMuted);
            QString moreStr = QString("+%1 人（面板仅展示前2人）").arg(m_poses.size() - 2);
            int mw = QFontMetrics(p.font()).horizontalAdvance(moreStr) + 12;
            p.setPen(Qt::NoPen);
            p.setBrush(QColor(0, 0, 0, 0x80));
            p.drawRoundedRect(QRect(W - mw - 6, panelY, mw, 16), 3, 3);
            p.setPen(Theme::TextMuted);
            p.drawText(QRect(W - mw - 6, panelY, mw, 16), Qt::AlignCenter, moreStr);
        }
    }
}
