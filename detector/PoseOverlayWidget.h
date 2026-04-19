#pragma once
#include "DetectorBase.h"  // QVector<DetectionResult>
#include "PoseResult.h"
#include <QWidget>
#include <QImage>
#include <QVector>
#include <QPainter>
#include <QPen>
#include <QBrush>
#include <QFont>
#include <QDateTime>
#include <QTimer>

// ─────────────────────────────────────────────────────────────────────────────
//  PoseOverlayWidget
//
//  大屏实时姿态渲染组件。将视频帧 + 姿态骨架叠加绘制，无需额外 OpenGL。
//
//  功能
//  ────
//  • 按 COCO-17 骨架定义绘制彩色肢体连接线
//  • 绘制可见关键点（实心圆 + 可选标签）
//  • 绘制人体包围框 + 置信度
//  • 左上角显示 FPS / 帧延迟信息
//  • 支持多人（所有 PoseResult 全部渲染）
//  • 所有参数可在运行时通过 setter 调整，无需重建
//
//  接线示例
//  ────────
//    PoseOverlayWidget* overlay = new PoseOverlayWidget(this);
//    overlay->showFullScreen();
//
//    // 每帧：先更新背景，再更新骨架
//    connect(camera, &CameraCapture::frameReady,
//            overlay, &PoseOverlayWidget::setFrame);
//    connect(poseMgr, &PoseDetectorManager::poseReady,
//            overlay, &PoseOverlayWidget::updatePoses);
// ─────────────────────────────────────────────────────────────────────────────
class PoseOverlayWidget : public QWidget {
    Q_OBJECT
public:
    explicit PoseOverlayWidget(QWidget* parent = nullptr)
        : QWidget(parent)
    {
        setAutoFillBackground(false);
        setAttribute(Qt::WA_OpaquePaintEvent);
        setMinimumSize(640, 480);

        // FPS 计时更新
        m_fpsTimer = new QTimer(this);
        connect(m_fpsTimer, &QTimer::timeout, this, [this]{
            m_displayFps = m_frameCount;
            m_frameCount = 0;
        });
        m_fpsTimer->start(1000);
    }

    // ── 外观配置 ──────────────────────────────────────────────────────────────
    void setLimbWidth(float w)              { m_limbWidth = w; }
    void setKeypointRadius(float r)         { m_kptRadius = r; }
    void setShowLabels(bool show)           { m_showLabels = show; update(); }
    void setShowBoundingBox(bool show)      { m_showBbox = show; update(); }
    void setShowFps(bool show)              { m_showFps = show; update(); }
    void setShowConfidence(bool show)       { m_showConf = show; update(); }
    void setScaleTolerance(float t)         { m_scaleTolerance = t; }

public slots:
    // ── 数据更新 ──────────────────────────────────────────────────────────────
    // 设置背景视频帧（在视频线程中调用，线程安全）
    void setFrame(const QImage& frame) {
        m_frame       = frame;
        m_frameTime   = QDateTime::currentMSecsSinceEpoch();
        m_frameCount++;
        update();
    }

    // 更新姿态结果（在 Qt 主线程 / QueuedConnection 中调用）
    void updatePoses(const QVector<PoseResult>& poses) {
        m_poses       = poses;
        m_poseTime    = QDateTime::currentMSecsSinceEpoch();
        update();
    }



    // 新增 public slot
    void setBackendName(const QString& name) {
        m_backendName = name;
        update();
    }

    // 新增合并更新 slot（原子操作，避免帧与姿态不同步）
    void updateFrame(const QImage& frame,
                     const QVector<PoseResult>& poses,
                     const QVector<DetectionResult>& dets = {})
    {
        m_frame     = frame;
        m_poses     = poses;
        m_latestDets = dets;    // 需同步增加 QVector<DetectionResult> m_latestDets 成员
        m_frameCount++;
        update();
    }

protected:
    void paintEvent(QPaintEvent*) override {
        QPainter p(this);
        p.setRenderHint(QPainter::Antialiasing);
        p.setRenderHint(QPainter::SmoothPixmapTransform);

        // ── 背景：填充黑色 + 居中缩放视频帧 ──────────────────────────────────
        p.fillRect(rect(), Qt::black);

        QRect videoRect;
        if (!m_frame.isNull()) {
            QSize scaled = m_frame.size().scaled(size(), Qt::KeepAspectRatio);
            int ox = (width()  - scaled.width())  / 2;
            int oy = (height() - scaled.height()) / 2;
            videoRect = QRect(QPoint(ox, oy), scaled);
            p.drawImage(videoRect, m_frame);
        } else {
            videoRect = rect();
        }

        // 坐标变换：原图像素 → widget 像素
        float scaleX = (float)videoRect.width()  / (m_frame.isNull() ? width()  : m_frame.width());
        float scaleY = (float)videoRect.height() / (m_frame.isNull() ? height() : m_frame.height());
        float ox     = videoRect.x();
        float oy     = videoRect.y();

        // ── 绘制所有人的姿态 ──────────────────────────────────────────────────
        for (int pi = 0; pi < m_poses.size(); ++pi)
            drawPose(p, m_poses[pi], ox, oy, scaleX, scaleY, pi);

        // ── HUD：FPS + 延迟 ──────────────────────────────────────────────────
        if (m_showFps)
            drawHud(p);
    }

private:
    // 新增成员变量
    QString m_backendName;
    QVector<DetectionResult> m_latestDets;
    // ─── 骨架颜色方案（左=蓝，右=红，躯干=绿，头=黄）────────────────────────
    static QColor limbColor(int src, int dst) {
        using K = COCOSkeleton::KptIdx;
        bool leftSide  = (src == K::LeftShoulder  || src == K::LeftElbow   ||
                          src == K::LeftWrist      || src == K::LeftHip     ||
                          src == K::LeftKnee       || src == K::LeftAnkle   ||
                          dst == K::LeftShoulder   || dst == K::LeftElbow   ||
                          dst == K::LeftWrist       || dst == K::LeftHip    ||
                          dst == K::LeftKnee        || dst == K::LeftAnkle);
        bool rightSide = (src == K::RightShoulder || src == K::RightElbow  ||
                          src == K::RightWrist     || src == K::RightHip    ||
                          src == K::RightKnee      || src == K::RightAnkle  ||
                          dst == K::RightShoulder  || dst == K::RightElbow  ||
                          dst == K::RightWrist      || dst == K::RightHip   ||
                          dst == K::RightKnee       || dst == K::RightAnkle);
        bool headArea  = (src <= K::RightEar || dst <= K::RightEar);

        if (headArea)  return QColor(255, 220,  30);  // 黄
        if (leftSide)  return QColor( 30, 144, 255);  // 蓝
        if (rightSide) return QColor(255,  80,  80);  // 红
        return             QColor( 50, 205,  50);     // 绿（躯干）
    }

    static QColor kptColor(int idx) {
        // 头部=黄，左侧=蓝，右侧=红
        using K = COCOSkeleton::KptIdx;
        if (idx <= K::RightEar) return QColor(255, 220, 30);
        static const bool leftIdx[]  = {false,false,false,false,false,
                                        true,false,true,false,true,false,
                                        true,false,true,false,true,false};
        static const bool rightIdx[] = {false,false,false,false,false,
                                        false,true,false,true,false,true,
                                        false,true,false,true,false,true};
        if (idx < 17 && leftIdx[idx])  return QColor(30, 144, 255);
        if (idx < 17 && rightIdx[idx]) return QColor(255, 80,  80);
        return QColor(180, 180, 180);
    }

    void drawPose(QPainter& p,
                  const PoseResult& pose,
                  float ox, float oy, float sx, float sy,
                  int personIdx)
    {
        const auto& kpts = pose.keypoints;
        if (kpts.isEmpty()) return;

        auto toW = [&](const Keypoint& kp) -> QPointF {
            return { ox + kp.x * sx, oy + kp.y * sy };
        };

        // ── 1. 包围框 ─────────────────────────────────────────────────────────
        if (m_showBbox && !pose.boundingBox.isNull()) {
            QRectF br(ox + pose.boundingBox.x() * sx,
                      oy + pose.boundingBox.y() * sy,
                      pose.boundingBox.width()  * sx,
                      pose.boundingBox.height() * sy);
            QPen bpen(QColor(0, 200, 255, 180), 1.5, Qt::DashLine);
            p.setPen(bpen);
            p.setBrush(Qt::NoBrush);
            p.drawRoundedRect(br, 4, 4);

            if (m_showConf) {
                p.setPen(QColor(0, 200, 255));
                p.setFont(QFont("Arial", 11, QFont::Bold));
                p.drawText(br.topLeft() + QPointF(4, -4),
                           QString("Person %1  %.0f%%")
                               .arg(personIdx + 1)
                               .arg(pose.poseScore * 100));
            }
        }

        // ── 2. 骨架连接线 ──────────────────────────────────────────────────────
        for (const auto& [s, d] : COCOSkeleton::limbs()) {
            if (s >= kpts.size() || d >= kpts.size()) continue;
            const auto& ks = kpts[s];
            const auto& kd = kpts[d];
            if (!ks.visible || !kd.visible) continue;

            QColor col = limbColor(s, d);
            float alpha = std::min(1.f, (ks.score + kd.score) / 2.f) * 255;
            col.setAlpha((int)alpha);

            QPen pen(col, m_limbWidth, Qt::SolidLine, Qt::RoundCap);
            p.setPen(pen);
            p.drawLine(toW(ks), toW(kd));
        }

        // ── 3. 关键点圆点 ──────────────────────────────────────────────────────
        for (int i = 0; i < kpts.size(); ++i) {
            const auto& kp = kpts[i];
            if (!kp.visible) continue;

            QColor col = kptColor(i);
            float r    = m_kptRadius * (0.8f + 0.4f * kp.score);

            p.setPen(QPen(Qt::white, 1.2f));
            p.setBrush(col);
            p.drawEllipse(toW(kp), (double)r, (double)r);

            // 关键点名称标签（可选）
            if (m_showLabels) {
                p.setPen(Qt::white);
                p.setFont(QFont("Arial", 9));
                p.drawText(toW(kp) + QPointF(r + 2, 4),
                           COCOSkeleton::kptName(i));
            }
        }
    }

    void drawHud(QPainter& p) {
        qint64 now  = QDateTime::currentMSecsSinceEpoch();
        int    lag  = (int)(now - m_poseTime);  // 推理→显示延迟 ms

        QString info = QString("FPS: %1   推理延迟: %2 ms   人数: %3")
                           .arg(m_displayFps)
                           .arg(lag)
                           .arg(m_poses.size());

        QFont f("Consolas", 13, QFont::Bold);
        p.setFont(f);

        QFontMetrics fm(f);
        QRect textRect = fm.boundingRect(info).adjusted(-8, -4, 8, 4);
        textRect.moveTo(12, 12);

        p.fillRect(textRect, QColor(0, 0, 0, 160));
        p.setPen(QColor(0, 255, 160));
        p.drawText(textRect, Qt::AlignVCenter | Qt::AlignLeft, info);
    }

    // ── 数据 ───────────────────────────────────────────────────────────────────
    QImage              m_frame;
    QVector<PoseResult> m_poses;

    // ── 样式参数 ───────────────────────────────────────────────────────────────
    float m_limbWidth      {3.5f};
    float m_kptRadius      {5.0f};
    float m_scaleTolerance {0.05f};
    bool  m_showLabels     {false};
    bool  m_showBbox       {true};
    bool  m_showFps        {true};
    bool  m_showConf       {true};

    // ── FPS 统计 ───────────────────────────────────────────────────────────────
    QTimer*  m_fpsTimer   {nullptr};
    int      m_frameCount {0};
    int      m_displayFps {0};
    qint64   m_frameTime  {0};
    qint64   m_poseTime   {0};
};
