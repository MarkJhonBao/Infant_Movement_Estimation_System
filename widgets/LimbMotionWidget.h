#pragma once
/**
 * LimbMotionWidget.h
 *
 * 左下角四肢运动监测面板：
 *  ① 实时曲线图：显示左臂/右臂/左腿/右腿代表关节点的位移历史（滚动窗口）
 *  ② 速度数字：由曲线差分估算，单位 px/s，带方向箭头与色阶
 *
 * 数据入口：pushPoseResult(poses, timestampMs)
 *   —— 由 MainWindow 在 onPoseResultsReady 中调用，无需额外线程。
 *
 * 绘制策略：纯 QPainter，不依赖 Qt Charts，零额外模块。
 */

#include <QWidget>
#include <QVector>
#include <QTimer>
#include <QQueue>
#include "../detector/PoseResult.h"

class LimbMotionWidget : public QWidget {
    Q_OBJECT
public:
    explicit LimbMotionWidget(QWidget* parent = nullptr);

    // ── 数据推送（主线程调用）────────────────────────────────────────────────
    void pushPoseResult(const QVector<PoseResult>& poses, qint64 timestampMs);

    // ── 可选：曲线滚动窗口长度（样本数，默认 120 ≈ 4 秒 @30fps）─────────────
    void setWindowSize(int n);

    QSize sizeHint() const override { return {320, 240}; }
    QSize minimumSizeHint() const override { return {200, 160}; }

protected:
    void paintEvent(QPaintEvent*) override;

private:
    // ── 四肢通道定义 ─────────────────────────────────────────────────────────
    struct LimbChannel {
        QString label;          // 显示名称
        QColor  color;          // 曲线颜色
        int     kpA;            // 代表关键点A（近端）
        int     kpB;            // 代表关键点B（远端）—— 取均值作为代表坐标
    };

    // ── 单采样点 ──────────────────────────────────────────────────────────────
    struct Sample {
        qint64 ts;      // 时间戳 ms
        float  x;       // 代表坐标 x（原图像素）
        float  y;       // 代表坐标 y
        bool   valid;   // 关键点是否可见
    };

    // ── 速度估算结果（每通道）────────────────────────────────────────────────
    struct VelResult {
        float speed{0.f};       // 合速度 px/s（平滑后）
        float vx{0.f};
        float vy{0.f};
        bool  active{false};    // 有足够数据
    };

    static const int kChannels = 4;
    LimbChannel   m_channels[kChannels];
    QQueue<Sample> m_history[kChannels];   // 各通道采样历史
    VelResult      m_vel[kChannels];       // 最新速度

    int m_windowSize{120};                 // 保留最近 N 个样本

    // ── 速度估算（对最近 K 个样本做线性差分平均）─────────────────────────────
    void updateVelocity(int ch);

    // ── 绘制子区域 ────────────────────────────────────────────────────────────
    void drawTitle  (QPainter& p, const QRect& rc) const;
    void drawCurves (QPainter& p, const QRect& rc) const;
    void drawSpeeds (QPainter& p, const QRect& rc) const;

    // ── 曲线归一化辅助 ────────────────────────────────────────────────────────
    // 返回某通道历史中位移（相对第一个有效点）的浮点序列
    QVector<float> buildDisplacementSeries(int ch) const;
};
