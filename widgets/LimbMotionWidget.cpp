/**
 * LimbMotionWidget.cpp
 *
 * 四肢运动曲线与速度面板
 * ─────────────────────────────────────────────────────────────────────────────
 * 布局（从上至下）：
 *   [标题行]                 约 20px
 *   [四通道滚动曲线图]        弹性，占 ~60% 高度
 *   [速度数字行 × 4 通道]    约 14px × 4 = 56px
 *
 * 曲线绘制逻辑：
 *   - 每通道维护一个 QQueue<Sample>（最多 windowSize 条）
 *   - 以各通道代表关键点 (kpA+kpB)/2 的 y 坐标作为曲线纵轴
 *     （y 坐标变化直观反映上下运动；x 坐标变化体现左右运动）
 *   - 先将历史 y 序列对"首点"做相对位移，再归一化到绘图区高度
 *   - X 轴按采样索引等间距铺满绘图宽度
 *
 * 速度估算逻辑：
 *   - 取最近 velWindow（默认 8）个有效样本
 *   - 对每对相邻样本计算 (Δx, Δy, Δt)，得 vx_i / vy_i
 *   - 取均值后做指数平滑（α=0.3），输出平滑速度
 */

#include "LimbMotionWidget.h"
#include "DashboardTheme.h"

#include <QPainter>
#include <QPainterPath>
#include <QFontMetrics>
#include <cmath>
#include <algorithm>

using namespace COCOSkeleton;

// ─── 通道定义（COCO-17 关键点索引）──────────────────────────────────────────
//  左臂  : LeftShoulder(5)  + LeftWrist(9)
//  右臂  : RightShoulder(6) + RightWrist(10)
//  左腿  : LeftHip(11)      + LeftAnkle(15)
//  右腿  : RightHip(12)     + RightAnkle(16)
// ─────────────────────────────────────────────────────────────────────────────

LimbMotionWidget::LimbMotionWidget(QWidget* parent)
    : QWidget(parent)
{
    // 初始化四通道元信息
    m_channels[0] = { "左臂", QColor(0x1e, 0x90, 0xff),  5,  9 };  // 蓝
    m_channels[1] = { "右臂", QColor(0xff, 0x6e, 0x1e),  6, 10 };  // 橙
    m_channels[2] = { "左腿", QColor(0x00, 0xff, 0xa0), 11, 15 };  // 绿
    m_channels[3] = { "右腿", QColor(0xff, 0x40, 0x40), 12, 16 };  // 红

    // 背景透明，由 DashboardWidget 提供底色
    setAttribute(Qt::WA_OpaquePaintEvent, false);
    setMinimumSize(200, 180);
}

// ─────────────────────────────────────────────────────────────────────────────
//  setWindowSize
// ─────────────────────────────────────────────────────────────────────────────
void LimbMotionWidget::setWindowSize(int n) {
    m_windowSize = qMax(20, n);
}

// ─────────────────────────────────────────────────────────────────────────────
//  pushPoseResult  —— 主线程调用，在 onPoseResultsReady 中触发
// ─────────────────────────────────────────────────────────────────────────────
void LimbMotionWidget::pushPoseResult(const QVector<PoseResult>& poses,
                                      qint64 timestampMs)
{
    // 只取第一个检测到的人（早产儿单人场景）
    const PoseResult* pose = poses.isEmpty() ? nullptr : &poses[0];

    for (int ch = 0; ch < kChannels; ++ch) {
        const LimbChannel& lc = m_channels[ch];
        Sample s;
        s.ts    = timestampMs;
        s.valid = false;
        s.x     = 0.f;
        s.y     = 0.f;

        if (pose && pose->keypoints.size() > qMax(lc.kpA, lc.kpB)) {
            const Keypoint& kA = pose->keypoints[lc.kpA];
            const Keypoint& kB = pose->keypoints[lc.kpB];
            // 两端点均可见时取均值；否则取可见的那个
            if (kA.visible && kB.visible) {
                s.x     = (kA.x + kB.x) * 0.5f;
                s.y     = (kA.y + kB.y) * 0.5f;
                s.valid = true;
            } else if (kA.visible) {
                s.x = kA.x; s.y = kA.y; s.valid = true;
            } else if (kB.visible) {
                s.x = kB.x; s.y = kB.y; s.valid = true;
            }
        }

        m_history[ch].enqueue(s);
        while (m_history[ch].size() > m_windowSize)
            m_history[ch].dequeue();

        updateVelocity(ch);
    }

    update();
}

// ─────────────────────────────────────────────────────────────────────────────
//  updateVelocity  —— 对最近若干有效样本做差分 + 指数平滑
// ─────────────────────────────────────────────────────────────────────────────
void LimbMotionWidget::updateVelocity(int ch) {
    const int velWin = 10;   // 差分窗口（样本数）
    const float alpha = 0.3f; // 指数平滑系数

    auto& q = m_history[ch];
    auto& vel = m_vel[ch];

    // 收集最近 velWin 个有效样本
    struct VSample { float x, y; qint64 ts; };
    QVector<VSample> valid;
    valid.reserve(velWin);
    for (int i = q.size() - 1; i >= 0 && valid.size() < velWin; --i)
        if (q[i].valid)
            valid.prepend({q[i].x, q[i].y, q[i].ts});

    if (valid.size() < 2) {
        vel.active = false;
        vel.speed  = 0.f;
        vel.vx     = 0.f;
        vel.vy     = 0.f;
        return;
    }

    // 对相邻有效样本差分
    float sumVx = 0.f, sumVy = 0.f;
    int cnt = 0;
    for (int i = 1; i < valid.size(); ++i) {
        double dt = (valid[i].ts - valid[i-1].ts) / 1000.0;
        if (dt < 0.001) continue;
        sumVx += (valid[i].x - valid[i-1].x) / (float)dt;
        sumVy += (valid[i].y - valid[i-1].y) / (float)dt;
        ++cnt;
    }
    if (cnt == 0) { vel.active = false; return; }

    float rawVx = sumVx / cnt;
    float rawVy = sumVy / cnt;
    float rawSpeed = std::sqrt(rawVx * rawVx + rawVy * rawVy);

    if (vel.active) {
        // 指数平滑
        vel.vx    = alpha * rawVx    + (1.f - alpha) * vel.vx;
        vel.vy    = alpha * rawVy    + (1.f - alpha) * vel.vy;
        vel.speed = alpha * rawSpeed + (1.f - alpha) * vel.speed;
    } else {
        vel.vx    = rawVx;
        vel.vy    = rawVy;
        vel.speed = rawSpeed;
    }
    vel.active = true;
}

// ─────────────────────────────────────────────────────────────────────────────
//  buildDisplacementSeries  —— 历史 y 坐标 → 相对位移序列
//  返回 float 序列（仅 valid 样本，invalid 插值为上一有效值）
// ─────────────────────────────────────────────────────────────────────────────
QVector<float> LimbMotionWidget::buildDisplacementSeries(int ch) const {
    const auto& q = m_history[ch];
    if (q.isEmpty()) return {};

    QVector<float> series;
    series.reserve(q.size());

    float baseY   = 0.f;
    bool  hasBase = false;
    float lastY   = 0.f;

    for (const auto& s : q) {
        float y = s.valid ? s.y : lastY;
        if (!hasBase && s.valid) {
            baseY   = y;
            hasBase = true;
        }
        if (s.valid) lastY = y;
        series.append(y - baseY);  // 相对位移（可正可负）
    }
    return series;
}

// ─────────────────────────────────────────────────────────────────────────────
//  paintEvent
// ─────────────────────────────────────────────────────────────────────────────
void LimbMotionWidget::paintEvent(QPaintEvent*) {
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing);

    int W = width(), H = height();

    // ── 背景 ───────────────────────────────────────────────────────────────
    p.setPen(QPen(QColor(0x00, 0xe5, 0xff, 0x60), 1));
    p.setBrush(QColor(0x04, 0x0d, 0x2c, 0xe8));
    p.drawRoundedRect(QRect(0, 0, W, H), 4, 4);

    // ── 角装饰 ─────────────────────────────────────────────────────────────
    const int cL = 10;
    p.setPen(QPen(Theme::Cyan, 1));
    p.drawLine(0, 0, cL, 0);    p.drawLine(0, 0, 0, cL);
    p.drawLine(W-1, 0, W-1-cL, 0); p.drawLine(W-1, 0, W-1, cL);
    p.drawLine(0, H-1, cL, H-1); p.drawLine(0, H-1, 0, H-1-cL);
    p.drawLine(W-1, H-1, W-1-cL, H-1); p.drawLine(W-1, H-1, W-1, H-1-cL);

    // ── 区域划分 ───────────────────────────────────────────────────────────
    // 标题行与其他 widget 对齐（sectionFont(10) 约需 22px）
    const int titleH  = 24;
    // 速度行：每行 18px，与 bodyFont(9) 行高对齐，底部留 4px padding
    const int speedH  = 18 * kChannels + 8;
    // 曲线区占剩余全部空间（减去上下各 4px 间距）
    const int chartH  = H - titleH - speedH - 10;

    QRect rcTitle(4, 2,               W - 8, titleH);
    QRect rcChart(4, titleH + 2,      W - 8, qMax(chartH, 40));
    QRect rcSpeed(4, titleH + qMax(chartH, 40) + 6, W - 8, speedH);

    drawTitle (p, rcTitle);
    drawCurves(p, rcChart);
    drawSpeeds(p, rcSpeed);
}

// ─────────────────────────────────────────────────────────────────────────────
//  drawTitle
// ─────────────────────────────────────────────────────────────────────────────
void LimbMotionWidget::drawTitle(QPainter& p, const QRect& rc) const {
    p.setFont(Theme::sectionFont(10));   // 与其他 widget 标题字号对齐
    p.setPen(Theme::Cyan);
    p.drawText(rc, Qt::AlignLeft | Qt::AlignVCenter,
               QStringLiteral("◈ 四肢关节运动曲线"));

    // 右侧图例（小色块 + 文字）
    const int dotW = 14, dotH = 6, gap = 3;
    int x = rc.right();
    for (int ch = kChannels - 1; ch >= 0; --ch) {
        QString lb = m_channels[ch].label;
        QFontMetrics fm(p.font());
        int tw = fm.horizontalAdvance(lb);
        x -= tw + dotW + gap * 2;
        if (x < rc.left()) break;

        // 色块
        p.setPen(Qt::NoPen);
        p.setBrush(m_channels[ch].color);
        int dy = rc.top() + (rc.height() - dotH) / 2;
        p.drawRoundedRect(QRect(x, dy, dotW, dotH), 2, 2);

        // 文字
        p.setPen(m_channels[ch].color);
        p.drawText(QRect(x + dotW + gap, rc.top(), tw + 2, rc.height()),
                   Qt::AlignLeft | Qt::AlignVCenter, lb);

        x -= gap;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  drawCurves
// ─────────────────────────────────────────────────────────────────────────────
void LimbMotionWidget::drawCurves(QPainter& p, const QRect& rc) const {
    if (rc.height() < 20 || rc.width() < 20) return;

    // 绘图区内边距
    const int px = 2, py = 4;
    QRect inner(rc.left() + px, rc.top() + py,
                rc.width() - 2*px, rc.height() - 2*py);

    // ── 背景网格 ───────────────────────────────────────────────────────────
    p.setPen(QPen(QColor(0x1a, 0x3a, 0x6a, 0xaa), 1, Qt::DotLine));
    int gridLines = 4;
    for (int g = 0; g <= gridLines; ++g) {
        int gy = inner.top() + (int)((float)g / gridLines * inner.height());
        p.drawLine(inner.left(), gy, inner.right(), gy);
    }
    // 零线（水平中线）
    int midY = inner.top() + inner.height() / 2;
    p.setPen(QPen(QColor(0x44, 0x66, 0x88, 0xcc), 1, Qt::SolidLine));
    p.drawLine(inner.left(), midY, inner.right(), midY);

    // ── 合并所有通道数据，确定全局 Y 范围 ──────────────────────────────────
    float globalMin = 0.f, globalMax = 0.f;
    bool  hasAny = false;
    QVector<QVector<float>> allSeries(kChannels);
    for (int ch = 0; ch < kChannels; ++ch) {
        allSeries[ch] = buildDisplacementSeries(ch);
        for (float v : allSeries[ch]) {
            if (!hasAny) { globalMin = globalMax = v; hasAny = true; }
            else { globalMin = qMin(globalMin, v); globalMax = qMax(globalMax, v); }
        }
    }
    float range = globalMax - globalMin;
    if (range < 1.f) range = 40.f;  // 无数据时用默认范围，保持网格显示

    // ─── 各通道曲线 ────────────────────────────────────────────────────────
    for (int ch = 0; ch < kChannels; ++ch) {
        const QVector<float>& series = allSeries[ch];
        if (series.size() < 2) continue;

        int nPts = series.size();
        float xStep = (float)inner.width() / (m_windowSize - 1);

        QPainterPath path;
        bool started = false;

        for (int i = 0; i < nPts; ++i) {
            // 归一化 Y：globalMin 映射到 inner.bottom()，globalMax 到 inner.top()
            float normY = (series[i] - globalMin) / range;
            float px2   = inner.left() + i * xStep;
            float py2   = inner.bottom() - normY * inner.height();

            if (!started) {
                path.moveTo(px2, py2);
                started = true;
            } else {
                path.lineTo(px2, py2);
            }
        }

        QColor col = m_channels[ch].color;

        // 发光描边（宽线，低透明）
        QPen glowPen(col, 3.0f);
        glowPen.setCapStyle(Qt::RoundCap);
        glowPen.setJoinStyle(Qt::RoundJoin);
        QColor glowCol = col; glowCol.setAlpha(50);
        p.setPen(QPen(glowCol, 5.0f, Qt::SolidLine, Qt::RoundCap));
        p.setBrush(Qt::NoBrush);
        p.drawPath(path);

        // 主线
        p.setPen(QPen(col, 1.5f, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
        p.drawPath(path);

        // 当前点圆点
        if (nPts > 0) {
            float normY = (series.last() - globalMin) / range;
            float ptX   = inner.left() + (nPts - 1) * xStep;
            float ptY   = inner.bottom() - normY * inner.height();
            p.setPen(QPen(Qt::white, 1));
            p.setBrush(col);
            p.drawEllipse(QPointF(ptX, ptY), 3.0, 3.0);
        }
    }

    // ── 边框 ───────────────────────────────────────────────────────────────
    p.setPen(QPen(QColor(0x1a, 0x3a, 0x6a), 1));
    p.setBrush(Qt::NoBrush);
    p.drawRect(inner);
}

// ─────────────────────────────────────────────────────────────────────────────
//  drawSpeeds
// ─────────────────────────────────────────────────────────────────────────────
void LimbMotionWidget::drawSpeeds(QPainter& p, const QRect& rc) const {
    // 防护：面板过小时直接跳过，避免 barH<=0 传入 drawRoundedRect 崩溃
    if (rc.height() < 8 || rc.width() < 40) return;

    const int rowH    = qMax(4, (rc.height() - 4) / kChannels);
    const int barMaxW = 70;  // 速度条最大像素宽度（稍宽，充分利用空间）

    // 找最大速度（用于归一化速度条）
    float maxSpeed = 1.f;
    for (int ch = 0; ch < kChannels; ++ch)
        if (m_vel[ch].active)
            maxSpeed = qMax(maxSpeed, m_vel[ch].speed);

    for (int ch = 0; ch < kChannels; ++ch) {
        int rowTop = rc.top() + 2 + ch * rowH;
        QColor col = m_channels[ch].color;

        // 通道标签
        p.setFont(Theme::bodyFont(9));    // 与其他 widget 正文字号对齐
        p.setPen(col);
        p.drawText(QRect(rc.left(), rowTop, 30, rowH),
                   Qt::AlignLeft | Qt::AlignVCenter,
                   m_channels[ch].label);

        int barLeft = rc.left() + 32;
        float ratio = m_vel[ch].active
                    ? std::clamp(m_vel[ch].speed / (maxSpeed * 1.1f), 0.f, 1.f)
                    : 0.f;
        int fillW = (int)(ratio * barMaxW);

        // 速度条背景
        const int barH   = qMax(2, rowH - 4);   // ⚠️ 最小 2px，防止负值传入 drawRoundedRect
        const int barTop = rowTop + (rowH - barH) / 2;
        p.setPen(Qt::NoPen);
        p.setBrush(QColor(0x1a, 0x2a, 0x4a));
        p.drawRoundedRect(QRect(barLeft, barTop, barMaxW, barH), 2, 2);

        // 速度条填充
        QColor barCol = ratio < 0.33f ? QColor(0x32, 0xcd, 0x32)
                      : ratio < 0.66f ? QColor(0xff, 0xdc, 0x1e)
                      :                 QColor(0xff, 0x50, 0x50);
        if (fillW > 0) {
            p.setBrush(barCol);
            p.drawRoundedRect(QRect(barLeft, barTop, fillW, barH), 2, 2);
        }

        // 速度数字 + 方向箭头
        if (m_vel[ch].active) {
            QString arrow;
            // 取 vx/vy 绝对值较大者决定主方向
            if (std::abs(m_vel[ch].vx) >= std::abs(m_vel[ch].vy))
                arrow = m_vel[ch].vx >= 0.f ? "→" : "←";
            else
                arrow = m_vel[ch].vy >= 0.f ? "↓" : "↑";

            QString valStr = QString("%1 px/s %2")
                                 .arg((int)m_vel[ch].speed)
                                 .arg(arrow);
            p.setPen(barCol);
            p.setFont(Theme::bodyFont(9));   // 与其他 widget 正文字号对齐
            int valLeft = barLeft + barMaxW + 4;
            int valW    = rc.right() - valLeft;
            if (valW > 10)
                p.drawText(QRect(valLeft, rowTop, valW, rowH),
                           Qt::AlignLeft | Qt::AlignVCenter, valStr);
        } else {
            // 无数据时显示灰色占位
            p.setPen(Theme::TextMuted);
            p.setFont(Theme::bodyFont(9));   // 与其他 widget 正文字号对齐
            int valLeft = barLeft + barMaxW + 4;
            int valW    = rc.right() - valLeft;
            if (valW > 10)
                p.drawText(QRect(valLeft, rowTop, valW, rowH),
                           Qt::AlignLeft | Qt::AlignVCenter,
                           QStringLiteral("-- px/s"));
        }
    }
}
