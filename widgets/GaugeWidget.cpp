#include "GaugeWidget.h"
#include "DashboardTheme.h"
#include <QPainter>
#include <QPainterPath>
#include <cmath>

GaugeWidget::GaugeWidget(QWidget* parent)
    : QWidget(parent), m_value(0.3), m_label("指标"), m_color(Theme::Blue)
{
    setMinimumSize(100, 130);
}

void GaugeWidget::setValue(double v) {
    m_value = qBound(0.0, v, 1.0);
    update();
}

void GaugeWidget::paintEvent(QPaintEvent*) {
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing);
    int W = width(), H = height();
    p.fillRect(rect(), Qt::transparent);

    int r  = qMin(W, H-30) / 2 - 8;
    int cx = W/2;
    int cy = (H-24)/2 + 4;

    // Outer circle
    p.setPen(QPen(m_color.darker(150), 2));
    p.setBrush(Qt::NoBrush);
    p.drawEllipse(QPointF(cx,cy), r, r);

    // Liquid fill clip path (circle)
    QPainterPath circlePath;
    circlePath.addEllipse(QPointF(cx,cy), r-1, r-1);
    p.setClipPath(circlePath);

    // Fill level
    double fillH = (2*r) * m_value;
    double fillY = cy + r - fillH;

    // Wave effect
    QLinearGradient fillGrad(cx, fillY, cx, cy+r);
    fillGrad.setColorAt(0, m_color.lighter(140));
    fillGrad.setColorAt(1, m_color.darker(140));
    p.setPen(Qt::NoPen);
    p.setBrush(fillGrad);

    // simple wave rectangle
    QPainterPath wave;
    double waveAmp = 4.0;
    wave.moveTo(cx-r-2, fillY + waveAmp);
    for (int x = cx-r-2; x <= cx+r+2; x += 4) {
        double angle = (x - (cx-r)) * M_PI / (r * 0.8);
        double y = fillY + waveAmp * std::sin(angle);
        wave.lineTo(x, y);
    }
    wave.lineTo(cx+r+2, cy+r+2);
    wave.lineTo(cx-r-2, cy+r+2);
    wave.closeSubpath();
    p.drawPath(wave);

    p.setClipping(false);

    // Outer glow ring
    p.setPen(QPen(m_color, 2));
    p.setBrush(Qt::NoBrush);
    p.drawEllipse(QPointF(cx,cy), r, r);

    // Percentage text
    p.setFont(Theme::numFont(18));
    p.setPen(Theme::TextPrimary);
    p.drawText(QRect(cx-r, cy-16, 2*r, 32), Qt::AlignCenter,
               QString("%1%").arg(static_cast<int>(m_value*100)));

    // Label below
    p.setFont(Theme::bodyFont(9));
    p.setPen(Theme::TextSecondary);
    p.drawText(QRect(0, H-24, W, 20), Qt::AlignCenter, m_label);
}
