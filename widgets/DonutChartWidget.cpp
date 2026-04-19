#include "DonutChartWidget.h"
#include "DashboardTheme.h"
#include <QPainter>
#include <cmath>

DonutChartWidget::DonutChartWidget(QWidget* parent) : QWidget(parent) {
    setMinimumSize(200, 160);
}

void DonutChartWidget::paintEvent(QPaintEvent*) {
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing);
    int W = width(), H = height();

    p.fillRect(rect(), Theme::BgPanel);
    p.setPen(QPen(Theme::BorderDim, 1));
    p.drawRect(rect().adjusted(0,0,-1,-1));

    p.setFont(Theme::sectionFont(10));
    p.setPen(Theme::Cyan);
    p.drawText(QRect(8, 4, W-16, 20), Qt::AlignLeft | Qt::AlignVCenter, m_title);

    if (m_slices.isEmpty()) return;

    double total = 0;
    for (auto& s : m_slices) total += s.value;
    if (total <= 0) return;

    // Legend area on right
    int legendW = 90;
    int chartAreaW = W - legendW - 8;

    int r  = qMin(chartAreaW, H-30) / 2 - 8;
    int cx = 8 + chartAreaW/2;
    int cy = 28 + (H-30)/2;

    // Draw concentric rings (multiple donut rings for each category)
    int ringCount = m_slices.size();
    int ringGap   = 4;
    int ringThick = qMax(6, (r - 8) / qMax(ringCount,1) - ringGap);

    for (int i = 0; i < ringCount; ++i) {
        int ri = r - i*(ringThick + ringGap);
        if (ri <= 0) break;
        QRectF outerRect(cx-ri, cy-ri, 2*ri, 2*ri);
        QRectF innerRect(cx-(ri-ringThick), cy-(ri-ringThick),
                         2*(ri-ringThick), 2*(ri-ringThick));

        double frac  = m_slices[i].value / total;
        double span  = frac * 360.0;
        QColor c     = Theme::ChartColors[i % Theme::ChartColorCount];

        // Background ring (dim)
        p.setPen(Qt::NoPen);
        p.setBrush(c.darker(300));
        p.drawEllipse(outerRect);
        p.setBrush(Theme::BgPanel);
        p.drawEllipse(innerRect);

        // Foreground arc
        p.setPen(QPen(c, ringThick, Qt::SolidLine, Qt::FlatCap));
        p.setBrush(Qt::NoBrush);
        p.drawArc(outerRect.adjusted(ringThick/2, ringThick/2,
                                     -ringThick/2, -ringThick/2),
                  static_cast<int>(-90*16),
                  static_cast<int>(-span*16));
    }

    // Legend
    int legX = W - legendW;
    int legY = 28;
    for (int i = 0; i < m_slices.size(); ++i) {
        QColor c = Theme::ChartColors[i % Theme::ChartColorCount];
        p.setPen(Qt::NoPen);
        p.setBrush(c);
        p.drawRect(legX, legY + i*18 + 4, 8, 8);
        p.setFont(Theme::bodyFont(8));
        p.setPen(Theme::TextSecondary);
        p.drawText(QRect(legX+12, legY+i*18, legendW-14, 18),
                   Qt::AlignLeft | Qt::AlignVCenter,
                   QString("• %1").arg(m_slices[i].label));
    }
}
