#include "BarChartWidget.h"
#include "DashboardTheme.h"
#include <QPainter>
#include <algorithm>

BarChartWidget::BarChartWidget(QWidget* parent) : QWidget(parent) {
    setMinimumSize(240, 160);
}

void BarChartWidget::paintEvent(QPaintEvent*) {
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing);
    int W = width(), H = height();

    p.fillRect(rect(), Theme::BgPanel);
    p.setPen(QPen(Theme::BorderDim, 1));
    p.drawRect(rect().adjusted(0,0,-1,-1));

    // Title
    p.setFont(Theme::sectionFont(10));
    p.setPen(Theme::Cyan);
    p.drawText(QRect(8, 4, W-16, 20), Qt::AlignLeft | Qt::AlignVCenter, m_title);

    if (m_series.isEmpty() || m_xLabels.isEmpty()) return;

    // Margins
    int left = 36, right = 8, top = 28, bottom = 24;
    int chartW = W - left - right;
    int chartH = H - top - bottom;

    // Find max value
    double maxVal = 1.0;
    for (auto& s : m_series)
        for (auto v : s.values)
            maxVal = qMax(maxVal, v);

    // Grid lines
    int gridCount = 4;
    p.setPen(QPen(Theme::BorderDim, 1, Qt::DotLine));
    p.setFont(Theme::bodyFont(7));
    p.setPen(Theme::TextMuted);
    for (int i = 0; i <= gridCount; ++i) {
        int y = top + chartH - static_cast<int>(i * chartH / gridCount);
        p.setPen(QPen(Theme::BorderDim, 1, Qt::DotLine));
        p.drawLine(left, y, left + chartW, y);
        p.setPen(Theme::TextMuted);
        p.setFont(Theme::bodyFont(7));
        double label = i * maxVal / gridCount;
        p.drawText(QRect(0, y-8, left-2, 16), Qt::AlignRight | Qt::AlignVCenter,
                   QString::number(static_cast<int>(label)));
    }

    // Legend
    int legendX = left;
    for (auto& s : m_series) {
        p.setPen(Qt::NoPen);
        p.setBrush(s.color);
        p.drawRect(legendX, top-14, 10, 8);
        p.setFont(Theme::bodyFont(7));
        p.setPen(Theme::TextSecondary);
        p.drawText(legendX+12, top-7, s.name);
        legendX += p.fontMetrics().horizontalAdvance(s.name) + 24;
    }

    // Bars
    int numX = m_xLabels.size();
    int groupW = chartW / numX;
    int barW = qMax(2, groupW / (m_series.size() + 1));

    for (int xi = 0; xi < numX; ++xi) {
        int groupLeft = left + xi * groupW + groupW/2
                        - (m_series.size() * barW + (m_series.size()-1)*2) / 2;

        for (int si = 0; si < m_series.size(); ++si) {
            if (xi >= m_series[si].values.size()) continue;
            double val = m_series[si].values[xi];
            int barH = static_cast<int>(val / maxVal * chartH);
            int bx   = groupLeft + si * (barW + 2);
            int by   = top + chartH - barH;

            // Gradient bar
            QLinearGradient grad(bx, by, bx, by+barH);
            grad.setColorAt(0, m_series[si].color.lighter(140));
            grad.setColorAt(1, m_series[si].color.darker(160));
            p.setPen(Qt::NoPen);
            p.setBrush(grad);
            p.drawRect(bx, by, barW, barH);
        }

        // X label
        p.setFont(Theme::bodyFont(7));
        p.setPen(Theme::TextSecondary);
        p.drawText(QRect(left + xi*groupW, top+chartH+2, groupW, 18),
                   Qt::AlignCenter, m_xLabels[xi]);
    }

    // X axis line
    p.setPen(QPen(Theme::BorderDim, 1));
    p.drawLine(left, top+chartH, left+chartW, top+chartH);
}
