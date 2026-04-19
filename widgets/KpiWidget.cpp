#include "KpiWidget.h"
#include "DashboardTheme.h"
#include <QPainter>

KpiWidget::KpiWidget(QWidget* parent) : QWidget(parent) {
    setMinimumHeight(120);
}

void KpiWidget::paintEvent(QPaintEvent*) {
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing);
    int W = width(), H = height();

    p.fillRect(rect(), Theme::BgPanel);
    p.setPen(QPen(Theme::BorderDim, 1));
    p.drawRect(rect().adjusted(0,0,-1,-1));

    if (m_items.isEmpty()) return;

    int cols = qMin(m_items.size(), 3);
    int rows = (m_items.size() + cols - 1) / cols;
    int cellW = (W - 8) / cols;
    int cellH = (H - 8) / rows;

    for (int i = 0; i < m_items.size(); ++i) {
        int col = i % cols;
        int row = i / cols;
        QRect cell(4 + col*cellW, 4 + row*cellH, cellW, cellH);

        // Dividers
        if (col > 0) {
            p.setPen(QPen(Theme::BorderDim, 1));
            p.drawLine(cell.left(), cell.top()+8, cell.left(), cell.bottom()-8);
        }

        // Label
        p.setFont(Theme::bodyFont(9));
        p.setPen(Theme::TextSecondary);
        p.drawText(QRect(cell.x(), cell.y()+4, cellW, 18),
                   Qt::AlignCenter, m_items[i].label);

        // Value
        p.setFont(Theme::numFont(22));
        p.setPen(Theme::Cyan);
        QString valStr = m_items[i].value;
        p.drawText(QRect(cell.x(), cell.y()+20, cellW, cellH-32),
                   Qt::AlignCenter, valStr);

        // Unit
        p.setFont(Theme::bodyFont(9));
        p.setPen(Theme::TextSecondary);
        p.drawText(QRect(cell.x(), cell.bottom()-18, cellW, 16),
                   Qt::AlignCenter, m_items[i].unit);
    }
}
