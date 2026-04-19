#include "PieChartWidget.h"
#include "DashboardTheme.h"
#include <QPainter>
#include <QPainterPath>
#include <cmath>

PieChartWidget::PieChartWidget(QWidget* parent) : QWidget(parent) {
    setMinimumSize(240, 200);
}

void PieChartWidget::paintEvent(QPaintEvent*) {
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing);
    int W = width(), H = height();

    // Panel background
    p.fillRect(rect(), Theme::BgPanel);
    p.setPen(QPen(Theme::BorderDim, 1));
    p.drawRect(rect().adjusted(0,0,-1,-1));

    // Section title
    p.setFont(Theme::sectionFont(10));
    p.setPen(Theme::Cyan);
    p.drawText(QRect(8, 4, W-16, 20), Qt::AlignLeft | Qt::AlignVCenter, m_title);

    if (m_slices.isEmpty()) return;

    double total = 0;
    for (auto& s : m_slices) total += s.value;
    if (total <= 0) return;

    // Pie area
    int cx = W/2 - 10, cy = H/2 + 10;
    int r  = qMin(W, H) / 2 - 30;
    QRectF pie(cx-r, cy-r, 2*r, 2*r);

    // Draw ring segments
    double angle = -90.0;
    for (int i = 0; i < m_slices.size(); ++i) {
        double span = m_slices[i].value / total * 360.0;
        QColor c = Theme::ChartColors[i % Theme::ChartColorCount];

        // outer ring
        p.setPen(Qt::NoPen);
        p.setBrush(c);
        p.drawPie(pie, static_cast<int>(angle*16), static_cast<int>(span*16));

        // label line + text
        double midAngle = (angle + span/2) * M_PI / 180.0;
        double lx = cx + (r + 12) * std::cos(midAngle);
        double ly = cy + (r + 12) * std::sin(midAngle);
        double tx = cx + (r + 28) * std::cos(midAngle);
        double ty = cy + (r + 28) * std::sin(midAngle);

        // 标签对齐方向：左右两侧分别使用不同的对齐方式
        int textAlign = Qt::AlignVCenter;
        double textOffset = 35;
        if (midAngle > M_PI/2 && midAngle < 3*M_PI/2) {
            textAlign |= Qt::AlignRight;
            tx -= textOffset;
        } else {
            textAlign |= Qt::AlignLeft;
        }

        p.setPen(QPen(c, 1));
        p.drawLine(QPointF(lx,ly), QPointF(tx,ty));

        p.setFont(Theme::bodyFont(8));
        p.setPen(c);
        
        // 自动计算标签所需宽度，不截断文字
        QFontMetrics fm(p.font());
        QString labelText = QString("%1  %2").arg(m_slices[i].label).arg(m_slices[i].value);
        int textWidth = fm.horizontalAdvance(labelText) + 8;
        int textHeight = fm.height();

        QRectF tr;
        if (textAlign & Qt::AlignLeft) {
            tr = QRectF(tx + 4, ty - textHeight/2, textWidth, textHeight);
        } else {
            tr = QRectF(tx - textWidth - 4, ty - textHeight/2, textWidth, textHeight);
        }
        
        // 确保标签不会超出控件边界
        tr.adjust(0, 0, 0, 0);
        if (tr.left() < 4) tr.moveLeft(4);
        if (tr.right() > W - 4) tr.moveRight(W - 4);
        if (tr.top() < 22) tr.moveTop(22);
        if (tr.bottom() > H - 4) tr.moveBottom(H - 4);

        p.drawText(tr, textAlign, labelText);

        angle += span;
    }

    // White centre hole
    p.setPen(Qt::NoPen);
    p.setBrush(Theme::BgPanel);
    p.drawEllipse(QPointF(cx, cy), r*0.45, r*0.45);

    // Inner cyan ring
    p.setPen(QPen(Theme::CyanDim, 1));
    p.setBrush(Qt::NoBrush);
    p.drawEllipse(QPointF(cx, cy), r*0.45, r*0.45);
}
