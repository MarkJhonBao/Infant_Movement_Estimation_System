#include "AlertWidget.h"
#include "DashboardTheme.h"
#include <QPainter>

static const QColor kAlertColors[] = {
    QColor(0x00,0xcc,0xff,0xcc),  // cyan-blue
    QColor(0x00,0x99,0xff,0xcc),  // blue
    QColor(0xff,0x88,0x00,0xcc),  // orange
};

AlertWidget::AlertWidget(QWidget* parent) : QWidget(parent) {
    setMinimumHeight(120);
}

void AlertWidget::paintEvent(QPaintEvent*) {
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing);
    int W = width(), H = height();

    p.fillRect(rect(), Theme::BgPanel);
    p.setPen(QPen(Theme::BorderDim, 1));
    p.drawRect(rect().adjusted(0,0,-1,-1));

    p.setFont(Theme::sectionFont(10));
    p.setPen(Theme::Cyan);
    p.drawText(QRect(8, 4, W-16, 20), Qt::AlignLeft | Qt::AlignVCenter, m_title);

    if (m_alerts.isEmpty()) return;

    int cols    = m_alerts.size();
    int boxW    = (W - 16 - (cols-1)*6) / cols;
    int boxH    = H - 36;
    int startX  = 8;
    int startY  = 28;

    for (int i = 0; i < cols; ++i) {
        QRect box(startX + i*(boxW+6), startY, boxW, boxH);
        QColor c = kAlertColors[i % 3];

        // Background fill
        p.setPen(Qt::NoPen);
        p.setBrush(c);
        p.drawRoundedRect(box, 4, 4);

        // Text
        p.setPen(Theme::TextPrimary);
        p.setFont(Theme::sectionFont(9));
        p.drawText(box, Qt::AlignCenter | Qt::TextWordWrap, m_alerts[i]);
    }
}
