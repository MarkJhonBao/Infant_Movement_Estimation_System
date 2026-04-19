#include "TitleBarWidget.h"
#include "DashboardTheme.h"
#include <QPainter>
#include <QPainterPath>

TitleBarWidget::TitleBarWidget(QWidget* parent) : QWidget(parent) {
    setFixedHeight(60);
    m_now = QDateTime::currentDateTime();
    connect(&m_timer, &QTimer::timeout, this, &TitleBarWidget::tick);
    m_timer.start(1000);
}

void TitleBarWidget::tick() {
    m_now = QDateTime::currentDateTime();
    update();
}

void TitleBarWidget::paintEvent(QPaintEvent*) {
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing);
    int W = width(), H = height();

    // Gradient background
    QLinearGradient bg(0,0,W,0);
    bg.setColorAt(0,   Theme::BgDeep);
    bg.setColorAt(0.5, QColor(0x08,0x28,0x60));
    bg.setColorAt(1,   Theme::BgDeep);
    p.fillRect(rect(), bg);

    // Bottom border glow
    QLinearGradient border(0, H-1, W, H-1);
    border.setColorAt(0,   Qt::transparent);
    border.setColorAt(0.3, Theme::Cyan);
    border.setColorAt(0.7, Theme::Cyan);
    border.setColorAt(1,   Qt::transparent);
    p.setPen(QPen(QBrush(border), 1.5));
    p.drawLine(0, H-1, W, H-1);

    // Decorative corner marks
    auto corner = [&](int x, int y, int dx, int dy){
        p.setPen(QPen(Theme::Cyan, 2));
        p.drawLine(x, y, x+dx*12, y);
        p.drawLine(x, y, x, y+dy*12);
    };
    corner(8, 8,  1,  1);
    corner(W-8, 8, -1,  1);

    // Title text
    p.setFont(Theme::titleFont(20));
    p.setPen(Theme::Cyan);
    p.drawText(rect(), Qt::AlignCenter,
               QStringLiteral("早产儿卧床肢体姿态监控与护理监控大屏"));

    // Date / time (top-right)
    QString dateStr = m_now.toString("yyyy-MM-dd");
    QString timeStr = m_now.toString("HH:mm:ss");
    QRect right(W-170, 0, 160, H);

    p.setFont(Theme::bodyFont(10));
    p.setPen(Theme::TextSecondary);
    p.drawText(right, Qt::AlignRight | Qt::AlignTop | Qt::TextWordWrap,
               dateStr + "\n" + timeStr);
}
