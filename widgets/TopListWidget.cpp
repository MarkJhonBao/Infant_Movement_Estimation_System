#include "TopListWidget.h"
#include "DashboardTheme.h"
#include <QPainter>

TopListWidget::TopListWidget(QWidget* parent) : QWidget(parent) {
    setMinimumSize(220, 260);
}

void TopListWidget::paintEvent(QPaintEvent*) {
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

    // Header row
    int hY = 26;
    if (!m_headers.isEmpty()) {
        p.setFont(Theme::bodyFont(8));
        p.setPen(Theme::TextMuted);
        // columns: rank(30), icon(28), name(flex), amount(55), pct(40)
        int colX[] = { 8, 38, 66, W-100, W-45 };
        for (int i = 0; i < qMin(m_headers.size(), 5); ++i)
            p.drawText(QRect(colX[i], hY, 60, 14), Qt::AlignLeft, m_headers[i]);
        hY += 16;
        p.setPen(QPen(Theme::BorderDim, 1));
        p.drawLine(8, hY, W-8, hY);
        hY += 4;
    }

    // Row height
    int available = H - hY - 4;
    int rowH = m_items.isEmpty() ? 0 : available / qMax(m_items.size(), 1);
    rowH = qMax(rowH, 18);

    static const QColor rankColors[] = {
        QColor(0xff,0xcc,0x00), QColor(0x88,0xaa,0xcc), QColor(0xcc,0x88,0x44)
    };

    for (int i = 0; i < m_items.size(); ++i) {
        const auto& item = m_items[i];
        int y = hY + i * rowH;
        if (y + rowH > H - 2) break;

        // Alternate row tint
        if (i % 2 == 0) {
            p.setPen(Qt::NoPen);
            p.setBrush(QColor(0xff,0xff,0xff, 8));
            p.drawRect(8, y, W-16, rowH-1);
        }

        // Rank badge
        QRect badge(8, y + (rowH-16)/2, 22, 16);
        QColor badgeColor = (i < 3) ? rankColors[i] : Theme::BgHighlight;
        p.setPen(Qt::NoPen);
        p.setBrush(badgeColor);
        p.drawRoundedRect(badge, 3, 3);
        p.setFont(Theme::bodyFont(7));
        p.setPen((i < 3) ? QColor(0,0,0) : Theme::TextSecondary);
        p.drawText(badge, Qt::AlignCenter, QString::number(item.rank));

        // Name
        p.setFont(Theme::bodyFont(9));
        p.setPen(Theme::TextPrimary);
        p.drawText(QRect(66, y, W-170, rowH), Qt::AlignLeft | Qt::AlignVCenter, item.name);

        // Amount
        p.setFont(Theme::sectionFont(9));
        p.setPen(Theme::BlueLight);
        p.drawText(QRect(W-104, y, 55, rowH), Qt::AlignRight | Qt::AlignVCenter,
                   QString::number(item.amount));

        // Percent
        p.setFont(Theme::bodyFont(9));
        p.setPen(Theme::TextSecondary);
        p.drawText(QRect(W-45, y, 38, rowH), Qt::AlignRight | Qt::AlignVCenter,
                   item.percent);
    }
}
