#pragma once
#include <QWidget>
#include <QTimer>
#include <QDateTime>

class TitleBarWidget : public QWidget {
    Q_OBJECT
public:
    explicit TitleBarWidget(QWidget* parent = nullptr);
protected:
    void paintEvent(QPaintEvent*) override;
private slots:
    void tick();
private:
    QTimer    m_timer;
    QDateTime m_now;
};
