#pragma once
#include <QWidget>
#include <QStringList>

class AlertWidget : public QWidget {
    Q_OBJECT
public:
    explicit AlertWidget(QWidget* parent = nullptr);
    void setTitle(const QString& t) { m_title = t; update(); }
    void setAlerts(const QStringList& a) { m_alerts = a; update(); }
protected:
    void paintEvent(QPaintEvent*) override;
private:
    QString     m_title;
    QStringList m_alerts;
};
