#pragma once
#include <QWidget>
#include <QVector>

struct KpiItem { QString label; QString value; QString unit; };

class KpiWidget : public QWidget {
    Q_OBJECT
public:
    explicit KpiWidget(QWidget* parent = nullptr);
    void setItems(const QVector<KpiItem>& items) { m_items = items; update(); }
protected:
    void paintEvent(QPaintEvent*) override;
private:
    QVector<KpiItem> m_items;
};
