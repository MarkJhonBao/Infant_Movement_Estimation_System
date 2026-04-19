#pragma once
#include <QWidget>
#include <QVector>
#include <QString>

struct DonutSlice { QString label; double value; };

class DonutChartWidget : public QWidget {
    Q_OBJECT
public:
    explicit DonutChartWidget(QWidget* parent = nullptr);
    void setTitle(const QString& t)             { m_title = t; update(); }
    void setSlices(const QVector<DonutSlice>& s){ m_slices = s; update(); }
protected:
    void paintEvent(QPaintEvent*) override;
private:
    QString             m_title;
    QVector<DonutSlice> m_slices;
};
