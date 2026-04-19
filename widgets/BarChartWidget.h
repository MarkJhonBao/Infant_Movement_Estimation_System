#pragma once
#include <QWidget>
#include <QVector>
#include <QString>

struct BarSeries {
    QString         name;
    QVector<double> values;
    QColor          color;
};

class BarChartWidget : public QWidget {
    Q_OBJECT
public:
    explicit BarChartWidget(QWidget* parent = nullptr);
    void setTitle(const QString& t)            { m_title = t; update(); }
    void setLabels(const QStringList& l)       { m_xLabels = l; update(); }
    void setSeries(const QVector<BarSeries>& s){ m_series = s; update(); }
protected:
    void paintEvent(QPaintEvent*) override;
private:
    QString          m_title;
    QStringList      m_xLabels;
    QVector<BarSeries> m_series;
};
