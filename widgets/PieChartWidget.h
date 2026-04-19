#pragma once
#include <QWidget>
#include <QVector>
#include <QString>

struct PieSlice { QString label; double value; };

class PieChartWidget : public QWidget {
    Q_OBJECT
public:
    explicit PieChartWidget(QWidget* parent = nullptr);
    void setTitle(const QString& t) { m_title = t; update(); }
    void setSlices(const QVector<PieSlice>& s) { m_slices = s; update(); }
protected:
    void paintEvent(QPaintEvent*) override;
private:
    QString            m_title;
    QVector<PieSlice>  m_slices;
};
