#pragma once
#include <QWidget>

class GaugeWidget : public QWidget {
    Q_OBJECT
    Q_PROPERTY(double value READ value WRITE setValue)
public:
    explicit GaugeWidget(QWidget* parent = nullptr);
    double value() const { return m_value; }
    void setValue(double v);
    void setLabel(const QString& l) { m_label = l; update(); }
    void setColor(const QColor& c)  { m_color = c; update(); }
protected:
    void paintEvent(QPaintEvent*) override;
private:
    double  m_value{0.0};
    QString m_label;
    QColor  m_color;
};
