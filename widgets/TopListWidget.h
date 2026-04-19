#pragma once
#include <QWidget>
#include <QVector>
#include <QString>

struct TopListItem {
    int     rank;
    QString name;
    int     amount;
    QString percent;
};

class TopListWidget : public QWidget {
    Q_OBJECT
public:
    explicit TopListWidget(QWidget* parent = nullptr);
    void setTitle(const QString& t)               { m_title = t; update(); }
    void setItems(const QVector<TopListItem>& v)  { m_items = v; update(); }
    void setColumnHeaders(const QStringList& h)   { m_headers = h; update(); }
protected:
    void paintEvent(QPaintEvent*) override;
private:
    QString              m_title;
    QVector<TopListItem> m_items;
    QStringList          m_headers;
};
