#pragma once
#include <QWidget>
#include <QGridLayout>
#include <QVector>
#include "KpiWidget.h"

class TitleBarWidget;
class PieChartWidget;
class GaugeWidget;
class AlertWidget;
class DetectionDisplayWidget;
class KpiWidget;
class TopListWidget;
class BarChartWidget;
class DonutChartWidget;
class LimbMotionWidget;

// ─────────────────────────────────────────────────────────────────────────────
//  DashboardWidget
//  早产儿卧床肢体姿态监控与护理系统 — 根布局组件
//  负责将所有子面板排布为参考设计的大屏布局
// ─────────────────────────────────────────────────────────────────────────────
class DashboardWidget : public QWidget {
    Q_OBJECT
public:
    explicit DashboardWidget(QWidget* parent = nullptr);

    // 供 MainWindow 接入信号
    DetectionDisplayWidget* detectionPanel() const { return m_detPanel; }
    LimbMotionWidget*       limbMotionPanel() const { return m_limbMotion; }
    
    // 更新KPI数据
    void updateKpiData(const QVector<KpiItem>& todayItems, const QVector<KpiItem>& monthItems);

    // ── 新增：动态刷新接口 ─────────────────────────────────────────────────
    void updateGaugeData(double anomalyRate, double nursingRate);
    void updateAlerts(const QStringList& alerts);

private:
    void buildLayout();
    void populateData();

    TitleBarWidget*         m_titleBar{nullptr};
    PieChartWidget*         m_pieChart{nullptr};
    GaugeWidget*            m_gauge1{nullptr};
    GaugeWidget*            m_gauge2{nullptr};
    AlertWidget*            m_alertWidget{nullptr};
    DetectionDisplayWidget* m_detPanel{nullptr};
    KpiWidget*              m_kpiToday{nullptr};   // 今日统计
    KpiWidget*              m_kpiMonth{nullptr};   // 本月统计
    TopListWidget*          m_topAnomalies{nullptr}; // 异常姿态 TOP 榜
    TopListWidget*          m_topBeds{nullptr};      // 床位风险 TOP 榜
    BarChartWidget*         m_barChart{nullptr};
    DonutChartWidget*       m_donutChart{nullptr};
    LimbMotionWidget*       m_limbMotion{nullptr};  // 左下角四肢运动曲线面板
};