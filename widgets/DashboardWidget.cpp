#include "DashboardWidget.h"
#include "DashboardTheme.h"
#include "TitleBarWidget.h"
#include "PieChartWidget.h"
#include "GaugeWidget.h"
#include "AlertWidget.h"
#include "DetectionDisplayWidget.h"
#include "KpiWidget.h"
#include "TopListWidget.h"
#include "BarChartWidget.h"
#include "DonutChartWidget.h"
#include "LimbMotionWidget.h"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QPalette>

// ─── Helper: 带边框的面板容器 ─────────────────────────────────────────────────
static QWidget* makePanelOf(QWidget* child) {
    auto* wrap = new QWidget;
    wrap->setAutoFillBackground(false);
    auto* l = new QVBoxLayout(wrap);
    l->setContentsMargins(0, 0, 0, 0);
    l->addWidget(child);
    return wrap;
}

// ─────────────────────────────────────────────────────────────────────────────
DashboardWidget::DashboardWidget(QWidget* parent) : QWidget(parent) {
    setAutoFillBackground(true);
    QPalette pal = palette();
    pal.setColor(QPalette::Window, QColor(0x04, 0x0d, 0x2c));
    setPalette(pal);
    buildLayout();
    populateData();
}

void DashboardWidget::buildLayout() {
    // ── 顶部标题栏（全宽）────────────────────────────────────────────────────
    m_titleBar = new TitleBarWidget;

    // ── 左侧列 ────────────────────────────────────────────────────────────────
    // 饼图：肢体姿态分类统计
    m_pieChart = new PieChartWidget;
    m_pieChart->setMinimumHeight(220);

    // 仪表盘行：异常姿态占比 + 护理完成率
    m_gauge1 = new GaugeWidget;
    m_gauge2 = new GaugeWidget;
    auto* gaugeRow    = new QWidget;
    auto* gaugeLayout = new QHBoxLayout(gaugeRow);
    gaugeLayout->setContentsMargins(0, 0, 0, 0);
    gaugeLayout->addWidget(m_gauge1);
    gaugeLayout->addWidget(m_gauge2);
    gaugeRow->setMinimumHeight(140);

    // 告警区：姿态异常预警
    m_alertWidget = new AlertWidget;
    m_alertWidget->setMinimumHeight(110);

    // 四肢运动曲线 + 速度面板（左下角红色区域）
    m_limbMotion = new LimbMotionWidget;
    m_limbMotion->setMinimumHeight(200);

    auto* leftCol    = new QWidget;
    auto* leftLayout = new QVBoxLayout(leftCol);
    leftLayout->setContentsMargins(4, 4, 4, 4);
    leftLayout->setSpacing(6);
    leftLayout->addWidget(m_pieChart);
    leftLayout->addWidget(gaugeRow);
    leftLayout->addWidget(m_alertWidget);
    leftLayout->addWidget(m_limbMotion, 1);   // 弹性填满剩余空间

    // ── 中心列 ────────────────────────────────────────────────────────────────
    // 主检测视频面板
    m_detPanel = new DetectionDisplayWidget;

    // 底部中心：甜甜圈图 + 柱状图
    m_donutChart = new DonutChartWidget;
    m_barChart   = new BarChartWidget;
    m_barChart->setTitle(QStringLiteral("各床位本月日均异常姿态次数对比:"));

    auto* bottomCentre = new QWidget;
    auto* bcLayout     = new QHBoxLayout(bottomCentre);
    bcLayout->setContentsMargins(0, 0, 0, 0);
    bcLayout->setSpacing(6);
    bcLayout->addWidget(m_donutChart, 4);
    bcLayout->addWidget(m_barChart,   6);
    bottomCentre->setFixedHeight(170);

    auto* centreCol    = new QWidget;
    auto* centreLayout = new QVBoxLayout(centreCol);
    centreLayout->setContentsMargins(4, 4, 4, 4);
    centreLayout->setSpacing(6);
    centreLayout->addWidget(m_detPanel,     1);
    centreLayout->addWidget(bottomCentre,   0);

    // ── 右侧列 ────────────────────────────────────────────────────────────────
    m_kpiToday = new KpiWidget;
    m_kpiToday->setFixedHeight(100);
    m_kpiMonth = new KpiWidget;
    m_kpiMonth->setFixedHeight(80);
    m_topAnomalies = new TopListWidget;
    m_topBeds      = new TopListWidget;

    auto* rightCol    = new QWidget;
    auto* rightLayout = new QVBoxLayout(rightCol);
    rightLayout->setContentsMargins(4, 4, 4, 4);
    rightLayout->setSpacing(6);
    rightLayout->addWidget(m_kpiToday);
    rightLayout->addWidget(m_kpiMonth);
    rightLayout->addWidget(m_topAnomalies, 1);
    rightLayout->addWidget(m_topBeds,      1);

    // ── 主体行 ────────────────────────────────────────────────────────────────
    auto* body       = new QWidget;
    auto* bodyLayout = new QHBoxLayout(body);
    bodyLayout->setContentsMargins(0, 0, 0, 0);
    bodyLayout->setSpacing(6);
    bodyLayout->addWidget(leftCol,   22);
    bodyLayout->addWidget(centreCol, 48);
    bodyLayout->addWidget(rightCol,  28);

    // ── 根布局 ────────────────────────────────────────────────────────────────
    auto* root = new QVBoxLayout(this);
    root->setContentsMargins(6, 6, 6, 6);
    root->setSpacing(4);
    root->addWidget(m_titleBar, 0);
    root->addWidget(body,        1);
}

void DashboardWidget::populateData() {
    // ── 饼图：肢体姿态分类检测统计 ────────────────────────────────────────────
    // 统计各部位在实时帧中被检出的姿态类别分布
    m_pieChart->setTitle(QStringLiteral("肢体姿态分类检测统计:"));
    m_pieChart->setSlices({
        {"上肢正常", 0.26},   // 左右上肢姿态正常合计
        {"下肢正常", 0.24},   // 左右下肢姿态正常合计
        {"头部正常", 0.18},   // 头部位置正常
        {"躯干正常", 0.16},   // 躯干姿态正常
        {"异常姿态", 0.11},   // 检出异常（任意部位）
        {"待判定",   0.05},   // 置信度不足，等待复判
    });

    // ── 仪表盘：异常姿态占比 & 护理响应完成率 ────────────────────────────────
    m_gauge1->setValue(0.11);                            // 异常检出率 11%
    m_gauge1->setLabel(QStringLiteral("异常姿态占比"));
    m_gauge1->setColor(Theme::Orange);                   // 橙色表示需关注

    m_gauge2->setValue(0.94);                            // 护理完成率 94%
    m_gauge2->setLabel(QStringLiteral("护理响应完成率"));
    m_gauge2->setColor(Theme::Cyan);                     // 青色表示良好

    // ── 告警区：实时姿态异常预警 ──────────────────────────────────────────────
    m_alertWidget->setTitle(QStringLiteral("姿态异常实时预警:"));
    m_alertWidget->setAlerts({
        QStringLiteral("B03 床 — 左上肢长期受压，持续 >15 min"),
        QStringLiteral("A07 床 — 右下肢异常弯曲，角度超出阈值"),
        QStringLiteral("C12 床 — 头部持续偏转，需人工复核"),
    });

    // ── KPI：今日实时统计 ─────────────────────────────────────────────────────
    m_kpiToday->setItems({
        {"今日监测帧数", "18,432", "帧"},
        {"异常姿态检出",   "  203",  "次"},
        {"护理响应处置",   "  196",  "次"},
    });

    // ── KPI：本月汇总统计 ─────────────────────────────────────────────────────
    m_kpiMonth->setItems({
        {"本月累计监测", "426,880", "帧"},
        {"本月异常检出",   " 4,217", "次"},
        {"本月护理完成率",    "97.2", "%"},
    });

    // ── TOP 榜：异常姿态类型分布（本月）──────────────────────────────────────
    m_topAnomalies->setTitle(QStringLiteral("异常姿态类型检出 TOP8（本月）:"));
    m_topAnomalies->setColumnHeaders({"排名", "", "异常姿态类型", "检出次数", "占比"});
    m_topAnomalies->setItems({
        {1, "上肢受压",    1043, "24.7%"},
        {2, "下肢过度弯曲", 876, "20.8%"},
        {3, "头部偏转异常", 654, "15.5%"},
        {4, "躯干侧翻",    521, "12.4%"},
        {5, "上肢过度伸展", 398, " 9.4%"},
        {6, "下肢交叉",    312, " 7.4%"},
        {7, "颈部前倾",    245, " 5.8%"},
        {8, "多部位联合异常",168, " 4.0%"},
    });

    // ── TOP 榜：各床位异常姿态次数排行 ───────────────────────────────────────
    m_topBeds->setTitle(QStringLiteral("床位异常姿态风险排行 TOP5（本月）:"));
    m_topBeds->setColumnHeaders({"排名", "", "床位编号", "异常次数", "风险等级"});
    m_topBeds->setItems({
        {1, "B03 床", 412, "高风险"},
        {2, "A07 床", 387, "高风险"},
        {3, "C12 床", 298, "中风险"},
        {4, "A02 床", 201, "中风险"},
        {5, "D05 床", 134, "低风险"},
    });

    // ── 甜甜圈图：肢体部位检测帧占比 ────────────────────────────────────────
    m_donutChart->setTitle(QStringLiteral("肢体部位检测帧占比（本月）:"));
    m_donutChart->setSlices({
        {"左上肢", 0.20},
        {"右上肢", 0.20},
        {"左下肢", 0.18},
        {"右下肢", 0.18},
        {"头部",   0.14},
        {"躯干",   0.10},
    });

    // ── 柱状图：各床位本月日均异常姿态次数对比 ───────────────────────────────
    // X 轴为月份（1–12），Y 轴为次数，按上肢/下肢/头部分系列
    QStringList months;
    for (int i = 1; i <= 12; ++i) months << QString::number(i) + QStringLiteral("月");
    m_barChart->setLabels(months);
    m_barChart->setSeries({
        {
            "上肢异常",
            {32, 28, 35, 40, 38, 30, 27, 33, 41, 36, 29, 34},
            Theme::Cyan
        },
        {
            "下肢异常",
            {24, 21, 26, 30, 28, 22, 20, 25, 31, 27, 21, 25},
            Theme::Orange
        },
        {
            "头部/躯干异常",
            {15, 13, 17, 19, 18, 14, 12, 16, 20, 17, 13, 16},
            Theme::Green
        },
    });

    // 检测面板默认推理后端名称
    m_detPanel->setBackendName(QStringLiteral("ONNX Runtime"));
}

void DashboardWidget::updateKpiData(const QVector<KpiItem>& todayItems, const QVector<KpiItem>& monthItems) {
    m_kpiToday->setItems(todayItems);
    m_kpiMonth->setItems(monthItems);
}

// ── 新增：动态刷新仪表盘数值 ──────────────────────────────────────────────────
void DashboardWidget::updateGaugeData(double anomalyRate, double nursingRate) {
    if (m_gauge1) {
        m_gauge1->setValue(anomalyRate);
        // 颜色随数值变化：高异常率显示红色，低异常率显示橙色
        m_gauge1->setColor(anomalyRate > 0.3 ? QColor(0xff, 0x40, 0x40) : QColor(0xff, 0x6e, 0x1e));
    }
    if (m_gauge2) {
        m_gauge2->setValue(nursingRate);
        // 护理完成率：高则青色，中则黄色，低则红色
        QColor nurseColor = nursingRate > 0.85 ? QColor(0x00, 0xe5, 0xff)
                          : nursingRate > 0.60 ? QColor(0xff, 0xe0, 0x40)
                          :                      QColor(0xff, 0x40, 0x40);
        m_gauge2->setColor(nurseColor);
    }
}

// ── 新增：动态刷新告警栏 ──────────────────────────────────────────────────────
void DashboardWidget::updateAlerts(const QStringList& alerts) {
    if (m_alertWidget)
        m_alertWidget->setAlerts(alerts);
}