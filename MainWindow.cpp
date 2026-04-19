/**
 * MainWindow.cpp
 *
 * Bug 3 修复：connectAllSignals() 的实现原本缺失（文件为空），
 * 导致 PoseDetectorManager::poseReadyWithId 信号从未连接到
 * DetectionDisplayWidget::setPosesWithId 槽，关节点骨架因此
 * 永远不会被绘制。
 *
 * 本文件提供完整实现。
 */

#include "MainWindow_Standard.h"

#include <QCloseEvent>
#include <QKeyEvent>
#include <QFileDialog>
#include <QMenuBar>
#include <QStatusBar>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QVideoFrame>
#include <QDateTime>
#include <QDebug>
#include <QDialog>
#include <QLabel>
#include <QScrollArea>
#include <QPushButton>
#include <QGroupBox>
#include <QTabWidget>
#include <QMessageBox>
#include <QFileInfo>
#include "widgets/KpiWidget.h"

// ── 构造 / 析构 ───────────────────────────────────────────────────────────────

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , m_dashboard(nullptr)
    , m_detectionPanel(nullptr)
    , m_limbMotionPanel(nullptr)
    , m_inferenceThread(nullptr)
    , m_mediaPlayer(nullptr)
    , m_videoSink(nullptr)
    , m_uiUpdateTimer(nullptr)
    , m_statsUpdateTimer(nullptr)
    , m_isRunning(false)
    , m_frameSequence(0)
    , m_inferenceFps(0.0)
    , m_detectedPersons(0)
    , m_useTensorRT(false)
    , m_leftArmFreq(0.0)
    , m_rightArmFreq(0.0)
    , m_leftLegFreq(0.0)
    , m_rightLegFreq(0.0)
    , m_lastKpiUpdateTime(0)
{
    setWindowTitle(QStringLiteral("早产儿卧床肢体姿态监控系统"));

    buildLayout();
    buildMenuBar();
    createStatusBar();

    // ⚠️ 不在此处调用 connectAllSignals()！
    // 此时 m_poseDetector / m_videoSink / m_uiUpdateTimer 均为 nullptr，
    // 即使守卫判断不崩溃，等 initSystem() 再次调用时会造成信号槽重复连接：
    //   - updateUiTick / updateStatistics 每 tick 被触发两次
    //   - poseReadyWithId 系列信号双连接导致 pushPoseResult 重复调用、告警双触发
    // 正确做法：仅在 initSystem() 中所有组件就绪后连接一次。
    QTimer::singleShot(0, this, &MainWindow::initSystem);
}

MainWindow::~MainWindow() {
    stopMonitoring();
}

// ── UI 构建 ──────────────────────────────────────────────────────────────────

void MainWindow::buildLayout() {
    m_dashboard = new DashboardWidget(this);
    setCentralWidget(m_dashboard);

    // DetectionDisplayWidget 由 DashboardWidget 内部创建并持有，
    // 通过 detectionPanel() 取得引用供外部连接信号。
    m_detectionPanel  = m_dashboard->detectionPanel();
    m_limbMotionPanel = m_dashboard->limbMotionPanel();
}

void MainWindow::buildMenuBar() {
    QMenu *fileMenu = menuBar()->addMenu(QStringLiteral("文件"));
    fileMenu->addAction(QStringLiteral("打开视频文件..."), this, &MainWindow::openVideoFile);
    fileMenu->addAction(QStringLiteral("打开摄像头"),      this, &MainWindow::openCamera);
    fileMenu->addSeparator();
    // ── 新增：模型文件选择（可选，不选则使用 Stub 合成模式）────────────────
    fileMenu->addAction(QStringLiteral("选择推理模型文件（可选）..."), this, &MainWindow::openModelFile);
    fileMenu->addSeparator();
    fileMenu->addAction(QStringLiteral("退出"), this, &QWidget::close);

    QMenu *backendMenu = menuBar()->addMenu(QStringLiteral("推理后端"));
    backendMenu->addAction(QStringLiteral("ONNX Runtime"), this, &MainWindow::switchBackendONNX);
    backendMenu->addAction(QStringLiteral("TensorRT"),     this, &MainWindow::switchBackendTensorRT);

    // ── 视图菜单 ─────────────────────────────────────────────────────────
    QMenu *viewMenu = menuBar()->addMenu(QStringLiteral("视图"));
    QAction *actKeypoints = viewMenu->addAction(QStringLiteral("显示关键点坐标叠加"));
    actKeypoints->setCheckable(true);
    actKeypoints->setChecked(true);
    connect(actKeypoints, &QAction::toggled, this, [this](bool on) {
        if (m_detectionPanel)
            m_detectionPanel->setShowKeypointCoords(on);
    });

    QAction *actLabels = viewMenu->addAction(QStringLiteral("显示关键点名称标签"));
    actLabels->setCheckable(true);
    actLabels->setChecked(false);
    connect(actLabels, &QAction::toggled, this, [this](bool on) {
        if (m_detectionPanel)
            m_detectionPanel->setShowLabels(on);
    });

    // ── 帮助菜单 ─────────────────────────────────────────────────────────
    QMenu *helpMenu = menuBar()->addMenu(QStringLiteral("帮助"));
    helpMenu->addAction(QStringLiteral("系统介绍"), this, &MainWindow::showSystemInfo);
}

void MainWindow::createStatusBar() {
    statusBar()->showMessage(QStringLiteral("就绪"));
}

// ── Bug 3 修复：connectAllSignals() ─────────────────────────────────────────
//
// 原文件为空，导致以下关键信号从未连接：
//
//   PoseDetectorManager::poseReadyWithId
//       → DetectionDisplayWidget::setPosesWithId    ← 骨架不显示的根因
//
//   DetectionDisplayWidget::frameReady
//       → PoseDetectorManager::submitFrame           ← 帧不送推理的根因
//
// 以下全部信号均通过 Qt::QueuedConnection 跨线程投递，
// 保证推理结果在主线程绘制，不存在数据竞争。
// ─────────────────────────────────────────────────────────────────────────────
void MainWindow::connectAllSignals() {

    // ── 1. 视频帧 → 推理器 ───────────────────────────────────────────────────
    //    DetectionDisplayWidget 在 setFrame() 中 emit frameReady(img, frameId)，
    //    PoseDetectorManager::submitFrame(img, frameId) 接收后转发给 PoseWorker。
    if (m_detectionPanel && m_poseDetector) {
        connect(m_detectionPanel, &DetectionDisplayWidget::frameReady,
                m_poseDetector.get(),
                qOverload<const QImage&, quint64>(&PoseDetectorManager::submitFrame),
                static_cast<Qt::ConnectionType>(Qt::QueuedConnection | Qt::UniqueConnection));
    }

    // ── 2. 推理结果 → UI 骨架绘制 ───────────────────────────────────────────
    //    【Bug 3 关键修复】
    //    poseReadyWithId 在推理线程 emit，通过 QueuedConnection 投递到主线程，
    //    setPosesWithId() 更新 m_poseCache 并在帧到达时绘制骨架。
    if (m_poseDetector && m_detectionPanel) {
        connect(m_poseDetector.get(), &PoseDetectorManager::poseReadyWithId,
                m_detectionPanel,     &DetectionDisplayWidget::setPosesWithId,
                static_cast<Qt::ConnectionType>(Qt::QueuedConnection | Qt::UniqueConnection));
    }

    // ── 3. 推理结果 → MainWindow 业务槽 ─────────────────────────────────────
    if (m_poseDetector) {
        connect(m_poseDetector.get(), &PoseDetectorManager::poseReadyWithId,
                this,                 &MainWindow::onPoseResultsReady,
                static_cast<Qt::ConnectionType>(Qt::QueuedConnection | Qt::UniqueConnection));
    }

    // ── 4. 视频 Sink → 帧回调 ────────────────────────────────────────────────
    if (m_videoSink) {
        connect(m_videoSink, &QVideoSink::videoFrameChanged,
                this,        &MainWindow::onVideoFrameReceived,
                static_cast<Qt::ConnectionType>(Qt::QueuedConnection | Qt::UniqueConnection));
    }

    // ── 5. UI 定时器 → 刷新槽 ────────────────────────────────────────────────
    // 使用 Qt::UniqueConnection 防止因多次调用 connectAllSignals() 造成重复连接。
    // （即便如此，initSystem() 完成后只应调用本函数一次，UniqueConnection 作为双保险。）
    if (m_uiUpdateTimer) {
        connect(m_uiUpdateTimer,    &QTimer::timeout, this, &MainWindow::updateUiTick,
                Qt::UniqueConnection);
        connect(m_statsUpdateTimer, &QTimer::timeout, this, &MainWindow::updateStatistics,
                Qt::UniqueConnection);
    }
}

// ── 系统生命周期 ──────────────────────────────────────────────────────────────

void MainWindow::initSystem() {
    // 推理器（Stub 模式或真实模型）
    m_poseDetector = std::make_unique<PoseDetectorManager>(this);
    m_poseDetector->setInputSize(288, 384);  // 模型 input_size=(288,384) — W×H
    m_poseDetector->setFP16(false);
    m_poseDetector->setVisibilityThreshold(0.3f);
    m_poseDetector->setPoseScoreThreshold(0.3f);

    // Bug 2/3 修复后：init() 即使 loadModel() 失败也会继续启动线程（Stub 模式）
    // 此处传入空字符串即可启动 Stub 合成模式；有真实模型时传入路径。
    m_poseDetector->init(m_modelPath);

    // 视频播放器
    m_mediaPlayer = new QMediaPlayer(this);
    m_videoSink   = new QVideoSink(this);
    m_mediaPlayer->setVideoSink(m_videoSink);

    // UI 刷新定时器
    m_uiUpdateTimer = new QTimer(this);
    m_uiUpdateTimer->setInterval(50);   // 20 Hz

    m_statsUpdateTimer = new QTimer(this);
    m_statsUpdateTimer->setInterval(5000);

    // 信号在所有组件创建完毕后连接（覆盖构造时的空连接）
    connectAllSignals();

    loadModel();
    statusBar()->showMessage(QStringLiteral("系统初始化完成"));
}

void MainWindow::loadModel() {
    if (m_modelPath.isEmpty()) {
        qInfo() << "[MainWindow] 无模型路径，使用 Stub 合成模式";
        if (m_detectionPanel)
            m_detectionPanel->setBackendName(QStringLiteral("Stub-Pose"));
        return;
    }
    // 实际项目中：m_poseDetector->init(m_modelPath);
}

void MainWindow::startMonitoring() {
    if (m_isRunning) return;
    m_isRunning = true;
    if (m_uiUpdateTimer)   m_uiUpdateTimer->start();
    if (m_statsUpdateTimer) m_statsUpdateTimer->start();
    statusBar()->showMessage(QStringLiteral("监控中..."));
}

void MainWindow::stopMonitoring() {
    m_isRunning = false;
    if (m_uiUpdateTimer)    m_uiUpdateTimer->stop();
    if (m_statsUpdateTimer) m_statsUpdateTimer->stop();
    if (m_mediaPlayer)      m_mediaPlayer->stop();
}

// ── 视频帧回调 ────────────────────────────────────────────────────────────────

void MainWindow::onVideoFrameReceived(const QVideoFrame &frame) {
    if (!m_isRunning || !m_detectionPanel) return;

    QVideoFrame copy(frame);
    copy.map(QVideoFrame::ReadOnly);
    QImage img = copy.toImage();
    copy.unmap();

    if (img.isNull()) return;

    // DetectionDisplayWidget::setFrame() 内部会：
    //   1. emit frameReady(img, frameId)  →  推理器
    //   2. queueFrame(img, frameId)       →  缩放线程
    m_detectionPanel->setFrame(img);
}

// ── 推理结果回调 ──────────────────────────────────────────────────────────────

void MainWindow::onPoseResultsReady(quint64 frameId,
                                     const QVector<PoseResult> &poses) {
    m_detectedPersons = poses.size();

    // 更新 FPS 显示
    if (m_detectionPanel)
        m_detectionPanel->setPoseCount(m_detectedPersons);

    // ── 推送姿态数据到四肢运动曲线面板 ──────────────────────────────────────
    if (m_limbMotionPanel && !poses.isEmpty()) {
        qint64 nowMs = QDateTime::currentMSecsSinceEpoch();
        m_limbMotionPanel->pushPoseResult(poses, nowMs);
    }

    // 检查异常姿态并触发报警
    for (const auto &pose : poses) {
        if (pose.poseScore < 0.4f) {
            onAlarmEvent(1, QStringLiteral("低置信度姿态（帧 %1）").arg(frameId));
        }
    }
}

void MainWindow::onInferenceFpsUpdated(double fps) {
    m_inferenceFps = fps;
    if (m_detectionPanel)
        m_detectionPanel->setFps(fps);
}

void MainWindow::onAlarmEvent(int level, const QString &message) {
    qWarning() << "[Alarm] Level" << level << ":" << message;
    statusBar()->showMessage(message, 3000);
}

// ── 定时刷新槽 ────────────────────────────────────────────────────────────────

void MainWindow::updateUiTick() {
    // DetectionDisplayWidget 有自己的 33ms 定时器驱动 paintEvent，
    // 此处进行状态栏 + KPI 轻量刷新（20Hz）。
    if (!m_isRunning) return;

    QString fps = m_inferenceFps > 0
                  ? QString::number(m_inferenceFps, 'f', 1)
                  : QStringLiteral("--");
    statusBar()->showMessage(
        QString("推理 FPS: %1 | 检测人数: %2 | 模型: %3")
            .arg(fps)
            .arg(m_detectedPersons)
            .arg(m_modelPath.isEmpty()
                 ? QStringLiteral("Stub合成")
                 : QFileInfo(m_modelPath).fileName()));

    // 每 1 秒（每 20 tick）更新一次 KPI 数字
    static int tickCount = 0;
    if (++tickCount >= 20) {
        tickCount = 0;
        RealtimeStats::instance().recordFrame(m_detectedPersons > 0, m_detectedPersons);
        updateKpiDisplay();
    }
}

void MainWindow::updateStatistics() {
    // 每 5 秒动态刷新 KPI + 仪表盘 + 告警
    updateKpiDisplay();
    updateRealtimeCharts();
}

// ── 新增：动态刷新仪表盘图表、仪表盘数值、告警栏 ──────────────────────────────
void MainWindow::updateRealtimeCharts() {
    auto& stats = RealtimeStats::instance();

    // 1. 仪表盘：根据实时检出率和运行时长计算异常占比
    double detRate  = stats.detectionRate();
    double anomRate = 0.0;
    qint64 det      = stats.detectedFrames();
    if (det > 0) {
        // 简单估算：将低置信度帧视为潜在异常
        anomRate = qMin(0.99, (double)stats.anomalyCount() / qMax(det, (qint64)1));
    }
    double nursingRate = qMax(0.80, 1.0 - anomRate * 0.1); // 演示：护理完成率与异常率反向关联

    m_dashboard->updateGaugeData(anomRate, nursingRate);

    // 2. 实时告警栏：根据运行状态动态更新
    QStringList alerts;
    if (!m_isRunning) {
        alerts << QStringLiteral("系统待机中，等待视频信号")
               << QStringLiteral("请通过【文件】菜单打开视频或摄像头")
               << QStringLiteral("模型文件可选，不选则使用 Stub 合成模式");
    } else {
        // 运行中：显示实时统计
        qint64 dur = stats.sessionDuration() / 1000;
        alerts << QString(QStringLiteral("运行时长 %1 秒 | 检测率 %2%"))
                       .arg(dur).arg(detRate, 0, 'f', 1);
        if (m_detectedPersons > 0)
            alerts << QString(QStringLiteral("当前检测到 %1 人 | 推理 FPS %.1f"))
                           .arg(m_detectedPersons).arg(m_inferenceFps);
        else
            alerts << QStringLiteral("暂未检测到人体，请调整摄像角度");
        alerts << QString(QStringLiteral("已处理帧数：%1")).arg(stats.totalFrames());
    }
    m_dashboard->updateAlerts(alerts);
}

void MainWindow::updateKpiDisplay() {
    auto& stats = RealtimeStats::instance();
    
    // 计算会话时长
    qint64 durationMs = stats.sessionDuration();
    int hours = durationMs / 3600000;
    int minutes = (durationMs % 3600000) / 60000;
    int seconds = (durationMs % 60000) / 1000;
    QString durationStr = QString::asprintf("%02d:%02d:%02d", hours, minutes, seconds);
    
    // 格式化数字（添加千分位逗号）
    auto formatNumber = [](qint64 n) -> QString {
        QString s = QString::number(n);
        int len = s.length();
        for (int i = len - 3; i > 0; i -= 3) {
            s.insert(i, ',');
        }
        return s;
    };
    
    // 实时统计KPI（今日/本次会话）
    QString today0 = formatNumber(stats.totalFrames());
    QString today1 = formatNumber(stats.detectedFrames());
    QVector<KpiItem> todayItems = {
        {"本次运行时长", durationStr, ""},
        {"监测帧数", today0, "帧"},
        {"检测到人体", today1, "次"},
    };
    
    // 运动统计 + 详细统计KPI
    QString month0 = QString::number(m_leftArmFreq + m_rightArmFreq, 'f', 2);
    QString month1 = QString::number(m_leftLegFreq + m_rightLegFreq, 'f', 2);
    QString month2 = QString::number(stats.detectionRate(), 'f', 1);
    QVector<KpiItem> monthItems = {
        {"上肢体频率", month0, "Hz"},
        {"下肢体频率", month1, "Hz"},
        {"检出率", month2, "%"},
    };
    
    m_dashboard->updateKpiData(todayItems, monthItems);
}

// ── 菜单操作 ──────────────────────────────────────────────────────────────────

void MainWindow::openVideoFile() {
    QString path = QFileDialog::getOpenFileName(
        this, QStringLiteral("打开视频文件"), {},
        QStringLiteral("视频文件 (*.mp4 *.avi *.mkv *.mov);;所有文件 (*)"));
    if (path.isEmpty()) return;

    if (m_mediaPlayer) {
        m_mediaPlayer->setSource(QUrl::fromLocalFile(path));
        m_mediaPlayer->play();
        startMonitoring();
    }
}

void MainWindow::openCamera() {
    // 实际项目中接入 QCamera；此处给出占位实现
    statusBar()->showMessage(QStringLiteral("摄像头功能：请接入 QCamera 实现"), 3000);
    startMonitoring();
}

// ── 新增：模型文件选择（可选）────────────────────────────────────────────────
void MainWindow::openModelFile() {
    QString path = QFileDialog::getOpenFileName(
        this,
        QStringLiteral("选择推理模型文件（可选，跳过则使用 Stub 合成模式）"),
        {},
        QStringLiteral("模型文件 (*.onnx *.trt *.engine *.plan);;所有文件 (*)"));

    if (path.isEmpty()) {
        // 用户取消或未选择 → 继续使用 Stub 模式，不报错
        statusBar()->showMessage(
            QStringLiteral("未选择模型文件，继续使用 Stub 合成模式"), 3000);
        return;
    }

    m_modelPath = path;
    statusBar()->showMessage(
        QString(QStringLiteral("已加载模型：%1")).arg(QFileInfo(path).fileName()), 3000);

    // 若推理器已创建，热重载模型
    if (m_poseDetector) {
        m_poseDetector->init(m_modelPath);
        if (m_detectionPanel)
            m_detectionPanel->setBackendName(
                m_useTensorRT ? QStringLiteral("TensorRT") : QStringLiteral("ONNX Runtime"));
    }
}

// ── 新增：系统介绍对话框 ──────────────────────────────────────────────────────
void MainWindow::showSystemInfo() {
    auto *dlg = new QDialog(this);
    dlg->setWindowTitle(QStringLiteral("系统介绍"));
    dlg->resize(700, 520);
    dlg->setStyleSheet(
        "QDialog { background:#040d2c; color:#eef5ff; }"
        "QLabel  { color:#eef5ff; }"
        "QTabWidget::pane { border:1px solid #1a3a6a; background:#06184a; }"
        "QTabBar::tab { background:#06184a; color:#88aacc; padding:6px 18px;"
        "               border:1px solid #1a3a6a; }"
        "QTabBar::tab:selected { background:#0c3080; color:#00e5ff; }"
        "QScrollArea { border:none; background:transparent; }"
        "QPushButton { background:#0c3080; color:#00e5ff; border:1px solid #00e5ff;"
        "              padding:6px 20px; border-radius:4px; }"
        "QPushButton:hover { background:#1a78e4; }");

    auto *tabs = new QTabWidget(dlg);

    // ── Tab 1：系统概述 ────────────────────────────────────────────────────
    auto makeScrollLabel = [](const QString &html) -> QScrollArea* {
        auto *lbl = new QLabel(html);
        lbl->setWordWrap(true);
        lbl->setTextFormat(Qt::RichText);
        lbl->setAlignment(Qt::AlignTop | Qt::AlignLeft);
        lbl->setStyleSheet("padding:12px; color:#eef5ff; line-height:160%;");
        auto *sc = new QScrollArea;
        sc->setWidget(lbl);
        sc->setWidgetResizable(true);
        sc->setStyleSheet("background:#06184a;");
        return sc;
    };

    const QString overviewHtml = QStringLiteral(
        "<h2 style='color:#00e5ff;'>早产儿卧床肢体姿态监控系统</h2>"
        "<p style='color:#88aacc;'>版本 v1.0.0 &nbsp;|&nbsp; 基于 Qt 6 + ONNX Runtime / TensorRT</p>"
        "<hr style='border-color:#1a3a6a;'/>"
        "<p>本系统面向 NICU（新生儿重症监护病房）场景，利用计算机视觉技术对早产儿的"
        "卧床肢体姿态进行 <b style='color:#00ffa0;'>7×24 小时实时监测</b>，"
        "辅助护理人员及时发现异常体位，降低并发症风险。</p>"
        "<h3 style='color:#00e5ff;'>🎯 核心功能</h3>"
        "<ul>"
        "<li>基于 COCO-17 骨架模型的人体姿态关键点检测（17 个关键点）</li>"
        "<li>实时骨架可视化叠加（连线 + 关键点 + 置信度）</li>"
        "<li>关键点坐标实时打印与视频帧叠加显示</li>"
        "<li>多目标检测，支持同时监测多名患儿</li>"
        "<li>异常姿态自动预警与分级报警</li>"
        "<li>护理操作记录与统计报表</li>"
        "</ul>"
        "<h3 style='color:#00e5ff;'>🏗️ 系统架构</h3>"
        "<ul>"
        "<li><b style='color:#ffa05a;'>视频输入层</b>：QMediaPlayer / QCamera 实时采集</li>"
        "<li><b style='color:#ffa05a;'>推理层</b>：后台线程独立推理，不阻塞 UI（支持 ONNX Runtime / TensorRT）</li>"
        "<li><b style='color:#ffa05a;'>渲染层</b>：帧缩放工作线程 + 主线程 20Hz 合成绘制</li>"
        "<li><b style='color:#ffa05a;'>数据层</b>：实时统计管理器 RealtimeStats（无锁原子操作）</li>"
        "</ul>");
    tabs->addTab(makeScrollLabel(overviewHtml), QStringLiteral("系统概述"));

    // ── Tab 2：使用说明 ────────────────────────────────────────────────────
    const QString usageHtml = QStringLiteral(
        "<h3 style='color:#00e5ff;'>📂 快速开始</h3>"
        "<ol>"
        "<li><b>打开视频</b>：菜单 <code style='color:#ffa05a;'>文件 → 打开视频文件</code>，"
        "支持 MP4 / AVI / MKV / MOV 格式。</li>"
        "<li><b>选择模型（可选）</b>：菜单 <code style='color:#ffa05a;'>文件 → 选择推理模型文件</code>，"
        "支持 <code>.onnx</code> / <code>.trt</code> / <code>.engine</code> 格式。"
        "<br/>若不选择模型，系统自动进入 <b style='color:#00ffa0;'>Stub 合成模式</b>，使用随机数据演示界面效果。</li>"
        "<li><b>切换后端</b>：菜单 <code style='color:#ffa05a;'>推理后端</code> 可在 ONNX Runtime 与 TensorRT 之间切换。</li>"
        "<li><b>视图选项</b>：菜单 <code style='color:#ffa05a;'>视图</code> 可开关关键点坐标叠加、关键点名称标签。</li>"
        "</ol>"
        "<h3 style='color:#00e5ff;'>🖥️ 界面说明</h3>"
        "<ul>"
        "<li><b>左侧面板</b>：肢体姿态分类饼图、异常占比 / 护理响应仪表盘、实时预警栏</li>"
        "<li><b>中心面板</b>：视频画面 + 骨架叠加 + 关键点坐标面板</li>"
        "<li><b>右侧面板</b>：实时 KPI 统计、异常姿态 TOP 榜、床位风险排行</li>"
        "<li><b>底部中心</b>：肢体占比甜甜圈图、月度趋势柱状图</li>"
        "</ul>"
        "<h3 style='color:#00e5ff;'>⌨️ 快捷键</h3>"
        "<ul>"
        "<li><code>Esc</code>：停止监控</li>"
        "</ul>");
    tabs->addTab(makeScrollLabel(usageHtml), QStringLiteral("使用说明"));

    // ── Tab 3：关键点说明 ──────────────────────────────────────────────────
    const QString kptHtml = QStringLiteral(
        "<h3 style='color:#00e5ff;'>🦴 COCO-17 关键点索引</h3>"
        "<table style='border-collapse:collapse; width:100%;'>"
        "<tr style='color:#00e5ff;'><th align='left'>索引</th><th align='left'>名称</th>"
        "<th align='left'>颜色</th><th align='left'>部位</th></tr>"
        "<tr><td>0</td><td>Nose</td><td>灰色</td><td>鼻子</td></tr>"
        "<tr><td>1</td><td>LeftEye</td><td style='color:#1e90ff;'>蓝色</td><td>左眼</td></tr>"
        "<tr><td>2</td><td>RightEye</td><td style='color:#ff5050;'>红色</td><td>右眼</td></tr>"
        "<tr><td>3</td><td>LeftEar</td><td style='color:#1e90ff;'>蓝色</td><td>左耳</td></tr>"
        "<tr><td>4</td><td>RightEar</td><td style='color:#ff5050;'>红色</td><td>右耳</td></tr>"
        "<tr><td>5</td><td>LeftShoulder</td><td style='color:#1e90ff;'>蓝色</td><td>左肩</td></tr>"
        "<tr><td>6</td><td>RightShoulder</td><td style='color:#ff5050;'>红色</td><td>右肩</td></tr>"
        "<tr><td>7</td><td>LeftElbow</td><td style='color:#1e90ff;'>蓝色</td><td>左肘</td></tr>"
        "<tr><td>8</td><td>RightElbow</td><td style='color:#ff5050;'>红色</td><td>右肘</td></tr>"
        "<tr><td>9</td><td>LeftWrist</td><td style='color:#1e90ff;'>蓝色</td><td>左腕</td></tr>"
        "<tr><td>10</td><td>RightWrist</td><td style='color:#ff5050;'>红色</td><td>右腕</td></tr>"
        "<tr><td>11</td><td>LeftHip</td><td style='color:#1e90ff;'>蓝色</td><td>左髋</td></tr>"
        "<tr><td>12</td><td>RightHip</td><td style='color:#ff5050;'>红色</td><td>右髋</td></tr>"
        "<tr><td>13</td><td>LeftKnee</td><td style='color:#1e90ff;'>蓝色</td><td>左膝</td></tr>"
        "<tr><td>14</td><td>RightKnee</td><td style='color:#ff5050;'>红色</td><td>右膝</td></tr>"
        "<tr><td>15</td><td>LeftAnkle</td><td style='color:#1e90ff;'>蓝色</td><td>左踝</td></tr>"
        "<tr><td>16</td><td>RightAnkle</td><td style='color:#ff5050;'>红色</td><td>右踝</td></tr>"
        "</table>"
        "<p style='margin-top:12px;'>骨架连线颜色规则：<span style='color:#1e90ff;'>蓝色</span>=左侧肢体，"
        "<span style='color:#ff5050;'>红色</span>=右侧肢体，灰色=中轴（头部/躯干）</p>");
    tabs->addTab(makeScrollLabel(kptHtml), QStringLiteral("关键点说明"));

    // ── Tab 4：关于 ───────────────────────────────────────────────────────
    const QString aboutHtml = QStringLiteral(
        "<h3 style='color:#00e5ff;'>关于本系统</h3>"
        "<ul>"
        "<li><b>开发框架</b>：Qt 6.x (C++17)</li>"
        "<li><b>推理框架</b>：ONNX Runtime 1.x / TensorRT 8.x+</li>"
        "<li><b>姿态模型</b>：COCO-17 Keypoint Detection（17 关键点）</li>"
        "<li><b>推理线程</b>：独立后台线程，主线程 20Hz UI 刷新</li>"
        "<li><b>视频输入</b>：Qt Multimedia（QMediaPlayer / QCamera）</li>"
        "</ul>"
        "<h3 style='color:#00e5ff;'>免责声明</h3>"
        "<p>本系统仅作为辅助工具，检测结果不能替代医护人员的专业判断。"
        "所有报警信息须经临床人员核实后方可采取护理措施。</p>"
        "<p style='color:#88aacc; margin-top:20px;'>© 2024-2026  Hospital Dashboard Project</p>");
    tabs->addTab(makeScrollLabel(aboutHtml), QStringLiteral("关于"));

    auto *btnClose = new QPushButton(QStringLiteral("关闭"), dlg);
    connect(btnClose, &QPushButton::clicked, dlg, &QDialog::accept);

    auto *mainLayout = new QVBoxLayout(dlg);
    mainLayout->setContentsMargins(12, 12, 12, 12);
    mainLayout->addWidget(tabs);
    mainLayout->addWidget(btnClose, 0, Qt::AlignRight);

    dlg->exec();
    dlg->deleteLater();
}

void MainWindow::switchBackendONNX() {
    m_useTensorRT = false;
    statusBar()->showMessage(QStringLiteral("已切换至 ONNX Runtime 后端"), 2000);
    if (m_detectionPanel)
        m_detectionPanel->setBackendName(QStringLiteral("ONNX Runtime"));
}

void MainWindow::switchBackendTensorRT() {
    m_useTensorRT = true;
    statusBar()->showMessage(QStringLiteral("已切换至 TensorRT 后端"), 2000);
    if (m_detectionPanel)
        m_detectionPanel->setBackendName(QStringLiteral("TensorRT"));
}

// ── 窗口事件 ──────────────────────────────────────────────────────────────────

void MainWindow::closeEvent(QCloseEvent *event) {
    stopMonitoring();
    event->accept();
}

void MainWindow::keyPressEvent(QKeyEvent *event) {
    if (event->key() == Qt::Key_Escape)
        stopMonitoring();
    else
        QMainWindow::keyPressEvent(event);
}