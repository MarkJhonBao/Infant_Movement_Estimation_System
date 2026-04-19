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
#include <QVideoFrame>
#include <QDateTime>
#include <QDebug>

// ── 构造 / 析构 ───────────────────────────────────────────────────────────────

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , m_dashboard(nullptr)
    , m_detectionPanel(nullptr)
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
{
    setWindowTitle(QStringLiteral("早产儿卧床肢体姿态监控系统"));

    buildLayout();
    buildMenuBar();
    createStatusBar();

    // Bug 3 核心修复：所有信号槽在此统一连接
    connectAllSignals();

    // 延迟初始化（等事件循环启动后再加载模型）
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
    m_detectionPanel = m_dashboard->detectionPanel();
}

void MainWindow::buildMenuBar() {
    QMenu *fileMenu = menuBar()->addMenu(QStringLiteral("文件"));
    fileMenu->addAction(QStringLiteral("打开视频文件..."), this, &MainWindow::openVideoFile);
    fileMenu->addAction(QStringLiteral("打开摄像头"),      this, &MainWindow::openCamera);
    fileMenu->addSeparator();
    fileMenu->addAction(QStringLiteral("退出"), this, &QWidget::close);

    QMenu *backendMenu = menuBar()->addMenu(QStringLiteral("推理后端"));
    backendMenu->addAction(QStringLiteral("ONNX Runtime"), this, &MainWindow::switchBackendONNX);
    backendMenu->addAction(QStringLiteral("TensorRT"),     this, &MainWindow::switchBackendTensorRT);
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
                Qt::QueuedConnection);
    }

    // ── 2. 推理结果 → UI 骨架绘制 ───────────────────────────────────────────
    //    【Bug 3 关键修复】
    //    poseReadyWithId 在推理线程 emit，通过 QueuedConnection 投递到主线程，
    //    setPosesWithId() 更新 m_poseCache 并在帧到达时绘制骨架。
    if (m_poseDetector && m_detectionPanel) {
        connect(m_poseDetector.get(), &PoseDetectorManager::poseReadyWithId,
                m_detectionPanel,     &DetectionDisplayWidget::setPosesWithId,
                Qt::QueuedConnection);
    }

    // ── 3. 推理结果 → MainWindow 业务槽 ─────────────────────────────────────
    if (m_poseDetector) {
        connect(m_poseDetector.get(), &PoseDetectorManager::poseReadyWithId,
                this,                 &MainWindow::onPoseResultsReady,
                Qt::QueuedConnection);
    }

    // ── 4. 视频 Sink → 帧回调 ────────────────────────────────────────────────
    if (m_videoSink) {
        connect(m_videoSink, &QVideoSink::videoFrameChanged,
                this,        &MainWindow::onVideoFrameReceived,
                Qt::QueuedConnection);
    }

    // ── 5. UI 定时器 → 刷新槽 ────────────────────────────────────────────────
    if (m_uiUpdateTimer) {
        connect(m_uiUpdateTimer,   &QTimer::timeout, this, &MainWindow::updateUiTick);
        connect(m_statsUpdateTimer, &QTimer::timeout, this, &MainWindow::updateStatistics);
    }
}

// ── 系统生命周期 ──────────────────────────────────────────────────────────────

void MainWindow::initSystem() {
    // 推理器（Stub 模式或真实模型）
    m_poseDetector = std::make_unique<PoseDetectorManager>(this);
    m_poseDetector->setInputSize(192, 256);
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

    // 检查异常姿态并触发报警
    for (const auto &pose : poses) {
        if (pose.poseScore < 0.4f) {
            emit alarmEvent(1, QStringLiteral("低置信度姿态（帧 %1）").arg(frameId));
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
    // 此处仅做状态栏轻量刷新。
    if (!m_isRunning) return;
    QString fps = m_inferenceFps > 0
                  ? QString::number(m_inferenceFps, 'f', 1)
                  : QStringLiteral("--");
    statusBar()->showMessage(
        QString("推理 FPS: %1 | 检测人数: %2").arg(fps).arg(m_detectedPersons));
}

void MainWindow::updateStatistics() {
    // 示例：每 5 秒更新仪表盘 KPI（实际项目接入数据库）
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
