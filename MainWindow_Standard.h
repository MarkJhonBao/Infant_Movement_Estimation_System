#pragma once

#include <QMainWindow>
#include <QTimer>
#include <QThread>
#include <QMediaPlayer>
#include <QVideoSink>
#include <QDateTime>
#include <atomic>
#include <memory>

#include "widgets/DashboardWidget.h"
#include "widgets/DetectionDisplayWidget.h"
#include "widgets/LimbMotionWidget.h"
#include "detector/PoseDetectorManager.h"

// ─────────────────────────────────────────────────────────────────────────────
//  实时统计数据管理器
// ─────────────────────────────────────────────────────────────────────────────
class RealtimeStats {
public:
    static RealtimeStats& instance() {
        static RealtimeStats inst;
        return inst;
    }

    void reset() {
        m_totalFrames = 0;
        m_detectedFrames = 0;
        m_anomalyCount = 0;
        m_totalPersons = 0;
        m_sessionStartTime = QDateTime::currentMSecsSinceEpoch();
    }

    void recordFrame(bool hasPose, int personCount = 0, bool isAnomaly = false) {
        m_totalFrames.fetch_add(1);
        if (hasPose) {
            m_detectedFrames.fetch_add(1);
            m_totalPersons.fetch_add(personCount);
        }
        if (isAnomaly) {
            m_anomalyCount.fetch_add(1);
        }
    }

    qint64 totalFrames() const { return m_totalFrames.load(); }
    qint64 detectedFrames() const { return m_detectedFrames.load(); }
    qint64 anomalyCount() const { return m_anomalyCount.load(); }
    qint64 totalPersons() const { return m_totalPersons.load(); }
    
    qint64 sessionDuration() const {
        return QDateTime::currentMSecsSinceEpoch() - m_sessionStartTime.load();
    }

    double detectionRate() const {
        qint64 total = m_totalFrames.load();
        return total > 0 ? (double)m_detectedFrames.load() / total * 100.0 : 0.0;
    }

private:
    RealtimeStats() : m_sessionStartTime(QDateTime::currentMSecsSinceEpoch()) {}
    ~RealtimeStats() = default;
    
    std::atomic<qint64> m_totalFrames{0};
    std::atomic<qint64> m_detectedFrames{0};
    std::atomic<qint64> m_anomalyCount{0};
    std::atomic<qint64> m_totalPersons{0};
    std::atomic<qint64> m_sessionStartTime{0};
};

/**
 * @brief 早产儿卧床肢体姿态监控系统主窗口
 * 
 * ✅ 标准Qt架构实现
 * ✅ 多线程推理分离
 * ✅ 20Hz UI同步刷新
 * ✅ 完整信号槽连接
 * ✅ 四大显示区域布局
 */
class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow() override;

protected:
    void closeEvent(QCloseEvent *event) override;
    void keyPressEvent(QKeyEvent *event) override;

signals:
    // ── 跨线程信号 ─────────────────────────────────────────────────────────
    void submitFrameForInference(const QImage &frame, quint64 frameId);

private slots:
    // ── 系统生命周期 ─────────────────────────────────────────────────────
    void initSystem();
    void startMonitoring();
    void stopMonitoring();
    void loadModel();

    // ── 视频输入回调 ─────────────────────────────────────────────────────
    void onVideoFrameReceived(const QVideoFrame &frame);
    
    // ── 推理结果回调 ─────────────────────────────────────────────────────
    void onPoseResultsReady(quint64 frameId, const QVector<PoseResult> &poses);
    void onInferenceFpsUpdated(double fps);
    void onAlarmEvent(int level, const QString &message);

    // ── UI定时刷新 ───────────────────────────────────────────────────────
    void updateUiTick();
    void updateStatistics();

    // ── 菜单操作 ─────────────────────────────────────────────────────────
    void openVideoFile();
    void openCamera();
    void openModelFile();           // 新增：选择模型文件（可选）
    void switchBackendONNX();
    void switchBackendTensorRT();

    // ── 帮助 ──────────────────────────────────────────────────────────────
    void showSystemInfo();          // 新增：显示系统介绍对话框

private:
    // ── UI构建 ───────────────────────────────────────────────────────────
    void buildLayout();
    void buildMenuBar();
    void createStatusBar();
    void connectAllSignals();
    void loadApplicationStyleSheet();

    // ── 核心组件 ─────────────────────────────────────────────────────────
    DashboardWidget        *m_dashboard;
    DetectionDisplayWidget *m_detectionPanel;
    LimbMotionWidget       *m_limbMotionPanel;  // 左下角四肢运动面板

    // ── 多线程推理架构 ───────────────────────────────────────────────────
    QThread *m_inferenceThread;
    std::unique_ptr<PoseDetectorManager> m_poseDetector;

    // ── 视频输入 ─────────────────────────────────────────────────────────
    QMediaPlayer *m_mediaPlayer;
    QVideoSink *m_videoSink;

    // ── 定时同步 ─────────────────────────────────────────────────────────
    QTimer *m_uiUpdateTimer;        // 20Hz 固定UI刷新
    QTimer *m_statsUpdateTimer;     // 5秒 统计更新

    // ── 运行状态 ─────────────────────────────────────────────────────────
    bool m_isRunning;
    quint64 m_frameSequence;
    double m_inferenceFps;
    int m_detectedPersons;

    // ── 配置 ─────────────────────────────────────────────────────────────
    bool m_useTensorRT;
    QString m_modelPath;

    // ── 实时统计 ───────────────────────────────────────────────────────
    double m_leftArmFreq{0.0};
    double m_rightArmFreq{0.0};
    double m_leftLegFreq{0.0};
    double m_rightLegFreq{0.0};
    qint64 m_lastKpiUpdateTime{0};

    // ── 动态KPI更新 ──────────────────────────────────────────────────
    void updateKpiDisplay();

    // ── 动态图表刷新 ─────────────────────────────────────────────────
    void updateRealtimeCharts();     // 新增：刷新仪表盘 / 告警 / 图表
};