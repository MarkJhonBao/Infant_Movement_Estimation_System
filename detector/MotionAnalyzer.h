#pragma once

#include <QObject>
#include <QVector>
#include <QPointF>
#include <QTimer>
#include <QQueue>
#include <QMutex>
#include <QMutexLocker>

#include "PoseResult.h"

/**
 * @brief 肢体运动频率分析器
 *
 * 实时跟踪关节点运动，计算5秒周期内的运动频率
 * 使用峰值检测算法统计肢体摆动次数
 */
class MotionAnalyzer : public QObject
{
    Q_OBJECT
public:
    enum JointGroup {
        LeftArm,
        RightArm,
        LeftLeg,
        RightLeg,
        JointGroupCount
    };

    static MotionAnalyzer& instance();

    void pushPoseData(const PoseResult &pose, quint64 timestamp);

    double getFrequency(JointGroup group);
    QVector<double> getAllFrequencies();

    void startStatisticsCycle(int intervalMs = 5000);

signals:
    void statisticsUpdated(double leftArm, double rightArm,
                           double leftLeg, double rightLeg);

private slots:
    void onStatisticsCycle();

private:
    MotionAnalyzer(QObject *parent = nullptr);
    ~MotionAnalyzer() override;

    // 注意：这些方法不是const，因为它们需要获取互斥锁
    double calculateGroupFrequency(JointGroup group);
    double calculateMovementMagnitude(int keypointIndex);
    int countPeaks(const QVector<double> &signal) const;

    struct JointHistory {
        QQueue<double> xPositions;
        QQueue<double> yPositions;
        QQueue<quint64> timestamps;
    };

    mutable QMutex m_mutex;  // 保护共享数据的互斥锁
    QVector<JointHistory> m_jointHistory;
    QTimer *m_statisticsTimer;

    double m_lastFreq[JointGroupCount];
    static const int HISTORY_MAX_SIZE = 500;
    static const int JOINT_INDICES[JointGroupCount][3];
};