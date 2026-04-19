#include "MotionAnalyzer.h"
#include <QtMath>
#include <QDebug>
#include <cmath>

const int MotionAnalyzer::JOINT_INDICES[4][3] = {
    {5, 7, 9},    // 左臂: 肩 - 肘 - 腕
    {6, 8, 10},   // 右臂: 肩 - 肘 - 腕
    {11, 13, 15}, // 左腿: 髋 - 膝 - 踝
    {12, 14, 16}  // 右腿: 髋 - 膝 - 踝
};

MotionAnalyzer& MotionAnalyzer::instance()
{
    static MotionAnalyzer instance;
    return instance;
}

MotionAnalyzer::MotionAnalyzer(QObject *parent)
    : QObject(parent)
    , m_jointHistory(17)
{
    for (int i = 0; i < 4; i++) {
        m_lastFreq[i] = 0.0;
    }

    m_statisticsTimer = new QTimer(this);
    connect(m_statisticsTimer, &QTimer::timeout, this, &MotionAnalyzer::onStatisticsCycle);
}

MotionAnalyzer::~MotionAnalyzer()
{
}

void MotionAnalyzer::startStatisticsCycle(int intervalMs)
{
    m_statisticsTimer->start(intervalMs);
}

void MotionAnalyzer::pushPoseData(const PoseResult &pose, quint64 timestamp)
{
    QMutexLocker locker(&m_mutex);
    
    for (int i = 0; i < pose.keypoints.size() && i < 17; i++) {
        const auto &kp = pose.keypoints[i];
        if (kp.score > 0.3) {
            m_jointHistory[i].xPositions.enqueue(kp.x);
            m_jointHistory[i].yPositions.enqueue(kp.y);
            m_jointHistory[i].timestamps.enqueue(timestamp);

            while (m_jointHistory[i].xPositions.size() > HISTORY_MAX_SIZE) {
                m_jointHistory[i].xPositions.dequeue();
                m_jointHistory[i].yPositions.dequeue();
                m_jointHistory[i].timestamps.dequeue();
            }
        }
    }
}

double MotionAnalyzer::getFrequency(JointGroup group)
{
    QMutexLocker locker(&m_mutex);
    return m_lastFreq[group];
}

QVector<double> MotionAnalyzer::getAllFrequencies()
{
    QMutexLocker locker(&m_mutex);
    QVector<double> result;
    for (int i = 0; i < 4; i++) {
        result.append(m_lastFreq[i]);
    }
    return result;
}

void MotionAnalyzer::onStatisticsCycle()
{
    QMutexLocker locker(&m_mutex);
    
    for (int i = 0; i < 4; i++) {
        m_lastFreq[i] = calculateGroupFrequency(static_cast<JointGroup>(i));
    }

    // 释放锁后再发出信号，避免死锁
    locker.unlock();
    
    emit statisticsUpdated(
        m_lastFreq[0], // LeftArm
        m_lastFreq[1], // RightArm
        m_lastFreq[2], // LeftLeg
        m_lastFreq[3]  // RightLeg
    );
}

double MotionAnalyzer::calculateGroupFrequency(JointGroup group)
{
    // 注意：这个方法会在onStatisticsCycle中被调用，而onStatisticsCycle已经持有锁
    // 所以这里不应再次获取锁，否则会导致死锁
    
    double totalMag = 0.0;
    int validJoints = 0;

    for (int j = 0; j < 3; j++) {
        int kpIdx = JOINT_INDICES[group][j];
        double mag = calculateMovementMagnitude(kpIdx);
        if (mag > 0) {
            totalMag += mag;
            validJoints++;
        }
    }

    if (validJoints == 0) return 0.0;

    return totalMag / validJoints;
}

double MotionAnalyzer::calculateMovementMagnitude(int keypointIndex)
{
    // 注意：这个方法应该在mutex已经被锁定的情况下被调用
    // 我们假设调用者已经获取了mutex锁
    
    const auto &history = m_jointHistory[keypointIndex];
    if (history.xPositions.size() < 10) return 0.0;

    QVector<double> displacements;
    double lastX = history.xPositions[0];
    double lastY = history.yPositions[0];

    for (int i = 1; i < history.xPositions.size(); i++) {
        double dx = history.xPositions[i] - lastX;
        double dy = history.yPositions[i] - lastY;
        displacements.append(std::sqrt(dx*dx + dy*dy));

        lastX = history.xPositions[i];
        lastY = history.yPositions[i];
    }

    return countPeaks(displacements) / 5.0; // 频率: 次/秒
}

int MotionAnalyzer::countPeaks(const QVector<double> &signal) const
{
    if (signal.size() < 3) return 0;

    int peaks = 0;
    double threshold = 2.0; // 最小位移阈值

    for (int i = 1; i < signal.size() - 1; i++) {
        if (signal[i] > signal[i-1] && signal[i] > signal[i+1] && signal[i] > threshold) {
            peaks++;
        }
    }

    return peaks;
}