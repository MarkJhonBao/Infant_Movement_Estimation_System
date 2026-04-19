#pragma once

#include <QObject>
#include <QSqlDatabase>
#include <QSqlQuery>
#include <QSqlError>
#include <QDateTime>

#include "detector/PoseResult.h"

/**
 * @brief 姿态检测结果数据库持久化
 * 
 * SQLite数据库，存储所有检测数据、姿态分类、告警事件
 * 自动每5秒批量写入，支持历史查询和统计
 */
class PoseDatabase : public QObject
{
    Q_OBJECT
public:
    static PoseDatabase& instance();
    
    bool initDatabase(const QString &dbPath = "pose_monitor.db");
    void close();
    
    bool isInitialized() const { return m_db.isOpen(); }

    // ── 数据写入接口 ─────────────────────────────────────────────────────
    void insertDetectionFrame(quint64 frameId, int personCount, double fps);
    void insertPoseResult(quint64 frameId, const PoseResult &pose);
    void insertAlarmEvent(int level, const QString &message, const QRectF &roi);
    void insertMotionStatistics(quint64 timestamp, 
                                double leftArmFreq, double rightArmFreq,
                                double leftLegFreq, double rightLegFreq);
    
    // ── 批量提交 ─────────────────────────────────────────────────────────
    void commitBatch();
    void startAutoCommit(int intervalMs = 5000);

private:
    PoseDatabase(QObject *parent = nullptr);
    ~PoseDatabase() override;

    bool createTables();
    bool prepareStatements();

    QSqlDatabase m_db;
    QSqlQuery m_insertFrameQuery;
    QSqlQuery m_insertPoseQuery;
    QSqlQuery m_insertAlarmQuery;
    QSqlQuery m_insertMotionQuery;
    
    QTimer *m_commitTimer;
    int m_pendingCount;
    
    static const int DATABASE_VERSION = 1;
};
