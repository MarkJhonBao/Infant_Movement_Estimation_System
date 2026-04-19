#include "PoseDatabase.h"
#include <QTimer>
#include <QSqlError>
#include <QDebug>

PoseDatabase& PoseDatabase::instance()
{
    static PoseDatabase instance;
    return instance;
}

PoseDatabase::PoseDatabase(QObject *parent)
    : QObject(parent)
    , m_pendingCount(0)
    , m_commitTimer(nullptr)
{
}

PoseDatabase::~PoseDatabase()
{
    close();
}

bool PoseDatabase::initDatabase(const QString &dbPath)
{
    m_db = QSqlDatabase::addDatabase("QSQLITE");
    m_db.setDatabaseName(dbPath);

    if (!m_db.open()) {
        qCritical() << "Cannot open database:" << m_db.lastError().text();
        return false;
    }

    m_db.exec("PRAGMA journal_mode = WAL;");
    m_db.exec("PRAGMA synchronous = NORMAL;");
    m_db.exec("PRAGMA cache_size = 10000;");
    m_db.exec("PRAGMA temp_store = MEMORY;");

    if (!createTables()) {
        return false;
    }

    if (!prepareStatements()) {
        return false;
    }

    return true;
}

void PoseDatabase::close()
{
    if (m_commitTimer) {
        m_commitTimer->stop();
        delete m_commitTimer;
        m_commitTimer = nullptr;
    }

    commitBatch();

    if (m_db.isOpen()) {
        m_db.close();
    }
}

bool PoseDatabase::createTables()
{
    QSqlQuery query(m_db);

    // 检测帧表
    bool ok = query.exec(
        "CREATE TABLE IF NOT EXISTS detection_frames ("
        "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "  frame_id INTEGER UNIQUE NOT NULL,"
        "  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,"
        "  person_count INTEGER NOT NULL,"
        "  fps REAL NOT NULL"
        ");");

    if (!ok) {
        qCritical() << "Create detection_frames failed:" << query.lastError().text();
        return false;
    }

    // 姿态结果表
    ok = query.exec(
        "CREATE TABLE IF NOT EXISTS pose_results ("
        "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "  frame_id INTEGER NOT NULL,"
        "  person_index INTEGER NOT NULL,"
        "  keypoint_count INTEGER NOT NULL,"
        "  confidence REAL NOT NULL,"
        "  bbox_x REAL NOT NULL,"
        "  bbox_y REAL NOT NULL,"
        "  bbox_width REAL NOT NULL,"
        "  bbox_height REAL NOT NULL,"
        "  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,"
        "  FOREIGN KEY (frame_id) REFERENCES detection_frames(frame_id)"
        ");");

    if (!ok) {
        qCritical() << "Create pose_results failed:" << query.lastError().text();
        return false;
    }

    // 关节点详细数据表
    ok = query.exec(
        "CREATE TABLE IF NOT EXISTS keypoints ("
        "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "  pose_result_id INTEGER NOT NULL,"
        "  keypoint_index INTEGER NOT NULL,"
        "  x REAL NOT NULL,"
        "  y REAL NOT NULL,"
        "  confidence REAL NOT NULL,"
        "  FOREIGN KEY (pose_result_id) REFERENCES pose_results(id)"
        ");");

    if (!ok) {
        qCritical() << "Create keypoints failed:" << query.lastError().text();
        return false;
    }

    // 运动统计表
    ok = query.exec(
        "CREATE TABLE IF NOT EXISTS motion_statistics ("
        "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,"
        "  left_arm_frequency REAL NOT NULL,"
        "  right_arm_frequency REAL NOT NULL,"
        "  left_leg_frequency REAL NOT NULL,"
        "  right_leg_frequency REAL NOT NULL,"
        "  total_movement_score REAL NOT NULL"
        ");");

    if (!ok) {
        qCritical() << "Create motion_statistics failed:" << query.lastError().text();
        return false;
    }

    // 告警事件表
    ok = query.exec(
        "CREATE TABLE IF NOT EXISTS alarm_events ("
        "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,"
        "  level INTEGER NOT NULL,"
        "  message TEXT NOT NULL,"
        "  roi_x REAL NOT NULL,"
        "  roi_y REAL NOT NULL,"
        "  roi_width REAL NOT NULL,"
        "  roi_height REAL NOT NULL"
        ");");

    if (!ok) {
        qCritical() << "Create alarm_events failed:" << query.lastError().text();
        return false;
    }

    // 创建索引
    query.exec("CREATE INDEX IF NOT EXISTS idx_frame_id ON detection_frames(frame_id);");
    query.exec("CREATE INDEX IF NOT EXISTS idx_pose_frame_id ON pose_results(frame_id);");
    query.exec("CREATE INDEX IF NOT EXISTS idx_stat_timestamp ON motion_statistics(timestamp);");

    return true;
}

bool PoseDatabase::prepareStatements()
{
    m_insertFrameQuery = QSqlQuery(m_db);
    m_insertFrameQuery.prepare(
        "INSERT OR IGNORE INTO detection_frames (frame_id, person_count, fps) "
        "VALUES (?, ?, ?);");

    m_insertPoseQuery = QSqlQuery(m_db);
    m_insertPoseQuery.prepare(
        "INSERT INTO pose_results (frame_id, person_index, keypoint_count, confidence, "
        "bbox_x, bbox_y, bbox_width, bbox_height) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?);");

    m_insertAlarmQuery = QSqlQuery(m_db);
    m_insertAlarmQuery.prepare(
        "INSERT INTO alarm_events (level, message, roi_x, roi_y, roi_width, roi_height) "
        "VALUES (?, ?, ?, ?, ?, ?);");

    m_insertMotionQuery = QSqlQuery(m_db);
    m_insertMotionQuery.prepare(
        "INSERT INTO motion_statistics (left_arm_frequency, right_arm_frequency, "
        "left_leg_frequency, right_leg_frequency, total_movement_score) "
        "VALUES (?, ?, ?, ?, ?);");

    return true;
}

void PoseDatabase::insertDetectionFrame(quint64 frameId, int personCount, double fps)
{
    m_insertFrameQuery.bindValue(0, frameId);
    m_insertFrameQuery.bindValue(1, personCount);
    m_insertFrameQuery.bindValue(2, fps);

    if (!m_insertFrameQuery.exec()) {
        qWarning() << "Insert frame failed:" << m_insertFrameQuery.lastError().text();
    }

    m_pendingCount++;
}

void PoseDatabase::insertPoseResult(quint64 frameId, const PoseResult &pose)
{
    m_insertPoseQuery.bindValue(0, frameId);
    m_insertPoseQuery.bindValue(1, 0);
    m_insertPoseQuery.bindValue(2, pose.keypoints.size());
    m_insertPoseQuery.bindValue(3, pose.poseScore);
    m_insertPoseQuery.bindValue(4, pose.boundingBox.x());
    m_insertPoseQuery.bindValue(5, pose.boundingBox.y());
    m_insertPoseQuery.bindValue(6, pose.boundingBox.width());
    m_insertPoseQuery.bindValue(7, pose.boundingBox.height());

    if (!m_insertPoseQuery.exec()) {
        qWarning() << "Insert pose failed:" << m_insertPoseQuery.lastError().text();
        return;
    }

    // 插入所有关节点
    qint64 poseId = m_insertPoseQuery.lastInsertId().toLongLong();
    QSqlQuery keypointQuery(m_db);
    keypointQuery.prepare(
        "INSERT INTO keypoints (pose_result_id, keypoint_index, x, y, confidence) "
        "VALUES (?, ?, ?, ?, ?);");

    for (int i = 0; i < pose.keypoints.size(); i++) {
        const auto &kp = pose.keypoints[i];
        keypointQuery.bindValue(0, poseId);
        keypointQuery.bindValue(1, i);
        keypointQuery.bindValue(2, kp.x);
        keypointQuery.bindValue(3, kp.y);
        keypointQuery.bindValue(4, kp.score);

        if (!keypointQuery.exec()) {
            qWarning() << "Insert keypoint failed:" << keypointQuery.lastError().text();
        }
    }

    m_pendingCount++;
}

void PoseDatabase::insertAlarmEvent(int level, const QString &message, const QRectF &roi)
{
    m_insertAlarmQuery.bindValue(0, level);
    m_insertAlarmQuery.bindValue(1, message);
    m_insertAlarmQuery.bindValue(2, roi.x());
    m_insertAlarmQuery.bindValue(3, roi.y());
    m_insertAlarmQuery.bindValue(4, roi.width());
    m_insertAlarmQuery.bindValue(5, roi.height());

    if (!m_insertAlarmQuery.exec()) {
        qWarning() << "Insert alarm failed:" << m_insertAlarmQuery.lastError().text();
    }

    m_pendingCount++;
}

void PoseDatabase::insertMotionStatistics(quint64 timestamp,
                                           double leftArmFreq, double rightArmFreq,
                                           double leftLegFreq, double rightLegFreq)
{
    if (!m_db.isOpen()) {
        return;
    }

    Q_UNUSED(timestamp);

    double totalScore = leftArmFreq + rightArmFreq + leftLegFreq + rightLegFreq;

    m_insertMotionQuery.bindValue(0, leftArmFreq);
    m_insertMotionQuery.bindValue(1, rightArmFreq);
    m_insertMotionQuery.bindValue(2, leftLegFreq);
    m_insertMotionQuery.bindValue(3, rightLegFreq);
    m_insertMotionQuery.bindValue(4, totalScore);

    if (!m_insertMotionQuery.exec()) {
        qWarning() << "Insert motion statistics failed:" << m_insertMotionQuery.lastError().text();
    }

    m_pendingCount++;
}

void PoseDatabase::commitBatch()
{
    if (m_pendingCount > 0) {
        m_db.transaction();
        m_db.commit();
        qDebug() << "Committed" << m_pendingCount << "database records";
        m_pendingCount = 0;
    }
}

void PoseDatabase::startAutoCommit(int intervalMs)
{
    if (m_commitTimer) {
        m_commitTimer->stop();
        delete m_commitTimer;
    }

    m_commitTimer = new QTimer(this);
    connect(m_commitTimer, &QTimer::timeout, this, &PoseDatabase::commitBatch);
    m_commitTimer->start(intervalMs);
}
