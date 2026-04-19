#include <QFile>
#include <QIODevice>

#include "DetectorManager.h"
#include "OnnxDetector.h"
#include "TensorRTDetector.h"
#include <QDebug>

// ─── DetectorWorker ───────────────────────────────────────────────────────────
DetectorWorker::DetectorWorker(DetectorBase* det, QObject* parent)
    : QObject(parent), m_detector(det) {}

void DetectorWorker::processFrame(const QImage& frame) {
    QMutexLocker lock(&m_mutex);
    if (m_busy) return;           // skip if previous frame still processing
    m_busy = true;
    lock.unlock();

    auto results = m_detector->detect(frame);

    QMutexLocker lock2(&m_mutex);
    m_busy = false;
    lock2.unlock();

    emit resultsReady(results);
}

// ─── DetectorManager ──────────────────────────────────────────────────────────
DetectorManager::DetectorManager(QObject* parent) : QObject(parent) {}

DetectorManager::~DetectorManager() {
    if (m_thread) {
        m_thread->quit();
        m_thread->wait();
    }
}

bool DetectorManager::init(const QString& modelPath, const QString& labelsPath) {
    // Create backend
    if (m_backend == Backend::TensorRT)
        m_detector = std::make_unique<TensorRTDetector>();
    else
        m_detector = std::make_unique<OnnxDetector>();

    // Load labels if provided
    if (!labelsPath.isEmpty()) {
        if (auto* onnx = dynamic_cast<OnnxDetector*>(m_detector.get()))
            onnx->loadLabels(labelsPath);
        else if (auto* trt = dynamic_cast<TensorRTDetector*>(m_detector.get())) {
            // For TRT read the labels ourselves
            QFile f(labelsPath);
            if (f.open(QIODevice::ReadOnly | QIODevice::Text)) {
                QStringList labels;
                QTextStream in(&f);
                while (!in.atEnd()) labels << in.readLine().trimmed();
                trt->setLabels(labels);
            }
        }
    }

    if (!m_detector->loadModel(modelPath)) {
        qWarning() << "[DetectorManager] Model load failed";
        return false;
    }

    // Spin up worker thread
    m_worker = new DetectorWorker(m_detector.get());
    m_thread = new QThread(this);
    m_worker->moveToThread(m_thread);

    connect(this,     &DetectorManager::_runDetection,
            m_worker, &DetectorWorker::processFrame, Qt::QueuedConnection);
    connect(m_worker, &DetectorWorker::resultsReady,
            this,     &DetectorManager::detectionReady, Qt::QueuedConnection);

    m_thread->start();
    m_ready = true;
    qInfo() << "[DetectorManager] Ready. Backend:" << backendName();
    return true;
}

void DetectorManager::submitFrame(const QImage& frame) {
    if (m_ready) emit _runDetection(frame);
}

QString DetectorManager::backendName() const {
    return m_detector ? m_detector->backendName() : QStringLiteral("None");
}
