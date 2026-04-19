// ─────────────────────────────────────────────────────────────────────────────
//  MainWindow.h  —  Integration example
//
//  Pipeline (top-down multi-person mode):
//
//    Camera/Video → DetectorManager (object detect, async)
//                       │  detectionReady(boxes)
//                       ▼
//                  PoseDetectorManager (pose estimate, async)
//                       │  poseReady(poses)
//                       ▼
//                  PoseOverlayWidget (render)
//
//  Both inference stages run on dedicated QThreads so the UI never blocks.
// ─────────────────────────────────────────────────────────────────────────────
#pragma once
#include <QMainWindow>
#include <QTimer>
#include <QCamera>
#include <QVideoProbe>   // Qt5; use QMediaCaptureSession in Qt6

#include "DetectorManager.h"
#include "PoseDetectorManager.h"
#include "PoseOverlayWidget.h"

class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    explicit MainWindow(QWidget* parent = nullptr) : QMainWindow(parent) {
        setWindowTitle("姿态估计 — TensorRT + mmpose");
        resize(1280, 720);

        // ── Overlay widget (central) ─────────────────────────────────────────
        m_overlay = new PoseOverlayWidget(this);
        setCentralWidget(m_overlay);

        // ── Object detector (person bounding boxes) ──────────────────────────
        m_detMgr = new DetectorManager(this);
        m_detMgr->setBackend(DetectorManager::Backend::TensorRT);

        // ── Pose detector ────────────────────────────────────────────────────
        m_poseMgr = new PoseDetectorManager(this);
        m_poseMgr->setInputSize(192, 256);   // RTMPose-s default
        m_poseMgr->setFP16(true);

        // ── Wire signals ─────────────────────────────────────────────────────
        //
        // When the object detector emits bounding boxes, forward the latest
        // frame + boxes to the pose estimator.
        connect(m_detMgr, &DetectorManager::detectionReady,
                this,     &MainWindow::onDetections);

        // When pose estimator is done, push results to the overlay widget.
        connect(m_poseMgr, &PoseDetectorManager::poseReady,
                this, [this](const QVector<PoseResult>& poses) {
                    m_overlay->updateFrame(m_latestFrame, poses, m_latestDets);
                });
    }

    // Call from main() after construction
    bool loadModels(const QString& detOnnx,   const QString& detLabels,
                    const QString& poseOnnx,  const QString& poseEngineCache = {})
    {
        if (!m_detMgr->init(detOnnx, detLabels)) return false;
        m_overlay->setBackendName(m_detMgr->backendName()
                                  + " + " + m_poseMgr->backendName());

        if (!m_poseMgr->init(poseOnnx, poseEngineCache)) return false;
        m_overlay->setBackendName(m_detMgr->backendName()
                                  + " + " + m_poseMgr->backendName());
        return true;
    }

    // Feed a new frame (call from camera callback / video reader loop)
    void pushFrame(const QImage& frame) {
        m_latestFrame = frame;
        m_detMgr->submitFrame(frame);   // triggers object detection async
    }

private slots:
    void onDetections(const QVector<DetectionResult>& dets) {
        // Filter to person class only (class 0 in COCO)
        QVector<DetectionResult> persons;
        for (const auto& d : dets)
            if (d.classId == 0) persons.push_back(d);

        m_latestDets = persons;

        if (persons.isEmpty()) {
            // No persons — clear overlay
            m_overlay->updateFrame(m_latestFrame, {}, dets);
        } else {
            // Run pose estimation for each detected person
            m_poseMgr->submitFrameWithBoxes(m_latestFrame, persons);
        }
    }

private:
    PoseOverlayWidget*    m_overlay{nullptr};
    DetectorManager*      m_detMgr{nullptr};
    PoseDetectorManager*  m_poseMgr{nullptr};

    QImage                    m_latestFrame;
    QVector<DetectionResult>  m_latestDets;
};


// ─────────────────────────────────────────────────────────────────────────────
//  main.cpp  —  Minimal entry point
// ─────────────────────────────────────────────────────────────────────────────
/*
#include <QApplication>
#include "MainWindow.h"

int main(int argc, char* argv[]) {
    QApplication app(argc, argv);

    MainWindow w;
    if (!w.loadModels(
            "yolov8n.onnx",          // object detector
            "coco_labels.txt",       // COCO class names
            "rtmpose-s_8xb256-420e_coco-256x192.onnx",  // mmpose RTMPose
            "rtmpose.engine"))       // engine cache (written on first run)
    {
        qCritical() << "Model load failed — check paths.";
        return 1;
    }

    w.show();

    // ── Simulate 30 fps with a test image (replace with real camera feed) ───
    QImage testImg(1280, 720, QImage::Format_RGB888);
    testImg.fill(Qt::darkGray);
    QTimer ticker;
    QObject::connect(&ticker, &QTimer::timeout, [&]() {
        w.pushFrame(testImg);
    });
    ticker.start(33);

    return app.exec();
}
*/
