#include <QApplication>
#include <QScreen>
#include "MainWindow_Standard.h"

int main(int argc, char* argv[]) {
    QApplication app(argc, argv);
    app.setApplicationName(QStringLiteral("HospitalDashboard"));
    app.setApplicationVersion(QStringLiteral("1.0.0"));
    app.setOrganizationName(QStringLiteral("Hospital"));

    // High-DPI
    app.setHighDpiScaleFactorRoundingPolicy(
        Qt::HighDpiScaleFactorRoundingPolicy::PassThrough);

    MainWindow w;
    // Start maximised to fill the monitor
    w.showMaximized();
    return app.exec();
}