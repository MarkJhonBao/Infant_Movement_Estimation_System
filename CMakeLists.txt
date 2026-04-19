cmake_minimum_required(VERSION 3.18)

# ── Qt路径（必须在 find_package 之前）──────────────────────────────────────
set(Qt6_DIR "E:/Qt/6.5.3/msvc2019_64/lib/cmake/Qt6" CACHE PATH "Qt6 Config")

project(HospitalDashboard VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# ── Qt ──────────────────────────────────────────────────────────────────────
find_package(Qt6 REQUIRED COMPONENTS
    Core Widgets Gui Multimedia MultimediaWidgets Charts Sql)

# ── CUDA（新式写法）─────────────────────────────────────────────────────────
find_package(CUDAToolkit REQUIRED)

# ── ONNX Runtime ────────────────────────────────────────────────────────────
set(ONNXRUNTIME_DIR "" CACHE PATH "ONNX Runtime install directory")

if(ONNXRUNTIME_DIR)
    set(ONNXRUNTIME_INCLUDE_DIRS ${ONNXRUNTIME_DIR}/include)
    find_library(ONNXRUNTIME_LIB
        NAMES onnxruntime
        PATHS ${ONNXRUNTIME_DIR}/lib
        NO_DEFAULT_PATH)
    if(ONNXRUNTIME_LIB)
        message(STATUS "Found ONNX Runtime: ${ONNXRUNTIME_LIB}")
        add_compile_definitions(HAVE_ONNXRUNTIME)
    else()
        message(WARNING "ONNX Runtime library not found in ${ONNXRUNTIME_DIR}/lib")
    endif()
else()
    message(STATUS "ONNXRUNTIME_DIR not set — ONNX Runtime disabled (stub mode)")
endif()

# ── TensorRT 10.x ────────────────────────────────────────────────────────────
set(TENSORRT_DIR "" CACHE PATH "TensorRT install directory")

if(TENSORRT_DIR)
    set(TENSORRT_INCLUDE_DIRS ${TENSORRT_DIR}/include)

    # TensorRT 10.x 的 lib 文件名带 _10 后缀
    find_library(TENSORRT_LIB
        NAMES nvinfer_10 nvinfer        # 先找 10.x，找不到回退旧版
        PATHS ${TENSORRT_DIR}/lib
        NO_DEFAULT_PATH)

    find_library(TENSORRT_ONNX
        NAMES nvonnxparser_10 nvonnxparser
        PATHS ${TENSORRT_DIR}/lib
        NO_DEFAULT_PATH)

    find_library(TENSORRT_PLUGIN
        NAMES nvinfer_plugin_10 nvinfer_plugin
        PATHS ${TENSORRT_DIR}/lib
        NO_DEFAULT_PATH)

    if(TENSORRT_LIB)
        message(STATUS "Found TensorRT: ${TENSORRT_LIB}")
        message(STATUS "Found nvonnxparser: ${TENSORRT_ONNX}")
        add_compile_definitions(HAVE_TENSORRT)
    else()
        message(WARNING "TensorRT library not found in ${TENSORRT_DIR}/lib")
        message(WARNING "请检查目录下实际的 .lib 文件名")
    endif()
else()
    message(STATUS "TENSORRT_DIR not set — TensorRT disabled (stub mode)")
endif()

# ── Sources ──────────────────────────────────────────────────────────────────
set(SOURCES
    main.cpp
    MainWindow.cpp
    widgets/TitleBarWidget.cpp
    widgets/PieChartWidget.cpp
    widgets/GaugeWidget.cpp
    widgets/AlertWidget.cpp
    widgets/DetectionDisplayWidget.cpp
    widgets/KpiWidget.cpp
    widgets/TopListWidget.cpp
    widgets/BarChartWidget.cpp
    widgets/DonutChartWidget.cpp
    widgets/DashboardWidget.cpp
    widgets/LimbMotionWidget.cpp
    detector/OnnxDetector.cpp
    detector/TensorRTDetector.cpp
    detector/DetectorManager.cpp
    detector/PoseDetector.cpp
    detector/MotionAnalyzer.cpp
    database/PoseDatabase.cpp
)

set(HEADERS
    MainWindow_Standard.h
    widgets/TitleBarWidget.h
    widgets/PieChartWidget.h
    widgets/GaugeWidget.h
    widgets/AlertWidget.h
    widgets/DetectionDisplayWidget.h
    widgets/KpiWidget.h
    widgets/TopListWidget.h
    widgets/BarChartWidget.h
    widgets/DonutChartWidget.h
    widgets/DashboardWidget.h
    widgets/LimbMotionWidget.h
    detector/DetectorBase.h
    detector/OnnxDetector.h
    detector/TensorRTDetector.h
    detector/DetectorManager.h
    detector/PoseDetector.h
    detector/PoseDetectorManager.h
    detector/PoseResult.h
    detector/PoseOverlayWidget.h
    detector/MotionAnalyzer.h
    database/PoseDatabase.h
)

set(RESOURCES resources/resources.qrc)

qt_add_executable(HospitalDashboard
    ${SOURCES}
    ${HEADERS}
    ${RESOURCES}
)

# ── Include 目录 ──────────────────────────────────────────────────────────────
target_include_directories(HospitalDashboard PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/widgets
    ${CMAKE_CURRENT_SOURCE_DIR}/detector
    ${CUDAToolkit_INCLUDE_DIRS}           # CUDA 头文件
)

# ── 基础链接库 ────────────────────────────────────────────────────────────────
target_link_libraries(HospitalDashboard PRIVATE
    Qt6::Core
    Qt6::Widgets
    Qt6::Gui
    Qt6::Multimedia
    Qt6::MultimediaWidgets
    Qt6::Charts
    Qt6::Sql
    CUDA::cudart                          # 新式 CUDA target
)

# ── 可选：ONNX Runtime ────────────────────────────────────────────────────────
if(ONNXRUNTIME_LIB)
    target_include_directories(HospitalDashboard PRIVATE ${ONNXRUNTIME_INCLUDE_DIRS})
    target_link_libraries(HospitalDashboard PRIVATE ${ONNXRUNTIME_LIB})
endif()

# ── 可选：TensorRT ────────────────────────────────────────────────────────────
if(TENSORRT_LIB)
    target_include_directories(HospitalDashboard PRIVATE
        ${TENSORRT_INCLUDE_DIRS})
    target_link_libraries(HospitalDashboard PRIVATE
        ${TENSORRT_LIB}
        $<$<BOOL:${TENSORRT_ONNX}>:${TENSORRT_ONNX}>
        $<$<BOOL:${TENSORRT_PLUGIN}>:${TENSORRT_PLUGIN}>
    )
endif()

# ── 安装 ─────────────────────────────────────────────────────────────────────
install(TARGETS HospitalDashboard
    BUNDLE  DESTINATION .
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
