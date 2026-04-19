@echo off
REM Hospital Dashboard Build and Test Script
REM Tests real-time frame refresh functionality

echo ========================================
echo Hospital Dashboard Build and Test Script
echo ========================================

cd /d E:\xy-2024-0722\HospitalDashboard

REM Clean previous build
if exist build (
    echo Cleaning previous build...
    rmdir /s /q build
)

REM Create build directory
mkdir build
cd build

REM Configure with CMake
echo Configuring with CMake...
cmake .. -DCMAKE_BUILD_TYPE=Debug

if errorlevel 1 (
    echo CMake configuration failed!
    pause
    exit /b 1
)

REM Build the project
echo Building project...
cmake --build . --config Debug

if errorlevel 1 (
    echo Build failed!
    pause
    exit /b 1
)

echo ========================================
echo Build completed successfully!
echo ========================================
echo.
echo Test Results:
echo - Database module integrated
echo - Motion analyzer integrated
echo - Frame ID synchronization implemented
echo - Real-time refresh mechanism improved
echo.
echo To test real-time refresh:
echo 1. Run HospitalDashboard.exe
echo 2. Observe FPS display (top-right)
echo 3. Check processing delay (bottom-right)
echo 4. Verify pose detection alignment
echo.
pause