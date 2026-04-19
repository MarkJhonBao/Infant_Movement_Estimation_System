#pragma once
// MainWindow.h — 转发包含头文件
//
// 原始版本在此处独立定义了 RealtimeStats 与 MainWindow 两个类，
// 与 MainWindow_Standard.h 中的同名类形成 ODR（One Definition Rule）违规：
//
//   问题：两份头文件均包含 Q_OBJECT，moc 会分别生成
//         moc_MainWindow.cpp 和 moc_MainWindow_Standard.cpp，
//         两者都导出 MainWindow::staticMetaObject / qt_metacall 等符号，
//         链接阶段产生"重复符号"错误。
//
// 修复：将此文件改为纯转发，仅包含 MainWindow_Standard.h，
//       所有使用方（包括 detector/MainWindow_example.h）均引用唯一定义。
//
#include "MainWindow_Standard.h"
