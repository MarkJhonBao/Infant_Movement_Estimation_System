#pragma once
#include <QPen>
#include <QColor>
#include <QFont>

namespace Theme {

// ── Background colours ────────────────────────────────────────────────────────
inline constexpr QColor BgDeep   {0x04, 0x0d, 0x2c};   // #040d2c
inline constexpr QColor BgPanel  {0x06, 0x18, 0x4a};   // #06184a
inline constexpr QColor BgCard   {0x09, 0x24, 0x66};   // #092466
inline constexpr QColor BgHighlight{0x0c, 0x30, 0x80}; // #0c3080

// ── Accent colours ────────────────────────────────────────────────────────────
inline constexpr QColor Cyan       {0x00, 0xe5, 0xff};  // #00e5ff
inline constexpr QColor CyanDim    {0x00, 0x8b, 0xb2};
inline constexpr QColor Blue       {0x1a, 0x78, 0xe4};
inline constexpr QColor BlueLight  {0x4d, 0xb8, 0xff};
inline constexpr QColor Orange     {0xff, 0x6e, 0x1e};
inline constexpr QColor OrangeLight{0xff, 0xa0, 0x5a};
inline constexpr QColor Green      {0x00, 0xff, 0xa0};
inline constexpr QColor Red        {0xff, 0x40, 0x40};
inline constexpr QColor Yellow     {0xff, 0xe0, 0x40};
inline constexpr QColor Purple     {0x8a, 0x2b, 0xe2};

// ── Chart palette ─────────────────────────────────────────────────────────────
static const QColor ChartColors[] = {
    {0x00, 0xe5, 0xff}, // cyan
    {0xff, 0x6e, 0x1e}, // orange
    {0x7b, 0x68, 0xee}, // purple
    {0x00, 0xff, 0xa0}, // green
    {0xff, 0xe0, 0x40}, // yellow
    {0xff, 0x40, 0x40}, // red
};
static const int ChartColorCount = 6;

// ── Text colours ─────────────────────────────────────────────────────────────
inline constexpr QColor TextPrimary  {0xee, 0xf5, 0xff};
inline constexpr QColor TextSecondary{0x88, 0xaa, 0xcc};
inline constexpr QColor TextMuted    {0x44, 0x66, 0x88};

// ── Borders ───────────────────────────────────────────────────────────────────
inline constexpr QColor BorderGlow  {0x00, 0xe5, 0xff, 0x80};
inline constexpr QColor BorderDim   {0x1a, 0x3a, 0x6a};

// ── Border-glow pen helper ────────────────────────────────────────────────────
inline QPen glowPen(const QColor& c = Cyan, int width = 1) {
    return QPen(c, width);
}

// ── Fonts ─────────────────────────────────────────────────────────────────────
inline QFont titleFont(int pt = 24) {
    QFont f("Microsoft YaHei", pt, QFont::Bold);
    f.setLetterSpacing(QFont::AbsoluteSpacing, 2);
    return f;
}
inline QFont sectionFont(int pt = 11) {
    return QFont("Microsoft YaHei", pt, QFont::DemiBold);
}
inline QFont bodyFont(int pt = 10) {
    return QFont("Microsoft YaHei", pt);
}
inline QFont numFont(int pt = 28) {
    QFont f("Arial", pt, QFont::Bold);
    return f;
}

} // namespace Theme
