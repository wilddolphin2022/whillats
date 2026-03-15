/*
 *  (c) 2025, wilddolphin2022
 *  For WebRTCsays.ai project
 *  https://github.com/wilddolphin2022
 *
 *  ASL sign language face implementation.
 */

#include "sign_language_face.h"
#include <cmath>
#include <algorithm>
#include <cstring>
#include <cctype>
#include <sstream>
#include <dirent.h>

#include "stb_image.h"

SignLanguageFace::SignLanguageFace() = default;
SignLanguageFace::~SignLanguageFace() = default;

bool SignLanguageFace::loadSignImages(const char* directory) {
    DIR* dir = opendir(directory);
    if (!dir) return false;

    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string name = entry->d_name;
        auto dot = name.rfind('.');
        if (dot == std::string::npos) continue;

        std::string ext = name.substr(dot + 1);
        for (auto& c : ext) c = std::tolower(c);
        if (ext != "png" && ext != "jpg" && ext != "jpeg" && ext != "bmp") continue;

        std::string key = name.substr(0, dot);
        for (auto& c : key) c = std::tolower(c);

        std::string path = std::string(directory) + "/" + name;
        int w, h, ch;
        uint8_t* pixels = stbi_load(path.c_str(), &w, &h, &ch, 3);
        if (!pixels) continue;

        SignImage img;
        img.width = w;
        img.height = h;
        img.rgb.resize(w * h * 3);
        std::memcpy(img.rgb.data(), pixels, w * h * 3);
        stbi_image_free(pixels);

        signs_[key] = std::move(img);
    }
    closedir(dir);
    return !signs_.empty();
}

bool SignLanguageFace::addSign(const std::string& key, const uint8_t* data, int size) {
    int w, h, ch;
    uint8_t* pixels = stbi_load_from_memory(data, size, &w, &h, &ch, 3);
    if (!pixels) return false;

    SignImage img;
    img.width = w;
    img.height = h;
    img.rgb.resize(w * h * 3);
    std::memcpy(img.rgb.data(), pixels, w * h * 3);
    stbi_image_free(pixels);

    signs_[key] = std::move(img);
    return true;
}

// 5x7 bitmap font for rendering labels on signs
static const uint8_t kFont5x7[][7] = {
    // ' ' (space)
    {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},
    // A-Z (indices 1-26)
    {0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11}, // A
    {0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E}, // B
    {0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E}, // C
    {0x1C, 0x12, 0x11, 0x11, 0x11, 0x12, 0x1C}, // D
    {0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F}, // E
    {0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10}, // F
    {0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0F}, // G
    {0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11}, // H
    {0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E}, // I
    {0x07, 0x02, 0x02, 0x02, 0x02, 0x12, 0x0C}, // J
    {0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11}, // K
    {0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F}, // L
    {0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11}, // M
    {0x11, 0x19, 0x15, 0x13, 0x11, 0x11, 0x11}, // N
    {0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E}, // O
    {0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10}, // P
    {0x0E, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0D}, // Q
    {0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11}, // R
    {0x0E, 0x11, 0x10, 0x0E, 0x01, 0x11, 0x0E}, // S
    {0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04}, // T
    {0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E}, // U
    {0x11, 0x11, 0x11, 0x11, 0x11, 0x0A, 0x04}, // V
    {0x11, 0x11, 0x11, 0x15, 0x15, 0x1B, 0x11}, // W
    {0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11}, // X
    {0x11, 0x11, 0x0A, 0x04, 0x04, 0x04, 0x04}, // Y
    {0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F}, // Z
};

static int fontIndex(char ch) {
    if (ch >= 'A' && ch <= 'Z') return ch - 'A' + 1;
    if (ch >= 'a' && ch <= 'z') return ch - 'a' + 1;
    return 0; // space
}

void SignLanguageFace::drawChar(std::vector<uint8_t>& rgb, int imgW, int imgH,
                                char ch, int x, int y, int size,
                                uint8_t r, uint8_t g, uint8_t b) {
    int idx = fontIndex(ch);
    int scale = size / 7;
    if (scale < 1) scale = 1;

    for (int row = 0; row < 7; row++) {
        uint8_t bits = kFont5x7[idx][row];
        for (int col = 0; col < 5; col++) {
            if (bits & (0x10 >> col)) {
                for (int sy = 0; sy < scale; sy++) {
                    for (int sx = 0; sx < scale; sx++) {
                        int px = x + col * scale + sx;
                        int py = y + row * scale + sy;
                        if (px >= 0 && px < imgW && py >= 0 && py < imgH) {
                            int pi = (py * imgW + px) * 3;
                            rgb[pi] = r; rgb[pi+1] = g; rgb[pi+2] = b;
                        }
                    }
                }
            }
        }
    }
}

void SignLanguageFace::renderText(std::vector<uint8_t>& rgb, int imgW, int imgH,
                                  const std::string& text, int cx, int cy, int fontSize,
                                  uint8_t r, uint8_t g, uint8_t b) {
    int scale = fontSize / 7;
    if (scale < 1) scale = 1;
    int charW = 6 * scale;
    int totalW = (int)text.size() * charW;
    int startX = cx - totalW / 2;
    int startY = cy - fontSize / 2;

    for (size_t i = 0; i < text.size(); i++) {
        drawChar(rgb, imgW, imgH, text[i], startX + (int)i * charW, startY, fontSize, r, g, b);
    }
}

void SignLanguageFace::drawFilledRect(std::vector<uint8_t>& rgb, int imgW, int imgH,
                                       int x, int y, int w, int h,
                                       uint8_t r, uint8_t g, uint8_t b) {
    for (int py = std::max(0, y); py < std::min(imgH, y + h); py++) {
        for (int px = std::max(0, x); px < std::min(imgW, x + w); px++) {
            int idx = (py * imgW + px) * 3;
            rgb[idx] = r; rgb[idx+1] = g; rgb[idx+2] = b;
        }
    }
}

void SignLanguageFace::drawRoundedRect(std::vector<uint8_t>& rgb, int imgW, int imgH,
                                        int cx, int cy, int w, int h, int radius,
                                        uint8_t r, uint8_t g, uint8_t b) {
    int x0 = cx - w / 2, y0 = cy - h / 2;
    int x1 = x0 + w, y1 = y0 + h;
    int rad = std::min(radius, std::min(w / 2, h / 2));

    for (int py = std::max(0, y0); py < std::min(imgH, y1); py++) {
        for (int px = std::max(0, x0); px < std::min(imgW, x1); px++) {
            bool inside = true;
            int dx = 0, dy = 0;
            if (px < x0 + rad && py < y0 + rad) { dx = px - (x0 + rad); dy = py - (y0 + rad); }
            else if (px >= x1 - rad && py < y0 + rad) { dx = px - (x1 - rad - 1); dy = py - (y0 + rad); }
            else if (px < x0 + rad && py >= y1 - rad) { dx = px - (x0 + rad); dy = py - (y1 - rad - 1); }
            else if (px >= x1 - rad && py >= y1 - rad) { dx = px - (x1 - rad - 1); dy = py - (y1 - rad - 1); }
            if (dx != 0 || dy != 0) inside = (dx * dx + dy * dy) <= (rad * rad);
            if (inside) {
                int idx = (py * imgW + px) * 3;
                rgb[idx] = r; rgb[idx+1] = g; rgb[idx+2] = b;
            }
        }
    }
}

void SignLanguageFace::drawCircle(std::vector<uint8_t>& rgb, int imgW, int imgH,
                                   int cx, int cy, int radius,
                                   uint8_t r, uint8_t g, uint8_t b) {
    for (int py = std::max(0, cy - radius); py <= std::min(imgH - 1, cy + radius); py++) {
        for (int px = std::max(0, cx - radius); px <= std::min(imgW - 1, cx + radius); px++) {
            int dx = px - cx, dy = py - cy;
            if (dx * dx + dy * dy <= radius * radius) {
                int idx = (py * imgW + px) * 3;
                rgb[idx] = r; rgb[idx+1] = g; rgb[idx+2] = b;
            }
        }
    }
}

void SignLanguageFace::drawLine(std::vector<uint8_t>& rgb, int imgW, int imgH,
                                 int x0, int y0, int x1, int y1, int thickness,
                                 uint8_t r, uint8_t g, uint8_t b) {
    int dx = std::abs(x1 - x0), dy = std::abs(y1 - y0);
    int steps = std::max(dx, dy);
    if (steps == 0) { drawCircle(rgb, imgW, imgH, x0, y0, thickness / 2, r, g, b); return; }
    for (int i = 0; i <= steps; i++) {
        int px = x0 + (x1 - x0) * i / steps;
        int py = y0 + (y1 - y0) * i / steps;
        drawCircle(rgb, imgW, imgH, px, py, thickness / 2, r, g, b);
    }
}

// Generates a stylized hand shape for a given ASL fingerspelling letter.
// Each letter gets a distinct hand pose rendered programmatically.
void SignLanguageFace::generateLetterSign(char letter, int size) {
    SignImage img;
    img.width = size;
    img.height = size;
    img.rgb.resize(size * size * 3);

    // Warm cream background
    std::fill(img.rgb.begin(), img.rgb.end(), 0);
    for (int i = 0; i < size * size; i++) {
        img.rgb[i * 3]     = 0xF5;
        img.rgb[i * 3 + 1] = 0xF0;
        img.rgb[i * 3 + 2] = 0xEB;
    }

    int cx = size / 2;
    int cy = size * 45 / 100;
    int palmW = size * 22 / 100;
    int palmH = size * 28 / 100;

    // Skin tone colors
    uint8_t skinR = 0xE8, skinG = 0xBE, skinB = 0x96;
    uint8_t darkR = 0xC4, darkG = 0x9A, darkB = 0x6C;

    // Palm
    drawRoundedRect(img.rgb, size, size, cx, cy, palmW, palmH, palmW / 4, skinR, skinG, skinB);

    int fingerW = size * 4 / 100;
    int fingerH = size * 14 / 100;
    int fingerSpacing = size * 6 / 100;
    int fingerBase = cy - palmH / 2;

    char upper = std::toupper(letter);

    // Distinct finger configurations per letter group
    // Fist letters (A, E, M, N, S, T): closed hand
    // Point letters (D, G, I, L, Z): index extended
    // Open letters (B, C, F, O, W): multiple fingers
    // Special (J, K, P, Q, R, U, V, X, Y): unique poses

    bool fist = (upper == 'A' || upper == 'E' || upper == 'M' || upper == 'N' || upper == 'S' || upper == 'T');
    bool point = (upper == 'D' || upper == 'G' || upper == 'I' || upper == 'Z');
    bool openHand = (upper == 'B' || upper == 'C' || upper == 'F' || upper == 'W');
    bool twoFingers = (upper == 'H' || upper == 'K' || upper == 'R' || upper == 'U' || upper == 'V');

    if (fist) {
        // Closed fist — thumb position varies
        drawRoundedRect(img.rgb, size, size, cx, cy - palmH / 6, palmW + 4, palmH * 2 / 3, palmW / 5, skinR, skinG, skinB);
        if (upper == 'A' || upper == 'S') {
            // Thumb across front
            drawLine(img.rgb, size, size, cx - palmW / 2, cy - palmH / 4, cx - palmW / 2 - fingerW * 2, cy - palmH / 2, fingerW, skinR, skinG, skinB);
        }
    } else if (point) {
        // Index finger extended upward
        int indexX = cx;
        drawLine(img.rgb, size, size, indexX, fingerBase, indexX, fingerBase - fingerH, fingerW, skinR, skinG, skinB);
        drawCircle(img.rgb, size, size, indexX, fingerBase - fingerH, fingerW / 2 + 1, skinR, skinG, skinB);
        if (upper == 'L') {
            // L-shape: thumb out to the side
            drawLine(img.rgb, size, size, cx - palmW / 2, cy, cx - palmW / 2 - fingerH * 2 / 3, cy, fingerW, skinR, skinG, skinB);
        }
    } else if (openHand) {
        // All fingers extended
        for (int f = 0; f < 4; f++) {
            int fx = cx - fingerSpacing * 3 / 2 + f * fingerSpacing;
            int fh = (f == 1 || f == 2) ? fingerH : fingerH * 85 / 100;
            drawLine(img.rgb, size, size, fx, fingerBase, fx, fingerBase - fh, fingerW, skinR, skinG, skinB);
            drawCircle(img.rgb, size, size, fx, fingerBase - fh, fingerW / 2 + 1, skinR, skinG, skinB);
        }
        if (upper == 'C') {
            // C-shape: fingers curved
            drawRoundedRect(img.rgb, size, size, cx, fingerBase - fingerH + 10, palmW, 8, 4, darkR, darkG, darkB);
        }
    } else if (twoFingers) {
        // Two fingers extended (index + middle)
        int f1x = cx - fingerSpacing / 2;
        int f2x = cx + fingerSpacing / 2;
        int spread = (upper == 'V' || upper == 'K') ? fingerSpacing / 3 : 0;
        drawLine(img.rgb, size, size, f1x, fingerBase, f1x - spread, fingerBase - fingerH, fingerW, skinR, skinG, skinB);
        drawLine(img.rgb, size, size, f2x, fingerBase, f2x + spread, fingerBase - fingerH, fingerW, skinR, skinG, skinB);
        drawCircle(img.rgb, size, size, f1x - spread, fingerBase - fingerH, fingerW / 2 + 1, skinR, skinG, skinB);
        drawCircle(img.rgb, size, size, f2x + spread, fingerBase - fingerH, fingerW / 2 + 1, skinR, skinG, skinB);
    } else if (upper == 'O') {
        // O-shape: thumb and index form circle
        drawCircle(img.rgb, size, size, cx, cy - palmH / 3, palmW / 3, darkR, darkG, darkB);
        drawCircle(img.rgb, size, size, cx, cy - palmH / 3, palmW / 3 - fingerW, skinR, skinG, skinB);
    } else if (upper == 'Y') {
        // Y: thumb and pinky extended
        drawLine(img.rgb, size, size, cx - palmW / 2, cy - palmH / 4, cx - palmW / 2 - fingerH / 2, cy - palmH / 2 - fingerH / 2, fingerW, skinR, skinG, skinB);
        drawLine(img.rgb, size, size, cx + palmW / 2, cy - palmH / 4, cx + palmW / 2 + fingerH / 2, cy - palmH / 2 - fingerH / 2, fingerW, skinR, skinG, skinB);
    } else if (upper == 'J') {
        // J: like I but with motion arc (draw pinky + arc)
        int pinkyX = cx + fingerSpacing * 3 / 2;
        drawLine(img.rgb, size, size, pinkyX, fingerBase, pinkyX, fingerBase - fingerH, fingerW, skinR, skinG, skinB);
        // Motion arc indicator
        for (int a = 0; a < 8; a++) {
            float angle = 3.14159f * a / 7.0f;
            int ax = pinkyX + (int)(fingerH / 3 * std::cos(angle));
            int ay = fingerBase - fingerH + (int)(fingerH / 3 * std::sin(angle));
            drawCircle(img.rgb, size, size, ax, ay, 2, darkR, darkG, darkB);
        }
    } else if (upper == 'X') {
        // X: index finger hooked
        int indexX = cx;
        drawLine(img.rgb, size, size, indexX, fingerBase, indexX, fingerBase - fingerH * 2 / 3, fingerW, skinR, skinG, skinB);
        drawLine(img.rgb, size, size, indexX, fingerBase - fingerH * 2 / 3, indexX + fingerW * 2, fingerBase - fingerH * 2 / 3 - fingerW * 2, fingerW, skinR, skinG, skinB);
    } else if (upper == 'Q' || upper == 'P') {
        // Pointing downward
        int indexX = cx;
        drawLine(img.rgb, size, size, indexX, cy + palmH / 2, indexX, cy + palmH / 2 + fingerH, fingerW, skinR, skinG, skinB);
        drawCircle(img.rgb, size, size, indexX, cy + palmH / 2 + fingerH, fingerW / 2 + 1, skinR, skinG, skinB);
    } else {
        // Default: index finger up
        drawLine(img.rgb, size, size, cx, fingerBase, cx, fingerBase - fingerH, fingerW, skinR, skinG, skinB);
        drawCircle(img.rgb, size, size, cx, fingerBase - fingerH, fingerW / 2 + 1, skinR, skinG, skinB);
    }

    // Wrist
    drawRoundedRect(img.rgb, size, size, cx, cy + palmH / 2 + size * 5 / 100, palmW * 80 / 100, size * 10 / 100, palmW / 6, darkR, darkG, darkB);

    // Letter label at bottom
    std::string label(1, upper);
    renderText(img.rgb, size, size, label, cx, size * 88 / 100, size * 8 / 100, 0x33, 0x33, 0x33);

    // ASL label at top
    renderText(img.rgb, size, size, "ASL", cx, size * 5 / 100, size * 4 / 100, 0x99, 0x99, 0x99);

    std::string key(1, std::tolower(letter));
    signs_[key] = std::move(img);
}

// Generates a sign image for a common ASL word with a distinctive hand/body illustration.
void SignLanguageFace::generateWordSign(const std::string& word, int size) {
    SignImage img;
    img.width = size;
    img.height = size;
    img.rgb.resize(size * size * 3);

    // Gradient background: soft blue top to cream bottom
    for (int y = 0; y < size; y++) {
        float t = (float)y / size;
        uint8_t r = (uint8_t)(0xE8 + t * (0xF5 - 0xE8));
        uint8_t g = (uint8_t)(0xEE + t * (0xF0 - 0xEE));
        uint8_t b = (uint8_t)(0xF8 + t * (0xEB - 0xF8));
        for (int x = 0; x < size; x++) {
            int idx = (y * size + x) * 3;
            img.rgb[idx] = r; img.rgb[idx+1] = g; img.rgb[idx+2] = b;
        }
    }

    int cx = size / 2;
    int cy = size * 40 / 100;
    uint8_t skinR = 0xE8, skinG = 0xBE, skinB = 0x96;
    uint8_t darkR = 0xC4, darkG = 0x9A, darkB = 0x6C;

    if (word == "hello" || word == "hi") {
        // Open palm wave
        int palmW = size * 24 / 100, palmH = size * 20 / 100;
        drawRoundedRect(img.rgb, size, size, cx + size * 8 / 100, cy, palmW, palmH, palmW / 4, skinR, skinG, skinB);
        for (int f = 0; f < 5; f++) {
            int fx = cx - size * 4 / 100 + f * size * 6 / 100;
            int fh = size * 12 / 100;
            drawLine(img.rgb, size, size, fx, cy - palmH / 2, fx - size * 2 / 100, cy - palmH / 2 - fh, size * 3 / 100, skinR, skinG, skinB);
        }
        // Motion lines
        for (int i = 0; i < 3; i++) {
            int lx = cx + size * 25 / 100 + i * size * 4 / 100;
            drawLine(img.rgb, size, size, lx, cy - size * 8 / 100, lx + size * 3 / 100, cy + size * 4 / 100, 2, 0x99, 0xBB, 0xDD);
        }
    } else if (word == "thank" || word == "thanks") {
        // Flat hand from chin forward
        int palmW = size * 22 / 100, palmH = size * 6 / 100;
        drawRoundedRect(img.rgb, size, size, cx, cy + size * 5 / 100, palmW, palmH, 4, skinR, skinG, skinB);
        for (int f = 0; f < 4; f++) {
            int fx = cx - size * 8 / 100 + f * size * 5 / 100;
            drawLine(img.rgb, size, size, fx, cy + size * 2 / 100, fx, cy - size * 2 / 100, size * 3 / 100, skinR, skinG, skinB);
        }
        // Chin reference
        drawCircle(img.rgb, size, size, cx, cy + size * 15 / 100, size * 3 / 100, darkR, darkG, darkB);
        // Arrow forward
        drawLine(img.rgb, size, size, cx, cy + size * 8 / 100, cx + size * 15 / 100, cy - size * 2 / 100, 3, 0x66, 0x88, 0xAA);
    } else if (word == "you") {
        // Point forward
        int palmW = size * 15 / 100, palmH = size * 20 / 100;
        drawRoundedRect(img.rgb, size, size, cx - size * 8 / 100, cy, palmW, palmH, palmW / 5, skinR, skinG, skinB);
        drawLine(img.rgb, size, size, cx, cy - palmH / 3, cx + size * 20 / 100, cy - palmH / 3, size * 4 / 100, skinR, skinG, skinB);
        drawCircle(img.rgb, size, size, cx + size * 20 / 100, cy - palmH / 3, size * 3 / 100, skinR, skinG, skinB);
    } else if (word == "how") {
        // Fists together, roll outward
        int fistS = size * 12 / 100;
        drawRoundedRect(img.rgb, size, size, cx - size * 8 / 100, cy, fistS, fistS, fistS / 4, skinR, skinG, skinB);
        drawRoundedRect(img.rgb, size, size, cx + size * 8 / 100, cy, fistS, fistS, fistS / 4, skinR, skinG, skinB);
        // Roll arrows
        for (int i = 0; i < 5; i++) {
            float a = 3.14159f * i / 4.0f - 1.57f;
            drawCircle(img.rgb, size, size,
                       cx - size * 8 / 100 + (int)(size * 10 / 100 * std::cos(a)),
                       cy + (int)(size * 10 / 100 * std::sin(a)), 2, 0x66, 0x88, 0xAA);
        }
    } else if (word == "are" || word == "is") {
        // R-hand from lips forward
        int f1x = cx - size * 3 / 100, f2x = cx + size * 3 / 100;
        int base = cy + size * 5 / 100;
        drawLine(img.rgb, size, size, f1x, base, f1x, base - size * 14 / 100, size * 4 / 100, skinR, skinG, skinB);
        drawLine(img.rgb, size, size, f2x, base, f2x, base - size * 14 / 100, size * 4 / 100, skinR, skinG, skinB);
        // Cross the fingers for R
        drawLine(img.rgb, size, size, f1x, base - size * 10 / 100, f2x, base - size * 12 / 100, 2, darkR, darkG, darkB);
    } else if (word == "what") {
        // Open palm, sweep across
        int palmW = size * 22 / 100, palmH = size * 18 / 100;
        drawRoundedRect(img.rgb, size, size, cx, cy, palmW, palmH, palmW / 4, skinR, skinG, skinB);
        for (int f = 0; f < 4; f++) {
            int fx = cx - size * 7 / 100 + f * size * 5 / 100;
            drawLine(img.rgb, size, size, fx, cy - palmH / 2, fx, cy - palmH / 2 - size * 10 / 100, size * 3 / 100, skinR, skinG, skinB);
        }
        // Sweep arrow
        drawLine(img.rgb, size, size, cx - size * 18 / 100, cy + size * 15 / 100, cx + size * 18 / 100, cy + size * 15 / 100, 3, 0x66, 0x88, 0xAA);
    } else if (word == "name") {
        // H-hand tap on H-hand
        int fW = size * 4 / 100;
        drawLine(img.rgb, size, size, cx - size * 5 / 100, cy - size * 8 / 100, cx - size * 5 / 100, cy + size * 8 / 100, fW, skinR, skinG, skinB);
        drawLine(img.rgb, size, size, cx + size * 5 / 100, cy - size * 8 / 100, cx + size * 5 / 100, cy + size * 8 / 100, fW, skinR, skinG, skinB);
        // Tap indicator
        drawLine(img.rgb, size, size, cx - size * 12 / 100, cy, cx + size * 12 / 100, cy, fW, darkR, darkG, darkB);
    } else if (word == "test") {
        // X-hands, pull down
        int fW = size * 4 / 100;
        drawLine(img.rgb, size, size, cx - size * 6 / 100, cy - size * 10 / 100, cx - size * 6 / 100, cy + size * 5 / 100, fW, skinR, skinG, skinB);
        drawLine(img.rgb, size, size, cx + size * 6 / 100, cy - size * 10 / 100, cx + size * 6 / 100, cy + size * 5 / 100, fW, skinR, skinG, skinB);
        drawLine(img.rgb, size, size, cx - size * 6 / 100, cy - size * 5 / 100, cx - size * 3 / 100, cy - size * 8 / 100, fW / 2, skinR, skinG, skinB);
        drawLine(img.rgb, size, size, cx + size * 6 / 100, cy - size * 5 / 100, cx + size * 3 / 100, cy - size * 8 / 100, fW / 2, skinR, skinG, skinB);
        // Down arrows
        drawLine(img.rgb, size, size, cx, cy + size * 10 / 100, cx, cy + size * 18 / 100, 3, 0x66, 0x88, 0xAA);
    } else if (word == "speech" || word == "talk" || word == "say") {
        // 4-hand from mouth forward
        int palmH = size * 6 / 100;
        for (int f = 0; f < 4; f++) {
            int fx = cx - size * 8 / 100 + f * size * 5 / 100;
            drawLine(img.rgb, size, size, fx, cy, fx, cy - size * 12 / 100, size * 3 / 100, skinR, skinG, skinB);
        }
        drawRoundedRect(img.rgb, size, size, cx, cy + palmH / 2, size * 20 / 100, palmH, 4, skinR, skinG, skinB);
        // Mouth reference
        drawCircle(img.rgb, size, size, cx, cy + size * 18 / 100, size * 5 / 100, darkR, darkG, darkB);
        drawLine(img.rgb, size, size, cx, cy + size * 12 / 100, cx + size * 15 / 100, cy - size * 5 / 100, 3, 0x66, 0x88, 0xAA);
    } else if (word == "this" || word == "that") {
        // Point down at palm
        drawRoundedRect(img.rgb, size, size, cx, cy + size * 8 / 100, size * 20 / 100, size * 6 / 100, 4, skinR, skinG, skinB);
        drawLine(img.rgb, size, size, cx, cy - size * 15 / 100, cx, cy + size * 5 / 100, size * 4 / 100, skinR, skinG, skinB);
        drawCircle(img.rgb, size, size, cx, cy + size * 5 / 100, size * 3 / 100, darkR, darkG, darkB);
    } else if (word == "synthesis" || word == "to") {
        // Generic two-hand gesture
        int palmS = size * 12 / 100;
        drawRoundedRect(img.rgb, size, size, cx - size * 12 / 100, cy, palmS, palmS * 3 / 2, palmS / 4, skinR, skinG, skinB);
        drawRoundedRect(img.rgb, size, size, cx + size * 12 / 100, cy, palmS, palmS * 3 / 2, palmS / 4, skinR, skinG, skinB);
    } else {
        // Unknown word fallback: open palm with question
        int palmW = size * 22 / 100, palmH = size * 26 / 100;
        drawRoundedRect(img.rgb, size, size, cx, cy, palmW, palmH, palmW / 4, skinR, skinG, skinB);
        for (int f = 0; f < 4; f++) {
            int fx = cx - size * 8 / 100 + f * size * 5 / 100;
            drawLine(img.rgb, size, size, fx, cy - palmH / 2, fx, cy - palmH / 2 - size * 12 / 100, size * 3 / 100, skinR, skinG, skinB);
        }
    }

    // Word label at bottom
    std::string upper_word = word;
    for (auto& c : upper_word) c = std::toupper(c);
    renderText(img.rgb, size, size, upper_word, cx, size * 82 / 100, size * 6 / 100, 0x33, 0x33, 0x33);

    // "ASL" tag
    renderText(img.rgb, size, size, "ASL", cx, size * 5 / 100, size * 4 / 100, 0x99, 0x99, 0x99);

    signs_[word] = std::move(img);
}

void SignLanguageFace::generateBuiltinSigns(int size) {
    // Fingerspelling alphabet
    for (char c = 'a'; c <= 'z'; c++) {
        generateLetterSign(c, size);
    }

    // Rest/idle pose
    {
        SignImage img;
        img.width = size;
        img.height = size;
        img.rgb.resize(size * size * 3);
        for (int i = 0; i < size * size; i++) {
            img.rgb[i * 3] = 0xF0; img.rgb[i * 3 + 1] = 0xF0; img.rgb[i * 3 + 2] = 0xF0;
        }
        int cx = size / 2, cy = size * 45 / 100;
        // Relaxed hands at sides
        drawRoundedRect(img.rgb, size, size, cx - size * 15 / 100, cy, size * 10 / 100, size * 14 / 100, size * 3 / 100, 0xE8, 0xBE, 0x96);
        drawRoundedRect(img.rgb, size, size, cx + size * 15 / 100, cy, size * 10 / 100, size * 14 / 100, size * 3 / 100, 0xE8, 0xBE, 0x96);
        renderText(img.rgb, size, size, "READY", cx, size * 85 / 100, size * 5 / 100, 0x88, 0x88, 0x88);
        signs_["rest"] = std::move(img);
    }

    // Common ASL words
    const char* common_words[] = {
        "hello", "hi", "thank", "thanks", "you", "how", "are", "is",
        "what", "name", "test", "speech", "talk", "say",
        "this", "that", "synthesis", "to", "of", "a"
    };
    for (const char* w : common_words) {
        generateWordSign(w, size);
    }
}

void SignLanguageFace::setText(const std::string& text, float total_duration_sec) {
    std::lock_guard<std::mutex> lock(render_mutex_);
    total_duration_ = total_duration_sec;
    word_timings_.clear();

    // Tokenize into words
    std::istringstream iss(text);
    std::vector<std::string> words;
    std::string w;
    while (iss >> w) {
        // Normalize to lowercase, strip punctuation
        std::string clean;
        for (char c : w) {
            if (std::isalpha(c)) clean += std::tolower(c);
        }
        if (!clean.empty()) words.push_back(clean);
    }

    if (words.empty()) return;

    // For words without a known sign, expand to fingerspelling.
    // Each fingerspelled letter gets its own timing slot.
    struct SignSlot { std::string key; };
    std::vector<SignSlot> slots;

    for (const auto& word : words) {
        if (signs_.count(word)) {
            slots.push_back({word});
        } else {
            // Fingerspell: each letter is a slot
            for (char c : word) {
                std::string key(1, c);
                slots.push_back({key});
            }
            // Add a brief pause between fingerspelled words
            slots.push_back({"rest"});
        }
    }

    // Distribute timing evenly
    float time_per_slot = total_duration_sec / (float)slots.size();
    // Clamp fingerspelling speed: at most 400ms per letter
    float max_letter_time = 0.4f;
    if (time_per_slot > max_letter_time) time_per_slot = max_letter_time;

    float t = 0.0f;
    for (const auto& slot : slots) {
        WordTiming wt;
        wt.word = slot.key;
        wt.start_sec = t;
        wt.end_sec = t + time_per_slot;
        word_timings_.push_back(wt);
        t += time_per_slot;
    }

    // Extend last slot to fill remaining duration
    if (!word_timings_.empty()) {
        word_timings_.back().end_sec = total_duration_sec;
    }
}

void SignLanguageFace::setWordTimings(const std::vector<WordTiming>& timings) {
    std::lock_guard<std::mutex> lock(render_mutex_);
    word_timings_ = timings;
    if (!timings.empty()) {
        total_duration_ = timings.back().end_sec;
    }
}

void SignLanguageFace::feedAudio(const int16_t* samples, size_t count, int sample_rate) {
    float dt = (float)count / (float)sample_rate;
    playback_time_.store(playback_time_.load() + dt);
}

std::string SignLanguageFace::currentSign() const {
    return current_sign_;
}

void SignLanguageFace::reset() {
    playback_time_.store(0.0f);
    current_sign_ = "rest";
}

void SignLanguageFace::rgbToYuv420(const uint8_t* rgb, int w, int h, YUVData& yuv) {
    yuv.width = w;
    yuv.height = h;
    yuv.y_size = w * h;
    yuv.uv_size = (w / 2) * (h / 2);
    yuv.y = std::make_unique<uint8_t[]>(yuv.y_size);
    yuv.u = std::make_unique<uint8_t[]>(yuv.uv_size);
    yuv.v = std::make_unique<uint8_t[]>(yuv.uv_size);

    for (int py = 0; py < h; py++) {
        for (int px = 0; px < w; px++) {
            int idx = (py * w + px) * 3;
            uint8_t r = rgb[idx], g = rgb[idx + 1], b = rgb[idx + 2];
            int Y = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
            yuv.y[py * w + px] = (uint8_t)std::max(0, std::min(255, Y));
            if ((py % 2 == 0) && (px % 2 == 0)) {
                int uv_idx = (py / 2) * (w / 2) + (px / 2);
                int U = ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
                int V = ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
                yuv.u[uv_idx] = (uint8_t)std::max(0, std::min(255, U));
                yuv.v[uv_idx] = (uint8_t)std::max(0, std::min(255, V));
            }
        }
    }
}

void SignLanguageFace::renderSign(const std::string& key, YUVData& out) {
    auto it = signs_.find(key);
    if (it == signs_.end()) {
        it = signs_.find("rest");
        if (it == signs_.end()) return;
    }

    const SignImage& img = it->second;
    if (img.width == out_w_ && img.height == out_h_) {
        rgbToYuv420(img.rgb.data(), img.width, img.height, out);
    } else {
        // Nearest-neighbor scale
        std::vector<uint8_t> scaled(out_w_ * out_h_ * 3);
        for (int y = 0; y < out_h_; y++) {
            int sy = y * img.height / out_h_;
            for (int x = 0; x < out_w_; x++) {
                int sx = x * img.width / out_w_;
                int src = (sy * img.width + sx) * 3;
                int dst = (y * out_w_ + x) * 3;
                scaled[dst] = img.rgb[src];
                scaled[dst + 1] = img.rgb[src + 1];
                scaled[dst + 2] = img.rgb[src + 2];
            }
        }
        rgbToYuv420(scaled.data(), out_w_, out_h_, out);
    }
}

bool SignLanguageFace::renderFrame(YUVData& out) {
    std::lock_guard<std::mutex> lock(render_mutex_);

    if (signs_.empty()) return false;

    float t = playback_time_.load();

    // Find current word/sign based on timing
    std::string sign_key = "rest";
    for (const auto& wt : word_timings_) {
        if (t >= wt.start_sec && t < wt.end_sec) {
            sign_key = wt.word;
            break;
        }
    }

    if (sign_key != current_sign_) {
        current_sign_ = sign_key;
        if (sign_callback_) {
            sign_callback_(sign_key, t);
        }
    }

    renderSign(sign_key, out);
    return true;
}
