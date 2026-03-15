#!/bin/bash
set -e

ONNXRUNTIME_VERSION="1.17.3"
STYLETTS2_HF_REPO="DDATT/StyleTTS2-ONNX-Cpp"

echo "=== Whillats StyleTTS2 Deploy Script (Ubuntu Linux) ==="

# --- Install system dependencies ---
echo "[1/6] Installing system dependencies..."
apt-get update -qq
apt-get install -y -qq build-essential cmake git autoconf automake libtool \
    pkg-config libpulse-dev libasound2-dev curl wget > /dev/null 2>&1
echo "  Done."

# --- Clone and build ---
WHILLATS_DIR="${WHILLATS_DIR:-/opt/whillats}"

if [ -d "$WHILLATS_DIR/.git" ]; then
    echo "[2/6] Updating existing repo at $WHILLATS_DIR..."
    cd "$WHILLATS_DIR"
    git fetch origin
    git checkout styletts2
    git pull origin styletts2
else
    echo "[2/6] Cloning whillats to $WHILLATS_DIR..."
    git clone --branch styletts2 --recursive \
        https://github.com/wilddolphin2025/whillats.git "$WHILLATS_DIR"
    cd "$WHILLATS_DIR"
fi

# --- Check for CUDA ---
USE_CUDA=""
if command -v nvcc &> /dev/null; then
    echo "  CUDA detected: $(nvcc --version | grep release)"
    USE_CUDA="-DGGML_CUDA=ON"
fi

# --- Build ---
echo "[3/6] Building whillats with StyleTTS2..."
cmake -B build \
    -DWHILLATS_STYLETTS2=ON \
    -DGGML_METAL=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    $USE_CUDA

cmake --build build --config Release -j $(nproc)
echo "  Build complete."

# --- Download ONNX models ---
MODEL_DIR="$WHILLATS_DIR/trained_models"
mkdir -p "$MODEL_DIR"

if [ -f "$MODEL_DIR/final_simp.onnx" ]; then
    echo "[4/6] ONNX models already downloaded, skipping."
else
    echo "[4/6] Downloading StyleTTS2 ONNX models from HuggingFace..."
    HF_BASE="https://huggingface.co/$STYLETTS2_HF_REPO/resolve/main"

    for f in bert_encoder.onnx plbert_simp.onnx final_simp.onnx ref_s.bin ref_p.bin; do
        echo "  Downloading $f..."
        curl -L -s -o "$MODEL_DIR/$f" "$HF_BASE/$f"
    done
    echo "  Models downloaded."
fi

# --- Find espeak-ng-data ---
ESPEAK_DATA="$WHILLATS_DIR/build/bin/Release/espeak-ng-data"
if [ ! -d "$ESPEAK_DATA" ]; then
    ESPEAK_DATA=$(find "$WHILLATS_DIR/build" -name "espeak-ng-data" -type d | head -1)
fi

if [ -z "$ESPEAK_DATA" ] || [ ! -d "$ESPEAK_DATA" ]; then
    echo "[WARNING] espeak-ng-data not found in build output. Trying system path..."
    if [ -d "/usr/lib/x86_64-linux-gnu/espeak-ng-data" ]; then
        ESPEAK_DATA="/usr/lib/x86_64-linux-gnu/espeak-ng-data"
    elif [ -d "/usr/share/espeak-ng-data" ]; then
        ESPEAK_DATA="/usr/share/espeak-ng-data"
    fi
fi

echo "[5/6] espeak-ng-data at: $ESPEAK_DATA"

# --- Verify ---
echo "[6/6] Running verification test..."
TEST_BIN="$WHILLATS_DIR/build/bin/Release/test_whillats"
if [ ! -f "$TEST_BIN" ]; then
    TEST_BIN=$(find "$WHILLATS_DIR/build" -name "test_whillats" -type f | head -1)
fi

if [ -f "$TEST_BIN" ]; then
    export LD_LIBRARY_PATH="$WHILLATS_DIR/build/lib/Release:$WHILLATS_DIR/build/_deps/onnxruntime-src/lib:$LD_LIBRARY_PATH"
    export ESPEAK_DATA_PATH="$ESPEAK_DATA"
    export STYLETTS2_MODEL_DIR="$MODEL_DIR"

    "$TEST_BIN" --tts
    echo ""
    echo "=== Deploy complete ==="
    echo ""
    echo "Binaries:"
    echo "  Library:  $WHILLATS_DIR/build/lib/Release/libwhillats.so"
    echo "  Test:     $TEST_BIN"
    echo ""
    echo "To run manually:"
    echo "  export LD_LIBRARY_PATH=$WHILLATS_DIR/build/lib/Release:$WHILLATS_DIR/build/_deps/onnxruntime-src/lib"
    echo "  export ESPEAK_DATA_PATH=$ESPEAK_DATA"
    echo "  export STYLETTS2_MODEL_DIR=$MODEL_DIR"
    echo "  $TEST_BIN --tts"
else
    echo "[ERROR] test_whillats binary not found!"
    exit 1
fi
