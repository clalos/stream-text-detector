#!/bin/bash

# Webcam RTSP Stream Script
# Creates an RTSP stream from webcam using mediamtx and FFmpeg
# Usage: ./webcam-rtsp-stream.sh [device] [stream_name]

set -euo pipefail

# Configuration
WEBCAM_DEVICE="${1:-/dev/video1}"
STREAM_NAME="${2:-mystream}"
RTSP_PORT="8554"
RTSP_URL="rtsp://localhost:${RTSP_PORT}/${STREAM_NAME}"
MEDIAMTX_EXECUTABLE="mediamtx"

# Process tracking
MEDIAMTX_PID=""
FFMPEG_PID=""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_debug() {
    echo -e "${BLUE}[DEBUG]${NC} $1"
}

# Check if required commands exist
check_dependencies() {
    local missing_deps=()
    
    if ! command -v "${MEDIAMTX_EXECUTABLE}" &> /dev/null; then
        missing_deps+=("mediamtx")
    fi
    
    if ! command -v ffmpeg &> /dev/null; then
        missing_deps+=("ffmpeg")
    fi
    
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        log_error "Missing dependencies: ${missing_deps[*]}"
        log_info "Install instructions:"
        for dep in "${missing_deps[@]}"; do
            case $dep in
                "mediamtx")
                    echo "  - Download from: https://github.com/bluenviron/mediamtx/releases"
                    echo "    Or install via package manager"
                    ;;
                "ffmpeg")
                    echo "  - Ubuntu/Debian: sudo apt install ffmpeg"
                    echo "  - macOS: brew install ffmpeg"
                    ;;
            esac
        done
        exit 1
    fi
}

# Check if webcam device exists
check_webcam() {
    if [ ! -e "${WEBCAM_DEVICE}" ]; then
        log_error "Webcam device ${WEBCAM_DEVICE} not found"
        log_info "Available video devices:"
        ls -la /dev/video* 2>/dev/null || echo "  No video devices found"
        exit 1
    fi
    
    log_info "Using webcam device: ${WEBCAM_DEVICE}"
}

# Cleanup function called on exit
cleanup() {
    log_info "Cleaning up processes..."
    
    # Kill FFmpeg if running
    if [ -n "${FFMPEG_PID}" ] && kill -0 "${FFMPEG_PID}" 2>/dev/null; then
        log_debug "Stopping FFmpeg (PID: ${FFMPEG_PID})"
        kill -TERM "${FFMPEG_PID}" 2>/dev/null || true
        wait "${FFMPEG_PID}" 2>/dev/null || true
    fi
    
    # Kill mediamtx if running
    if [ -n "${MEDIAMTX_PID}" ] && kill -0 "${MEDIAMTX_PID}" 2>/dev/null; then
        log_debug "Stopping mediamtx (PID: ${MEDIAMTX_PID})"
        kill -TERM "${MEDIAMTX_PID}" 2>/dev/null || true
        wait "${MEDIAMTX_PID}" 2>/dev/null || true
    fi
    
    log_info "Cleanup complete"
}

# Setup signal handlers
trap cleanup EXIT
trap cleanup SIGINT
trap cleanup SIGTERM

# Start mediamtx RTSP server
start_mediamtx() {
    log_info "Starting mediamtx RTSP server on port ${RTSP_PORT}..."
    
    # Create temporary config if needed
    local config_file="/tmp/mediamtx.yml"
    cat > "${config_file}" << EOF
# MediaMTX configuration for webcam streaming
rtspAddress: :${RTSP_PORT}
rtmpAddress: :1935
hlsAddress: :8888
webRTCAddress: :8889

# Path settings
paths:
  ${STREAM_NAME}:
    # Accept any source
EOF
    
    "${MEDIAMTX_EXECUTABLE}" "${config_file}" &
    MEDIAMTX_PID=$!
    
    log_debug "mediamtx started with PID: ${MEDIAMTX_PID}"
    
    # Wait for mediamtx to start
    sleep 2
    
    if ! kill -0 "${MEDIAMTX_PID}" 2>/dev/null; then
        log_error "Failed to start mediamtx"
        exit 1
    fi
    
    log_info "mediamtx is running"
}

# Start FFmpeg webcam capture
start_ffmpeg() {
    log_info "Starting FFmpeg webcam capture..."
    log_info "Webcam: ${WEBCAM_DEVICE} -> RTSP: ${RTSP_URL}"
    
    # FFmpeg command to capture from webcam and stream to RTSP
    ffmpeg \
        -f v4l2 \
        -i "${WEBCAM_DEVICE}" \
        -vcodec libx264 \
        -preset fast \
        -tune zerolatency \
        -crf 23 \
        -maxrate 2M \
        -bufsize 4M \
        -g 50 \
        -f rtsp \
        "${RTSP_URL}" \
        -loglevel warning \
        &
    
    FFMPEG_PID=$!
    log_debug "FFmpeg started with PID: ${FFMPEG_PID}"
    
    # Wait for FFmpeg to initialize
    sleep 3
    
    if ! kill -0 "${FFMPEG_PID}" 2>/dev/null; then
        log_error "Failed to start FFmpeg"
        exit 1
    fi
    
    log_info "FFmpeg is streaming webcam to RTSP"
}


# Test stream availability
test_stream() {
    log_info "Testing stream availability..."
    
    # Use ffprobe to test the stream
    if ffprobe -v quiet -select_streams v:0 -show_entries stream=width,height,r_frame_rate "${RTSP_URL}" &>/dev/null; then
        log_info "Stream is available and accessible"
        return 0
    else
        log_warn "Stream test failed, but this might be normal during startup"
        return 1
    fi
}

# Main execution
main() {
    log_info "Starting Webcam RTSP Stream"
    log_info "Device: ${WEBCAM_DEVICE}, Stream: ${STREAM_NAME}, URL: ${RTSP_URL}"
    echo
    
    # Pre-flight checks
    check_dependencies
    check_webcam
    
    # Start services in order
    start_mediamtx
    start_ffmpeg
    
    # Test stream (optional, continue even if it fails)
    test_stream || true
    
    echo
    log_info "RTSP stream is now available!"
    echo
    log_info "Stream URL: ${RTSP_URL}"
    echo
    log_info "You can view the stream with:"
    log_info "  mpv ${RTSP_URL}"
    log_info "  ffplay ${RTSP_URL}"
    log_info "  vlc ${RTSP_URL}"
    echo
    log_info "Press Ctrl+C to stop all services"
    
    # Wait for any process to exit or signal
    while true; do
        # Check if any critical process has died
        if [ -n "${MEDIAMTX_PID}" ] && ! kill -0 "${MEDIAMTX_PID}" 2>/dev/null; then
            log_error "mediamtx process died unexpectedly"
            break
        fi
        
        if [ -n "${FFMPEG_PID}" ] && ! kill -0 "${FFMPEG_PID}" 2>/dev/null; then
            log_error "FFmpeg process died unexpectedly"
            break
        fi
        
        sleep 1
    done
}

# Show usage information
show_usage() {
    echo "Usage: $0 [WEBCAM_DEVICE] [STREAM_NAME]"
    echo
    echo "Arguments:"
    echo "  WEBCAM_DEVICE   Webcam device path (default: /dev/video1)"
    echo "  STREAM_NAME     RTSP stream name (default: mystream)"
    echo
    echo "Examples:"
    echo "  $0                              # Use defaults"
    echo "  $0 /dev/video0                  # Use different webcam"
    echo "  $0 /dev/video0 webcam           # Custom device and stream name"
    echo
    echo "The script will:"
    echo "  1. Start mediamtx RTSP server on port 8554"
    echo "  2. Capture webcam with FFmpeg and stream to RTSP"
    echo "  3. Print the RTSP URL for viewing with external players"
    echo "  4. Handle Ctrl+C gracefully to stop all processes"
}

# Handle help argument
if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    show_usage
    exit 0
fi

# Run main function
main "$@"