#!/bin/bash

# Example usage script for Stream Text Detector
# This script demonstrates various ways to use the application

echo "Stream Text Detector - Example Usage"
echo "===================================="

# Check if binary exists
if [ ! -f "./std" ]; then
    echo "Binary not found. Building application..."
    go build -o std .
    if [ $? -ne 0 ]; then
        echo "Build failed. Please check your Go installation and dependencies."
        exit 1
    fi
fi

echo ""
echo "Example 1: Basic usage with single word detection"
echo "Command: ./std -url rtsp://example.com/stream -word BREAKING"
echo ""

echo "Example 2: Multiple words with custom interval"
echo "Command: ./std -url rtsp://camera.local/live -word \"BREAKING,URGENT,ALERT\" -interval 2s"
echo ""

echo "Example 3: Multi-language OCR with higher confidence threshold"
echo "Command: ./std -url http://stream.example.com/video -word \"notizie,breaking\" -lang eng+ita -confidence 0.9"
echo ""

echo "Example 4: Key-value logging format"
echo "Command: ./std -url rtsp://192.168.1.100/stream -word NEWS -logfmt kv"
echo ""

echo "Example 5: High-frequency monitoring"
echo "Command: ./std -url rtsp://broadcast.example.com/live -word \"LIVE,NOW\" -interval 500ms -confidence 0.85"
echo ""

echo "To run any of these examples, copy the command and replace the URL with a valid stream."
echo "Press Ctrl+C to stop the application when running."
echo ""

# Uncomment the line below to run a test with a placeholder URL
# (This will fail without a real stream, but demonstrates the CLI interface)
# ./std -url rtsp://placeholder.example.com/stream -word TEST -interval 1s