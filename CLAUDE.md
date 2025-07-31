# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Stream Text Detector is a Go CLI application that monitors live video streams (RTSP/HTTP) for specific text content using Optical Character Recognition (OCR). It uses OpenCV for video processing and Tesseract for OCR, with a concurrent pipeline architecture for real-time frame processing.

## Build and Development Commands

### Dependencies
```bash
make deps          # Download and verify dependencies
make deps-update   # Update all dependencies
```

### Building
```bash
make build         # Build binary with OpenCV support
make build-all     # Cross-compile for Linux, Windows, macOS
go build -o std .  # Direct build command
```

### Testing
```bash
make test          # Run tests (requires OpenCV and Tesseract)
make test-coverage # Run tests with coverage report
```

### Running
```bash
make run ARGS="-url rtsp://example.com -word BREAKING"
./std -url rtsp://example.com -word "BREAKING,URGENT" -interval 2s
```

## Architecture

### Core Components
- **main.go**: CLI entry point with flag parsing and graceful shutdown
- **detector.go**: Main detection logic with concurrent pipeline:
  - Frame capture goroutine (captures at intervals)
  - OCR processing goroutine (processes frames with Tesseract)
  - Result logging goroutine (logs matches with structured output)

### Build Configuration
All code now requires OpenCV and Tesseract to be installed on the system. The build tag separation has been removed for simplicity.

### Key Types
- `Config`: CLI configuration parsed from flags
- `Detector`: Main processing engine managing video capture and OCR
- `Frame`: Video frame with metadata (index, timestamp)
- `DetectionResult`: OCR results with confidence and word matches

### Dependencies
- `gocv.io/x/gocv`: OpenCV Go bindings for video processing
- `github.com/otiai10/gosseract/v2`: Tesseract OCR wrapper

### Performance Targets
- Memory: ≤300MB RAM for 1080p streams at 1fps  
- CPU: ≤1 CPU core usage
- Accuracy: ≥95% precision/recall on broadcast captions

## System Requirements

The application requires OpenCV 4.x with FFmpeg support and Tesseract OCR 5.x to be installed on the system before building or running. See README.md for platform-specific installation instructions.