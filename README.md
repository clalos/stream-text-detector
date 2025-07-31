# Stream Text Detector

A lightweight command-line application that continuously monitors live video streams (RTSP/HTTP) for specific text content using Optical Character Recognition (OCR). The tool captures frames at configurable intervals, processes them with Tesseract OCR, and logs when target words are detected.

## Features

- **Video Stream Support**: Handles RTSP and HTTP(S) video streams
- **Real-time OCR**: Uses Tesseract OCR with configurable languages and confidence thresholds
- **Concurrent Processing**: Efficient frame capture and OCR processing using goroutines
- **Text Detection**: Case-insensitive word matching with structured logging
- **Performance Optimized**: Designed for ≤300MB RAM and ≤1 CPU core usage
- **Cross-platform**: Supports Linux, macOS, and Windows

## System Prerequisites

Before building and running the application, ensure you have the following dependencies installed:

### OpenCV 4.x with FFmpeg Support

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install libopencv-dev pkg-config
```

**macOS (Homebrew):**
```bash
brew install opencv pkg-config
```

**Windows (Chocolatey):**
```powershell
choco install opencv
```

### Tesseract OCR 5.x

**Ubuntu/Debian:**
```bash
sudo apt install tesseract-ocr libtesseract-dev
# Install additional language packs if needed
sudo apt install tesseract-ocr-eng tesseract-ocr-ita
```

**macOS (Homebrew):**
```bash
brew install tesseract
```

**Windows (Chocolatey):**
```powershell
choco install tesseract
```

### Go 1.24+

Download and install Go from [https://golang.org/dl/](https://golang.org/dl/)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/clalos/stream-text-detector.git
cd stream-text-detector
```

2. Download dependencies:
```bash
go mod download
```

3. Build the application:
```bash
go build -o std .
```

## Usage

### Basic Usage

```bash
./std -url rtsp://example.com/stream -word "BREAKING"
```

### Advanced Usage

```bash
./std \
  -url rtsp://camera.example.com/live \
  -word "BREAKING,URGENT,ALERT" \
  -interval 2s \
  -lang eng+ita \
  -confidence 0.85 \
  -logfmt json
```

### Command-Line Options

| Flag          | Type     | Default | Description                                    |
|---------------|----------|---------|------------------------------------------------|
| `-url`        | string   | —       | RTSP/HTTP(S) video stream source (required)   |
| `-word`       | string   | —       | Target word(s) to detect, comma-separated (required) |
| `-interval`   | duration | `1s`    | Frame sampling interval                        |
| `-lang`       | string   | `eng`   | Tesseract language codes, plus-separated (e.g., eng+ita) |
| `-confidence` | float    | `0.80`  | Minimum OCR confidence (0.0-1.0)             |
| `-logfmt`     | enum     | `json`  | Log format: `json` or `kv`                    |

### Example Output

**JSON Format:**
```json
{
  "time": "2024-01-15T10:30:45Z",
  "level": "INFO",
  "msg": "Text detected in stream",
  "timestamp": "2024-01-15T10:30:45Z",
  "frame_index": 42,
  "matched_words": ["BREAKING"],
  "confidence": 0.92,
  "extracted_text": "BREAKING NEWS: Market Update",
  "stream_url": "rtsp://example.com/stream"
}
```

**Key-Value Format:**
```
time=2024-01-15T10:30:45Z level=INFO msg="Text detected in stream" timestamp=2024-01-15T10:30:45Z frame_index=42 matched_words="[BREAKING]" confidence=0.92 extracted_text="BREAKING NEWS: Market Update" stream_url=rtsp://example.com/stream
```

## Performance Characteristics

The application is optimized for efficiency:

- **Memory Usage**: ≤300MB RAM for 1080p streams at 1fps
- **CPU Usage**: ≤1 CPU core when no GPU acceleration is available
- **Accuracy**: ≥95% precision/recall on SD/HD broadcast captions
- **Latency**: Real-time processing with configurable frame intervals

## Architecture

The application uses a concurrent pipeline architecture:

1. **Frame Capture**: Goroutine captures frames from video stream at specified intervals
2. **Image Preprocessing**: Frames are converted to grayscale, resized, and thresholded for optimal OCR accuracy
3. **OCR Processing**: Tesseract extracts text from preprocessed frames
4. **Word Matching**: Case-insensitive matching against target words
5. **Structured Logging**: Results are logged with timestamps and metadata

## Troubleshooting

### Common Issues

**"Failed to open video capture"**
- Verify the stream URL is accessible
- Check network connectivity
- Ensure OpenCV was built with FFmpeg support

**"Failed to set OCR language"**
- Install the required Tesseract language packs
- Verify language codes are correct (e.g., 'eng' for English)

**High Memory Usage**
- Reduce frame capture interval
- Lower video resolution if possible
- Check for memory leaks in long-running instances

**Poor OCR Accuracy**
- Adjust confidence threshold
- Try different Tesseract language models
- Verify video quality and text clarity

### Debug Mode

For debugging, you can enable verbose logging by modifying the log level in the source code or using environment variables supported by your system.

## Development

### Building from Source

```bash
# Install dependencies
go mod download

# Run tests (requires OpenCV and Tesseract to be installed)
go test ./...

# Build with optimizations
go build -ldflags="-s -w" -o std .
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

## License

This project is open source. See the LICENSE file for details.

## Acknowledgments

- [GoCV](https://gocv.io/) - Go bindings for OpenCV
- [gosseract](https://github.com/otiai10/gosseract) - Go wrapper for Tesseract OCR
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) - Open source OCR engine