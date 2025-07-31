# Installation Guide

This guide provides detailed instructions for installing the Stream Text Detector on various operating systems.

## System Requirements

- Go 1.24 or later
- OpenCV 4.x with FFmpeg support
- Tesseract OCR 5.x with development headers
- pkg-config (for dependency resolution)

## Platform-Specific Installation

### Ubuntu/Debian

```bash
# Update package list
sudo apt update

# Install OpenCV and development headers
sudo apt install libopencv-dev pkg-config

# Install Tesseract OCR and development headers
sudo apt install tesseract-ocr libtesseract-dev

# Install additional language packs (optional)
sudo apt install tesseract-ocr-eng tesseract-ocr-ita tesseract-ocr-fra

# Verify installations
pkg-config --modversion opencv4
tesseract --version
```

### CentOS/RHEL/Fedora

```bash
# For CentOS/RHEL (enable EPEL repository first)
sudo yum install epel-release
sudo yum install opencv-devel tesseract-devel pkg-config

# For Fedora
sudo dnf install opencv-devel tesseract-devel pkg-config

# Install language packs
sudo yum install tesseract-langpack-eng tesseract-langpack-ita
# OR for Fedora:
sudo dnf install tesseract-langpack-eng tesseract-langpack-ita
```

### macOS

Using Homebrew:

```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install opencv pkg-config tesseract

# Verify installations
pkg-config --modversion opencv4
tesseract --version
```

Using MacPorts:

```bash
# Install MacPorts dependencies
sudo port install opencv4 +universal tesseract +universal

# Set environment variables
export PKG_CONFIG_PATH=/opt/local/lib/pkgconfig:$PKG_CONFIG_PATH
```

### Windows

Using Chocolatey:

```powershell
# Install Chocolatey if not already installed
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

# Install dependencies
choco install opencv tesseract

# You may need to set environment variables manually
# Add OpenCV and Tesseract bin directories to PATH
```

Using vcpkg (recommended for development):

```cmd
# Clone vcpkg
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat

# Install dependencies
vcpkg install opencv4[ffmpeg]:x64-windows
vcpkg install tesseract:x64-windows

# Integrate with system
vcpkg integrate install
```

### Docker Alternative (Development Only)

If you prefer using Docker for development (not for production as per requirements):

```dockerfile
FROM golang:1.24-bookworm

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libopencv-dev \
    libtesseract-dev \
    tesseract-ocr \
    tesseract-ocr-eng \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN go mod download
RUN go build -o std .

CMD ["./std"]
```

## Building the Application

Once dependencies are installed:

```bash
# Clone the repository
git clone https://github.com/clalos/stream-text-detector.git
cd stream-text-detector

# Download Go dependencies
go mod download

# Build the application
go build -o std .

# Verify the build
./std -help
```

## Troubleshooting

### OpenCV Issues

**"Package opencv4 was not found"**

1. Ensure OpenCV is installed with development headers
2. Check if pkg-config can find OpenCV:
   ```bash
   pkg-config --list-all | grep opencv
   ```
3. Set PKG_CONFIG_PATH if needed:
   ```bash
   export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH
   ```

**OpenCV version compatibility**

- GoCV requires OpenCV 4.x
- Check your OpenCV version: `pkg-config --modversion opencv4`
- If you have OpenCV 3.x, upgrade to 4.x

### Tesseract Issues

**"leptonica/allheaders.h: No such file or directory"**

1. Install Tesseract development headers:
   ```bash
   sudo apt install libtesseract-dev libleptonica-dev
   ```

2. For macOS with Homebrew:
   ```bash
   brew install leptonica
   ```

**Missing language packs**

```bash
# List available languages
tesseract --list-langs

# Install additional language packs
sudo apt install tesseract-ocr-[lang-code]
```

### CGO Issues

If you encounter CGO-related errors:

```bash
# Ensure CGO is enabled
export CGO_ENABLED=1

# Set compiler flags if needed
export CGO_CFLAGS="-I/usr/local/include"
export CGO_LDFLAGS="-L/usr/local/lib"
```

### Cross-Compilation

Cross-compilation requires the target platform's OpenCV and Tesseract libraries. This is complex and not recommended. Instead, build natively on each target platform.

## Performance Optimization

### GPU Acceleration (Optional)

If you have an NVIDIA GPU and want to enable GPU acceleration:

```bash
# Install CUDA development toolkit
# Ubuntu/Debian:
sudo apt install nvidia-cuda-toolkit

# Build OpenCV with CUDA support (advanced)
# This requires building OpenCV from source with CUDA flags enabled
```

### Memory Optimization

For production deployments with memory constraints:

```bash
# Build with optimizations
go build -ldflags="-s -w" -o std .

# Use UPX for further compression (optional)
upx --lzma std
```

## Verification

Test your installation:

```bash
# Run a quick test (will fail without valid stream, but tests CLI parsing)
./std -url test://invalid -word test 2>&1 | grep "Failed to open video capture"

# If you see the error above, the installation is working correctly
# The error is expected since "test://invalid" is not a valid stream URL
```