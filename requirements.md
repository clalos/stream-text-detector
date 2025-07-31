# Stream Text Detector – Requirements & Development Plan

*(simple CLI written in Go)*
IMPORTANT!: no docker

---

## 1  Overview

Build a lightweight command-line application that continuously pulls a live video stream (e.g. RTSP from a set-top box or IP camera), grabs one frame every second, applies Optical Character Recognition (OCR) to that frame, and logs whenever a target word (or list of words) appears on-screen. The tool must be easy to compile and run on any developer workstation without container, CI/CD, or observability extras.

---

## 2  Functional Requirements

| ID  | Description                                                                                         |
| --- | --------------------------------------------------------------------------------------------------- |
| F-1 | Accept any RTSP/HTTP video URL as input (command-line flag).                                        |
| F-2 | Capture the **current** frame every *n* seconds (default = 1.0 s; flag-configurable).               |
| F-3 | Run OCR on the captured frame and extract all text strings.                                         |
| F-4 | Compare extracted text against a user-supplied word or word-list (case-insensitive, configurable).  |
| F-5 | When a match occurs, write a structured log entry with timestamp, matched word(s), and frame index. |
| F-6 | Keep running until the stream ends or the user terminates the process.                              |

---

## 3  Non-Functional Requirements

* **Language**: Go latest version(standard toolchain only).
* **Portability**: Linux, macOS, Windows (x86-64; GPU optional).
* **Accuracy target**: ≥ 95 % precision/recall on SD/HD broadcast captions after recommended preprocessing (see §6).
* **Performance**: sustained 1 fps OCR on 1080p streams with ≤ 300 MB RAM and ≤ 1 CPU core when no GPU is present.
* **Ease of use**: single binary, invoked as

  ```bash
  std -url rtsp://… -word "BREAKING" -interval 1s
  ```
* **Observability**: plain stdout/stderr logs (no metrics backend).

---

## 4  Technical Stack & Dependencies

| Purpose                          | Chosen tool                                                      | Why this tool                                                                                          |
| -------------------------------- | ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| Video capture & frame extraction | **GoCV** (Go bindings for OpenCV 4.x) ([GitHub][1])              | Mature, cross-platform, LGPL-compatible, supports RTSP/FFmpeg pipelines and GPU acceleration.          |
| OCR engine                       | **Tesseract-OCR v5** with **gosseract** Go wrapper ([GitHub][2]) | Best-in-class open-source OCR, LSTM models, multi-language packs; gosseract provides idiomatic Go API. |
| Logging                          | `log/slog` (std-lib)                                             | No external infra required; human + machine readable.                                                  |
| Image preprocessing              | OpenCV (part of GoCV).                                           | Offers grayscale, resize, threshold operations to improve OCR accuracy.                                |

> **System prerequisites**: an OpenCV ≥ 4.12 build (with FFMPEG enabled) and Tesseract headers/libs must be present at compile-time; both are available via common package managers (apt, brew, choco).

---

## 5  High-Level Architecture & Data Flow

```
RTSP Stream ─▶ VideoCapture (GoCV) ─▶ Frame every N s ─▶ Pre-processing
                                            │
                                            ▼
                 Word match? ◀─ OCR (Tesseract via gosseract)
                                            │
                                            ▼
                                      Structured Log
```

1. **Stream Init**: open RTSP URL with `gocv.OpenVideoCapture`, which transparently handles network buffering. ([GitHub][3])
2. **Scheduler**: ticker loop triggers frame grab at configured interval.
3. **Pre-processing** (pipeline):
   * i.e. Convert to grayscale → optional resize up to 150 % → adaptive threshold. These steps empirically raise Tesseract recall on broadcast text. ([blog.gdeltproject.org][4])
4. **OCR call**: pass pre-processed image bytes to gosseract client; supply language model and whitelist characters if known.
5. **Matcher**: compare normalized OCR output against target word(s); allow regex or fuzzy distance ≤ 1 (optional flag).
6. **Logger**: write event (ISO-8601 timestamp, frame #, word, confidence, stream URL).

Concurrency note: run capture and OCR in separate goroutines connected by a buffered channel to avoid frame drops when OCR is slower than 1 s.

---

## 6  Accuracy Guidelines

1. **Use latest LSTM language data** (`eng.traineddata` and italian data).
---

## 7  Command-Line Interface (CLI) Specification

| Flag          | Type                | Default | Description                              |
| ------------- | ------------------- | ------- | ---------------------------------------- |
| `-url`        | string              | —       | RTSP/HTTP(S) video stream source.        |
| `-word`       | string (repeatable) | —       | Target word(s) to detect.                |
| `-interval`   | duration            | `1s`    | Frame sampling interval.                 |
| `-lang`       | string              | `eng`   | Tesseract language codes (comma-sep).    |
| `-confidence` | float               | `0.80`  | Minimum OCR confidence to count a match. |
| `-logfmt`     | enum                | `json`  | `json` or `kv` output.                   |

---

### References

* GoCV – Go bindings for OpenCV 4.x ([GitHub][1])
* gosseract – Go wrapper for Tesseract OCR ([GitHub][2])
* GoCV RTSP capture example ([GitHub][3])
* Case study: boosting Tesseract accuracy on broadcast TV frames by up-scaling ([blog.gdeltproject.org][4])

---

[1]: https://github.com/hybridgroup/gocv?utm_source=chatgpt.com "hybridgroup/gocv: Go package for computer vision using OpenCV 4 ..."
[2]: https://github.com/otiai10/gosseract?utm_source=chatgpt.com "GitHub - otiai10/gosseract: Go package for OCR (Optical Character ..."
[3]: https://github.com/hybridgroup/gocv/issues/449?utm_source=chatgpt.com "hybridgroup/gocv - How to access the RTSP stream video? - GitHub"
[4]: https://blog.gdeltproject.org/using-tesseract-to-ocr-television-news-a-case-study-of-cnn-bloomberg-news/?utm_source=chatgpt.com "Using Tesseract To OCR Television News: A Case Study Of CNN ..."
