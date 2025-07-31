// Package main implements a Stream Text Detector CLI application that continuously
// monitors video streams for specific text content using OCR.
//
// The application captures frames from RTSP/HTTP video streams at configurable
// intervals, processes them with Tesseract OCR, and logs when target words are detected.
// It uses a concurrent pipeline architecture with separate goroutines for frame capture,
// OCR processing, and result logging to achieve real-time performance.
//
// # Usage
//
// Basic usage with required parameters:
//
//	./std -url rtsp://example.com/stream -word "BREAKING"
//
// Multiple target words with custom settings:
//
//	./std -url rtsp://example.com/stream -word "BREAKING,URGENT,LIVE" \
//	      -interval 2s -confidence 0.9 -lang eng -logfmt json
//
// # System Requirements
//
// The application requires OpenCV 4.x with FFmpeg support and Tesseract OCR 5.x
// to be installed on the system. Memory usage is typically ≤300MB for 1080p streams
// at 1fps sampling rate.
//
// # Architecture
//
// The detector uses a three-stage concurrent pipeline:
//   - Frame capture: Samples frames at specified intervals from video stream
//   - OCR processing: Applies image preprocessing and Tesseract OCR analysis
//   - Result logging: Outputs structured logs when target words are detected
//
// All goroutines coordinate through buffered channels and respond to context
// cancellation for graceful shutdown.
package main

import (
	"context"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"
)

// Config holds the application configuration parsed from command-line flags.
// All fields are populated from CLI arguments and validated during parsing.
type Config struct {
	// URL is the video stream source. Supports RTSP, HTTP, and HTTPS protocols.
	// Examples: "rtsp://camera.example.com/stream", "http://example.com/video.mp4"
	URL string

	// Words contains the target text strings to detect in the video stream.
	// Detection is performed case-insensitively using substring matching.
	// Multiple words are specified as comma-separated values.
	Words []string

	// Interval specifies how frequently to capture and process frames.
	// Shorter intervals provide more responsive detection but increase CPU usage.
	// Typical values range from 500ms to 5s depending on requirements.
	Interval time.Duration

	// Language specifies Tesseract OCR language codes for text recognition.
	// Multiple languages can be specified as comma-separated values.
	// Common values: "eng", "eng+fra", "spa". Default is "eng".
	Language string

	// Confidence is the minimum OCR confidence threshold (0.0-1.0) required
	// for word matching. Higher values reduce false positives but may miss
	// valid detections. Default is 0.80 (80%).
	Confidence float64

	// LogFormat determines the structured logging output format.
	// Supported values: "json" for JSON format, "kv" for key=value format.
	// Default is "json".
	LogFormat string
}

// parseFlags parses command-line arguments and returns the application configuration.
// It validates all required parameters and applies default values where appropriate.
//
// Required flags:
//   - url: video stream source URL
//   - word: comma-separated target words to detect
//
// Optional flags with defaults:
//   - interval: frame sampling interval (default: 1s)
//   - lang: Tesseract language codes (default: "eng")
//   - confidence: minimum OCR confidence 0.0-1.0 (default: 0.80)
//   - logfmt: output format "json" or "kv" (default: "json")
//
// Returns an error if required flags are missing or values are invalid.
func parseFlags() (*Config, error) {
	// Create a new FlagSet to avoid global flag conflicts in tests
	fs := flag.NewFlagSet("std", flag.ContinueOnError)
	
	var (
		url        = fs.String("url", "", "RTSP/HTTP(S) video stream source (required)")
		wordsFlag  = fs.String("word", "", "Target word(s) to detect (comma-separated, required)")
		interval   = fs.Duration("interval", time.Second, "Frame sampling interval")
		lang       = fs.String("lang", "eng", "Tesseract language codes (comma-separated)")
		confidence = fs.Float64("confidence", 0.80, "Minimum OCR confidence to count a match")
		logfmt     = fs.String("logfmt", "json", "Log format: json or kv")
	)
	
	if err := fs.Parse(os.Args[1:]); err != nil {
		return nil, err
	}

	if *url == "" {
		return nil, fmt.Errorf("url flag is required")
	}

	if *wordsFlag == "" {
		return nil, fmt.Errorf("word flag is required")
	}

	if *logfmt != "json" && *logfmt != "kv" {
		return nil, fmt.Errorf("logfmt must be 'json' or 'kv'")
	}

	if *confidence < 0.0 || *confidence > 1.0 {
		return nil, fmt.Errorf("confidence must be between 0.0 and 1.0")
	}

	words := strings.Split(*wordsFlag, ",")
	for i, word := range words {
		words[i] = strings.TrimSpace(word)
	}

	return &Config{
		URL:        *url,
		Words:      words,
		Interval:   *interval,
		Language:   *lang,
		Confidence: *confidence,
		LogFormat:  *logfmt,
	}, nil
}

// setupLogger configures structured logging based on the specified format.
// It creates a logger with INFO level that outputs to stdout.
//
// Supported formats:
//   - "json": outputs structured JSON logs suitable for log aggregation systems
//   - "kv": outputs human-readable key=value format for console viewing
//   - any other value defaults to JSON format
//
// The logger includes structured fields for all detection events including
// timestamps, frame indices, matched words, confidence scores, and stream URLs.
func setupLogger(format string) *slog.Logger {
	var handler slog.Handler
	
	opts := &slog.HandlerOptions{
		Level: slog.LevelInfo,
	}

	switch format {
	case "json":
		handler = slog.NewJSONHandler(os.Stdout, opts)
	case "kv":
		handler = slog.NewTextHandler(os.Stdout, opts)
	default:
		handler = slog.NewJSONHandler(os.Stdout, opts)
	}

	return slog.New(handler)
}

func main() {
	config, err := parseFlags()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error parsing flags: %v\n", err)
		os.Exit(1)
	}

	logger := setupLogger(config.LogFormat)
	slog.SetDefault(logger)

	logger.Info("Starting Stream Text Detector",
		"url", config.URL,
		"words", config.Words,
		"interval", config.Interval,
		"language", config.Language,
		"confidence", config.Confidence,
		"log_format", config.LogFormat,
	)

	// Create context for graceful shutdown
	ctx, cancel := context.WithCancel(context.Background())

	// Handle interrupt signals
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)

	// Create and run the detector
	detector, err := NewDetector(config, logger)
	if err != nil {
		logger.Error("Failed to create detector", "error", err)
		os.Exit(1)
	}

	// Setup signal handler in goroutine
	go func() {
		<-sigChan
		logger.Info("Received shutdown signal, stopping...")
		cancel()
	}()

	// Run the detector and handle shutdown properly
	runErr := detector.Run(ctx)

	// Ensure context is cancelled (in case Run returned due to error)
	cancel()

	// detector.Run() now guarantees all goroutines have stopped before returning,
	// so we can safely close resources without race conditions
	if closeErr := detector.Close(); closeErr != nil {
		logger.Error("Error during detector cleanup", "error", closeErr)
	}

	// Handle run errors after cleanup
	if runErr != nil {
		logger.Error("Detector failed", "error", runErr)
		os.Exit(1)
	}

	logger.Info("Stream Text Detector stopped")
}