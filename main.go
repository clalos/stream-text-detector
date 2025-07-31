// Package main implements a Stream Text Detector CLI application that continuously
// monitors video streams for specific text content using OCR.
//
// The application captures frames from RTSP/HTTP video streams at configurable
// intervals, processes them with Tesseract OCR, and logs when target words are detected.
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
type Config struct {
	URL        string
	Words      []string
	Interval   time.Duration
	Language   string
	Confidence float64
	LogFormat  string
}

// parseFlags parses command-line arguments and returns the application configuration.
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
	defer cancel()

	// Handle interrupt signals
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)

	go func() {
		<-sigChan
		logger.Info("Received shutdown signal, stopping...")
		cancel()
	}()

	// Create and run the detector
	detector, err := NewDetector(config, logger)
	if err != nil {
		logger.Error("Failed to create detector", "error", err)
		os.Exit(1)
	}
	defer detector.Close()

	if err := detector.Run(ctx); err != nil {
		logger.Error("Detector failed", "error", err)
		os.Exit(1)
	}

	logger.Info("Stream Text Detector stopped")
}