package main

import (
	"log/slog"
	"os"
	"strings"
	"testing"
	"time"
)

func TestParseFlags(t *testing.T) {
	tests := []struct {
		name    string
		args    []string
		want    *Config
		wantErr bool
	}{
		{
			name: "valid basic config",
			args: []string{"-url", "rtsp://example.com", "-word", "test"},
			want: &Config{
				URL:        "rtsp://example.com",
				Words:      []string{"test"},
				Interval:   time.Second,
				Language:   "eng",
				Confidence: 0.80,
				LogFormat:  "json",
			},
			wantErr: false,
		},
		{
			name: "valid config with multiple words",
			args: []string{"-url", "rtsp://example.com", "-word", "test,word,match"},
			want: &Config{
				URL:        "rtsp://example.com",
				Words:      []string{"test", "word", "match"},
				Interval:   time.Second,
				Language:   "eng",
				Confidence: 0.80,
				LogFormat:  "json",
			},
			wantErr: false,
		},
		{
			name: "valid config with all options",
			args: []string{
				"-url", "rtsp://example.com",
				"-word", "test",
				"-interval", "2s",
				"-lang", "eng+ita",
				"-confidence", "0.9",
				"-logfmt", "kv",
			},
			want: &Config{
				URL:        "rtsp://example.com",
				Words:      []string{"test"},
				Interval:   2 * time.Second,
				Language:   "eng+ita",
				Confidence: 0.9,
				LogFormat:  "kv",
			},
			wantErr: false,
		},
		{
			name:    "missing url",
			args:    []string{"-word", "test"},
			wantErr: true,
		},
		{
			name:    "missing word",
			args:    []string{"-url", "rtsp://example.com"},
			wantErr: true,
		},
		{
			name:    "invalid log format",
			args:    []string{"-url", "rtsp://example.com", "-word", "test", "-logfmt", "invalid"},
			wantErr: true,
		},
		{
			name:    "invalid confidence",
			args:    []string{"-url", "rtsp://example.com", "-word", "test", "-confidence", "1.5"},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Save original args and restore after test
			origArgs := os.Args
			defer func() { os.Args = origArgs }()

			// Set test args
			os.Args = append([]string{"test"}, tt.args...)

			got, err := parseFlags()
			if (err != nil) != tt.wantErr {
				t.Errorf("parseFlags() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if !tt.wantErr && got != nil && tt.want != nil {
				if got.URL != tt.want.URL {
					t.Errorf("parseFlags() URL = %v, want %v", got.URL, tt.want.URL)
				}
				if !equalStringSlices(got.Words, tt.want.Words) {
					t.Errorf("parseFlags() Words = %v, want %v", got.Words, tt.want.Words)
				}
				if got.Interval != tt.want.Interval {
					t.Errorf("parseFlags() Interval = %v, want %v", got.Interval, tt.want.Interval)
				}
				if got.Language != tt.want.Language {
					t.Errorf("parseFlags() Language = %v, want %v", got.Language, tt.want.Language)
				}
				if got.Confidence != tt.want.Confidence {
					t.Errorf("parseFlags() Confidence = %v, want %v", got.Confidence, tt.want.Confidence)
				}
				if got.LogFormat != tt.want.LogFormat {
					t.Errorf("parseFlags() LogFormat = %v, want %v", got.LogFormat, tt.want.LogFormat)
				}
			}
		})
	}
}

func TestFindMatches(t *testing.T) {
	config := &Config{
		Words: []string{"BREAKING", "NEWS", "urgent"},
	}
	
	logger := slog.New(slog.NewTextHandler(os.Stdout, nil))
	
	// Create a detector with mock components (we'll only test findMatches)
	detector := &Detector{
		config: config,
		logger: logger,
	}

	tests := []struct {
		name string
		text string
		want []string
	}{
		{
			name: "single match",
			text: "This is a BREAKING story",
			want: []string{"BREAKING"},
		},
		{
			name: "multiple matches",
			text: "BREAKING NEWS: urgent update",
			want: []string{"BREAKING", "NEWS", "urgent"},
		},
		{
			name: "case insensitive match",
			text: "breaking news is urgent",
			want: []string{"BREAKING", "NEWS", "urgent"},
		},
		{
			name: "no matches",
			text: "This is just regular text",
			want: nil,
		},
		{
			name: "empty text",
			text: "",
			want: nil,
		},
		{
			name: "partial word matches",
			text: "breakingpoint newscaster urgently",
			want: []string{"BREAKING", "NEWS", "urgent"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := detector.findMatches(tt.text)
			if !equalStringSlices(got, tt.want) {
				t.Errorf("findMatches() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestSetupLogger(t *testing.T) {
	tests := []struct {
		name   string
		format string
	}{
		{
			name:   "json logger",
			format: "json",
		},
		{
			name:   "kv logger",
			format: "kv",
		},
		{
			name:   "default to json",
			format: "invalid",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			logger := setupLogger(tt.format)
			if logger == nil {
				t.Error("setupLogger() returned nil")
			}
		})
	}
}

// Helper function to compare string slices
func equalStringSlices(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	
	// Convert to maps for comparison (order doesn't matter)
	mapA := make(map[string]bool)
	mapB := make(map[string]bool)
	
	for _, s := range a {
		mapA[s] = true
	}
	for _, s := range b {
		mapB[s] = true
	}
	
	for s := range mapA {
		if !mapB[s] {
			return false
		}
	}
	
	return true
}

// Benchmark tests for performance validation
func BenchmarkFindMatches(b *testing.B) {
	config := &Config{
		Words: []string{"BREAKING", "NEWS", "URGENT", "ALERT", "LIVE"},
	}
	
	logger := slog.New(slog.NewTextHandler(os.Stdout, nil))
	detector := &Detector{
		config: config,
		logger: logger,
	}

	text := strings.Repeat("This is a BREAKING NEWS story with URGENT updates and LIVE coverage. ", 100)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = detector.findMatches(text)
	}
}