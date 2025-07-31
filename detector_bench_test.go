package main

import (
	"context"
	"fmt"
	"log/slog"
	"runtime"
	"sync"
	"testing"
	"time"

	"gocv.io/x/gocv"
)

// BenchmarkWorkerPool benchmarks the performance of the OCR worker pool
// against single-threaded processing to demonstrate performance improvements.
func BenchmarkWorkerPool(b *testing.B) {
	// Skip if no OpenCV/Tesseract available
	if testing.Short() {
		b.Skip("Skipping benchmark in short mode")
	}

	// Create a test configuration
	config := &Config{
		URL:        "test",
		Words:      []string{"test", "sample", "benchmark"},
		Language:   "eng",
		Confidence: 0.80,
	}

	logger := slog.New(slog.NewTextHandler(nil, &slog.HandlerOptions{Level: slog.LevelError}))

	// Create test frames - using synthetic data since we don't have a real stream
	testFrames := make([]Frame, 100)
	for i := 0; i < 100; i++ {
		// Create a simple test image (100x100 white image)
		img := gocv.NewMatWithSize(100, 100, gocv.MatTypeCV8UC3)
		img.SetTo(gocv.NewScalar(255, 255, 255, 0)) // White background
		
		testFrames[i] = Frame{
			Image:     img,
			Index:     int64(i + 1),
			Timestamp: time.Now(),
		}
	}

	defer func() {
		// Clean up test frames
		for _, frame := range testFrames {
			if !frame.Image.Empty() {
				frame.Image.Close()
			}
		}
	}()

	b.Run("ParallelWorkerPool", func(b *testing.B) {
		b.ResetTimer()
		
		for i := 0; i < b.N; i++ {
			// Create detector with worker pool
			detector := &Detector{
				config:           config,
				logger:           logger,
				metrics:          &StreamMetrics{},
				frameBufferSize:  runtime.NumCPU() * 10,
				resultBufferSize: runtime.NumCPU() * 5,
			}

			// Create OCR worker pool
			workerPool, err := NewOCRWorkerPool(detector)
			if err != nil {
				b.Fatalf("Failed to create worker pool: %v", err)
			}
			detector.ocrWorkerPool = workerPool

			// Create channels
			frameChan := make(chan Frame, detector.frameBufferSize)
			resultChan := make(chan DetectionResult, detector.resultBufferSize)

			// Create context with timeout
			ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
			
			var wg sync.WaitGroup

			// Start worker pool
			wg.Add(1)
			go func() {
				defer wg.Done()
				defer close(resultChan)
				workerPool.ProcessFrames(ctx, frameChan, resultChan)
			}()

			// Start result consumer
			wg.Add(1)
			go func() {
				defer wg.Done()
				for range resultChan {
					// Consume results
				}
			}()

			// Send test frames
			go func() {
				defer close(frameChan)
				for _, frame := range testFrames {
					select {
					case frameChan <- frame:
					case <-ctx.Done():
						return
					}
				}
			}()

			// Wait for completion
			wg.Wait()
			cancel()
			
			// Clean up
			workerPool.Close()
		}
	})

	b.Run("SingleThreaded", func(b *testing.B) {
		b.ResetTimer()
		
		for i := 0; i < b.N; i++ {
			// Create detector
			detector := &Detector{
				config:  config,
				logger:  logger,
				metrics: &StreamMetrics{},
			}

			// Create single OCR worker
			worker, err := NewOCRWorker(0, detector)
			if err != nil {
				b.Fatalf("Failed to create worker: %v", err)
			}

			// Process frames sequentially
			for _, frame := range testFrames {
				_ = worker.processFrame(frame)
			}

			// Clean up
			worker.Close()
		}
	})
}

// BenchmarkBufferSizes benchmarks different buffer sizes to show optimal configuration.
func BenchmarkBufferSizes(b *testing.B) {
	if testing.Short() {
		b.Skip("Skipping benchmark in short mode")
	}

	bufferSizes := []int{10, 20, 50, 100, 200}
	
	for _, size := range bufferSizes {
		b.Run(fmt.Sprintf("BufferSize_%d", size), func(b *testing.B) {
			b.ResetTimer()
			
			for i := 0; i < b.N; i++ {
				// Create channels with different buffer sizes
				frameChan := make(chan Frame, size)
				
				// Producer
				go func() {
					for j := 0; j < 50; j++ {
						frameChan <- Frame{Index: int64(j)}
					}
					close(frameChan)
				}()

				// Consumer
				count := 0
				for range frameChan {
					count++
				}
			}
		})
	}
}

// BenchmarkMetricsUpdate benchmarks the performance impact of metrics tracking.
func BenchmarkMetricsUpdate(b *testing.B) {
	metrics := &StreamMetrics{}
	
	b.Run("MetricsEnabled", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			metrics.framesProcessed.Add(1)
			metrics.UpdateProcessingTime(time.Millisecond * 100)
			metrics.UpdateBufferUtilization(75)
		}
	})
	
	b.Run("NoMetrics", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			// Simulate processing without metrics
			_ = i * 2
		}
	})
}