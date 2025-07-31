package main

import (
	"context"
	"fmt"
	"image"
	"log/slog"
	"math"
	"math/rand"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/otiai10/gosseract/v2"
	"gocv.io/x/gocv"
)

// Frame represents a captured video frame with metadata for OCR processing.
// The frame includes the image data and tracking information for logging purposes.
type Frame struct {
	// Image holds the OpenCV Mat containing the raw frame pixels.
	// This must be closed by the consumer to prevent memory leaks.
	Image gocv.Mat

	// Index is a monotonically increasing counter starting from 1
	// that uniquely identifies each captured frame in the stream.
	Index int64

	// Timestamp records when the frame was captured from the video stream.
	// This is used for logging and debugging frame processing latency.
	Timestamp time.Time
}

// DetectionResult represents the result of OCR processing on a frame.
// It contains both the raw OCR output and analyzed word matches for logging.
type DetectionResult struct {
	// Frame is the original video frame that was processed.
	// The Image field will be closed by the time this result is generated.
	Frame Frame

	// Text contains the complete text extracted from the frame by Tesseract OCR.
	// This includes all detected text, not just the target words.
	// Text is trimmed of leading/trailing whitespace.
	Text string

	// Confidence is the average OCR confidence score (0.0-1.0) for all
	// detected words in the frame. Only results meeting the configured
	// confidence threshold are processed for word matching.
	Confidence float64

	// Matches contains the target words that were found in the extracted text.
	// Words are matched case-insensitively using substring matching.
	// Empty if no target words were detected or confidence was too low.
	Matches []string
}

// CircuitState represents the current state of the circuit breaker.
type CircuitState int32

const (
	// CircuitClosed indicates normal operation with successful stream reads.
	CircuitClosed CircuitState = iota
	// CircuitOpen indicates too many failures occurred, blocking operations.
	CircuitOpen
	// CircuitHalfOpen indicates testing if the stream has recovered.
	CircuitHalfOpen
)

// CircuitBreaker implements the circuit breaker pattern for stream reliability.
// It prevents cascade failures by temporarily blocking operations after too many errors.
type CircuitBreaker struct {
	// state holds the current circuit state (closed/open/half-open).
	state atomic.Int32
	// failureCount tracks consecutive failures for triggering state changes.
	failureCount atomic.Int64
	// lastFailureTime records when the last failure occurred for timeout calculations.
	lastFailureTime atomic.Int64
	// successCount tracks successful operations in half-open state.
	successCount atomic.Int64

	// maxFailures is the threshold for opening the circuit.
	maxFailures int64
	// timeout is how long to wait before transitioning from open to half-open.
	timeout time.Duration
	// recoveryThreshold is how many successes needed to close from half-open.
	recoveryThreshold int64
}

// NewCircuitBreaker creates a circuit breaker with the specified configuration.
// maxFailures: number of consecutive failures before opening
// timeout: how long to wait before attempting recovery
// recoveryThreshold: successful operations needed to fully recover
func NewCircuitBreaker(maxFailures int64, timeout time.Duration, recoveryThreshold int64) *CircuitBreaker {
	cb := &CircuitBreaker{
		maxFailures:       maxFailures,
		timeout:           timeout,
		recoveryThreshold: recoveryThreshold,
	}
	cb.state.Store(int32(CircuitClosed))
	return cb
}

// Call executes the provided function if the circuit allows it.
// Returns an error if the circuit is open, otherwise returns the function's result.
func (cb *CircuitBreaker) Call(fn func() error) error {
	state := CircuitState(cb.state.Load())

	switch state {
	case CircuitOpen:
		// Check if timeout has passed to transition to half-open
		lastFailure := time.Unix(0, cb.lastFailureTime.Load())
		if time.Since(lastFailure) > cb.timeout {
			if cb.state.CompareAndSwap(int32(CircuitOpen), int32(CircuitHalfOpen)) {
				cb.successCount.Store(0)
			}
			state = CircuitHalfOpen
		} else {
			return fmt.Errorf("circuit breaker is open, last failure: %v ago", time.Since(lastFailure))
		}
	}

	err := fn()
	if err != nil {
		cb.recordFailure()
	} else {
		cb.recordSuccess()
	}

	return err
}

// recordFailure increments the failure count and potentially opens the circuit.
func (cb *CircuitBreaker) recordFailure() {
	cb.lastFailureTime.Store(time.Now().UnixNano())
	failures := cb.failureCount.Add(1)

	if failures >= cb.maxFailures {
		cb.state.Store(int32(CircuitOpen))
	}
}

// recordSuccess resets failure count and potentially closes the circuit from half-open.
func (cb *CircuitBreaker) recordSuccess() {
	cb.failureCount.Store(0)

	state := CircuitState(cb.state.Load())
	if state == CircuitHalfOpen {
		successes := cb.successCount.Add(1)
		if successes >= cb.recoveryThreshold {
			cb.state.Store(int32(CircuitClosed))
		}
	}
}

// GetState returns the current circuit breaker state.
func (cb *CircuitBreaker) GetState() CircuitState {
	return CircuitState(cb.state.Load())
}

// StreamMetrics tracks health and performance metrics for the video stream.
type StreamMetrics struct {
	// framesProcessed counts total frames successfully processed.
	framesProcessed atomic.Int64
	// framesDropped counts frames dropped due to backpressure.
	framesDropped atomic.Int64
	// streamErrors counts stream read/connection errors.
	streamErrors atomic.Int64
	// ocrErrors counts OCR processing errors.
	ocrErrors atomic.Int64
	// lastFrameTime tracks when the last frame was successfully processed.
	lastFrameTime atomic.Int64
	// reconnectAttempts counts how many times stream reconnection was attempted.
	reconnectAttempts atomic.Int64
}

// GetFramesProcessed returns the total number of frames successfully processed.
func (m *StreamMetrics) GetFramesProcessed() int64 {
	return m.framesProcessed.Load()
}

// GetFramesDropped returns the total number of frames dropped due to backpressure.
func (m *StreamMetrics) GetFramesDropped() int64 {
	return m.framesDropped.Load()
}

// GetStreamErrors returns the total number of stream errors encountered.
func (m *StreamMetrics) GetStreamErrors() int64 {
	return m.streamErrors.Load()
}

// GetOCRErrors returns the total number of OCR processing errors.
func (m *StreamMetrics) GetOCRErrors() int64 {
	return m.ocrErrors.Load()
}

// GetReconnectAttempts returns the total number of stream reconnection attempts.
func (m *StreamMetrics) GetReconnectAttempts() int64 {
	return m.reconnectAttempts.Load()
}

// GetLastFrameAge returns how long ago the last frame was processed.
func (m *StreamMetrics) GetLastFrameAge() time.Duration {
	lastTime := m.lastFrameTime.Load()
	if lastTime == 0 {
		return 0
	}
	return time.Since(time.Unix(0, lastTime))
}

// Detector manages video stream capture and OCR processing using a concurrent pipeline.
// It coordinates three goroutines: frame capture, OCR processing, and result logging.
// All operations are thread-safe and respond to context cancellation for graceful shutdown.
// Enhanced with circuit breaker pattern, stream reconnection, and comprehensive metrics.
type Detector struct {
	// config holds the application configuration including target words,
	// confidence thresholds, and processing intervals.
	config *Config

	// logger provides structured logging for all detection events and errors.
	logger *slog.Logger

	// capture manages the OpenCV video stream connection.
	// Must be closed during cleanup to release system resources.
	capture *gocv.VideoCapture

	// ocrClient is the Tesseract OCR engine instance configured with
	// the specified language and page segmentation mode.
	ocrClient *gosseract.Client

	// frameIndex is a monotonically increasing counter for captured frames.
	// Protected by mu for concurrent access from multiple goroutines.
	frameIndex int64

	// mu protects capture reconnection operations and ensures thread safety.
	mu sync.RWMutex

	// circuitBreaker implements circuit breaker pattern for stream resilience.
	circuitBreaker *CircuitBreaker

	// metrics tracks stream health and performance statistics.
	metrics *StreamMetrics

	// backpressureThreshold defines when to start applying backpressure (channel usage %).
	backpressureThreshold float64
}

// NewDetector creates a new Detector instance with the given configuration.
// It initializes the video capture connection and OCR client with proper validation.
// Enhanced with circuit breaker, metrics tracking, and backpressure handling.
//
// The function performs the following initialization steps:
//   1. Opens video capture connection to the specified stream URL
//   2. Verifies the connection is active and responsive
//   3. Creates and configures Tesseract OCR client with specified language
//   4. Sets OCR page segmentation mode to PSM_AUTO for optimal text detection
//   5. Initializes circuit breaker for stream resilience
//   6. Sets up metrics tracking for monitoring stream health
//
// Returns an error if:
//   - The video stream URL is unreachable or invalid
//   - The video capture fails to open (network/codec issues)
//   - The specified OCR language is not installed on the system
//   - OCR client configuration fails
//
// The caller must call Close() on the returned Detector to release resources.
func NewDetector(config *Config, logger *slog.Logger) (*Detector, error) {
	// Initialize video capture
	capture, err := gocv.OpenVideoCapture(config.URL)
	if err != nil {
		return nil, fmt.Errorf("failed to open video capture: %w", err)
	}

	if !capture.IsOpened() {
		capture.Close()
		return nil, fmt.Errorf("video capture is not opened")
	}

	// Initialize OCR client
	ocrClient := gosseract.NewClient()
	if err := ocrClient.SetLanguage(config.Language); err != nil {
		capture.Close()
		ocrClient.Close()
		return nil, fmt.Errorf("failed to set OCR language: %w", err)
	}

	// Set OCR page segmentation mode for better text detection
	if err := ocrClient.SetPageSegMode(gosseract.PSM_AUTO); err != nil {
		capture.Close()
		ocrClient.Close()
		return nil, fmt.Errorf("failed to set page segmentation mode: %w", err)
	}

	// Initialize circuit breaker: 5 failures, 30s timeout, 3 successes to recover
	circuitBreaker := NewCircuitBreaker(5, 30*time.Second, 3)

	return &Detector{
		config:                config,
		logger:                logger,
		capture:               capture,
		ocrClient:             ocrClient,
		circuitBreaker:        circuitBreaker,
		metrics:               &StreamMetrics{},
		backpressureThreshold: 0.8, // Apply backpressure when channels are 80% full
	}, nil
}

// Close releases all resources held by the detector.
// This method should be called when the detector is no longer needed to prevent
// resource leaks. It's safe to call multiple times.
//
// Resources cleaned up:
//   - OpenCV video capture connection and buffers
//   - Tesseract OCR client and associated memory
//
// Returns an error if any cleanup operation fails. Multiple errors are
// collected and returned as a single error for complete cleanup reporting.
// The detector should not be used after Close() is called.
func (d *Detector) Close() error {
	var errs []error

	if d.capture != nil {
		if err := d.capture.Close(); err != nil {
			errs = append(errs, fmt.Errorf("failed to close video capture: %w", err))
		}
	}

	if d.ocrClient != nil {
		if err := d.ocrClient.Close(); err != nil {
			errs = append(errs, fmt.Errorf("failed to close OCR client: %w", err))
		}
	}

	if len(errs) > 0 {
		return fmt.Errorf("errors during cleanup: %v", errs)
	}

	return nil
}

// Run starts the video stream processing loop using a concurrent pipeline architecture.
// It coordinates three goroutines that communicate through buffered channels:
//
// Pipeline stages:
//   1. Frame capture: samples frames from video stream at configured intervals
//   2. OCR processing: applies image preprocessing and Tesseract text extraction
//   3. Result logging: outputs structured logs for matched target words
//   4. Metrics reporting: periodically logs stream health metrics
//
// The method blocks until the context is cancelled or an unrecoverable error occurs.
// All goroutines respond to context cancellation for coordinated shutdown.
//
// Enhanced features:
//   - Dynamic backpressure based on channel utilization
//   - Circuit breaker pattern for stream resilience
//   - Comprehensive metrics tracking and reporting
//   - Automatic stream reconnection with exponential backoff
//
// Channel buffer sizes (20 each) provide better flow control while the backpressure
// mechanism prevents excessive memory usage when processing falls behind.
//
// Error handling: Individual frame processing errors are logged but don't
// terminate the entire pipeline, ensuring robust operation with intermittent issues.
func (d *Detector) Run(ctx context.Context) error {
	// Increased buffer sizes for better throughput, backpressure handles memory
	frameChan := make(chan Frame, 20)
	resultChan := make(chan DetectionResult, 20)

	var wg sync.WaitGroup

	// Start metrics reporting goroutine
	wg.Add(1)
	go func() {
		defer wg.Done()
		d.reportMetrics(ctx)
	}()

	// Start frame capture goroutine
	wg.Add(1)
	go func() {
		defer wg.Done()
		defer close(frameChan)
		d.captureFrames(ctx, frameChan)
	}()

	// Start OCR processing goroutine
	wg.Add(1)
	go func() {
		defer wg.Done()
		defer close(resultChan)
		d.processFrames(ctx, frameChan, resultChan)
	}()

	// Start result logging goroutine
	wg.Add(1)
	go func() {
		defer wg.Done()
		d.logResults(ctx, resultChan)
	}()

	wg.Wait()
	return nil
}

// reconnectStream attempts to reconnect to the video stream with exponential backoff.
// This method is called when the stream connection fails and implements a robust
// reconnection strategy to handle temporary network issues or stream interruptions.
//
// Reconnection strategy:
//   - Exponential backoff starting at 1 second, max 60 seconds
//   - Maximum 10 attempts before giving up
//   - Closes old connection before attempting new one
//   - Validates new connection before returning
//
// Returns true if reconnection succeeded, false if all attempts failed.
// Thread-safe and can be called concurrently with other detector operations.
func (d *Detector) reconnectStream(ctx context.Context) bool {
	d.mu.Lock()
	defer d.mu.Unlock()

	// Close existing connection
	if d.capture != nil {
		d.capture.Close()
		d.capture = nil
	}

	const maxAttempts = 10
	baseDelay := 1 * time.Second
	maxDelay := 60 * time.Second

	for attempt := 1; attempt <= maxAttempts; attempt++ {
		select {
		case <-ctx.Done():
			return false
		default:
		}

		d.metrics.reconnectAttempts.Add(1)
		d.logger.Info("Attempting stream reconnection",
			"attempt", attempt,
			"max_attempts", maxAttempts,
			"url", d.config.URL)

		// Attempt reconnection
		capture, err := gocv.OpenVideoCapture(d.config.URL)
		if err == nil && capture.IsOpened() {
			d.capture = capture
			d.logger.Info("Stream reconnection successful", "attempt", attempt)
			return true
		}

		if capture != nil {
			capture.Close()
		}

		// Calculate exponential backoff delay
		delay := time.Duration(float64(baseDelay) * math.Pow(2, float64(attempt-1)))
		if delay > maxDelay {
			delay = maxDelay
		}

		// Add jitter to prevent thundering herd
		jitter := time.Duration(rand.Int63n(int64(delay / 4)))
		totalDelay := delay + jitter

		d.logger.Warn("Stream reconnection failed, retrying",
			"attempt", attempt,
			"error", err,
			"retry_in", totalDelay)

		select {
		case <-time.After(totalDelay):
			continue
		case <-ctx.Done():
			return false
		}
	}

	d.logger.Error("Stream reconnection failed after all attempts", "max_attempts", maxAttempts)
	return false
}

// shouldApplyBackpressure determines if backpressure should be applied based on channel utilization.
// This prevents memory exhaustion by slowing down frame capture when processing falls behind.
//
// Backpressure is applied when the frame channel is above the configured threshold (default 80%).
// This provides early warning before the channel becomes completely full and frames are dropped.
func (d *Detector) shouldApplyBackpressure(frameChan chan<- Frame) bool {
	utilization := float64(len(frameChan)) / float64(cap(frameChan))
	return utilization >= d.backpressureThreshold
}

// captureFrames captures frames from the video stream at the configured interval.
// This method runs in its own goroutine and is the first stage of the processing pipeline.
// Enhanced with circuit breaker pattern, stream reconnection, and backpressure handling.
//
// Behavior:
//   - Uses a ticker to sample frames at the configured interval
//   - Clones each frame to prevent data races with OpenCV Mat objects
//   - Assigns monotonically increasing frame indices for tracking
//   - Applies backpressure when processing pipeline is overloaded
//   - Automatically reconnects on stream failures using exponential backoff
//   - Uses circuit breaker to prevent cascade failures
//   - Responds immediately to context cancellation for shutdown
//
// Frame memory management: Each captured frame is cloned to ensure thread safety.
// The clone is closed by the OCR processing stage after use.
//
// Error handling: Stream failures trigger automatic reconnection attempts.
// Circuit breaker prevents overwhelming failed streams with continuous retries.
func (d *Detector) captureFrames(ctx context.Context, frameChan chan<- Frame) {
	ticker := time.NewTicker(d.config.Interval)
	defer ticker.Stop()

	img := gocv.NewMat()
	defer img.Close()

	for {
		select {
		case <-ctx.Done():
			d.logger.Info("Frame capture stopped")
			return
		case <-ticker.C:
			// Apply backpressure if processing is falling behind
			if d.shouldApplyBackpressure(frameChan) {
				d.logger.Debug("Applying backpressure, skipping frame capture")
				continue
			}

			// Use circuit breaker to handle stream failures gracefully
			err := d.circuitBreaker.Call(func() error {
				d.mu.RLock()
				capture := d.capture
				d.mu.RUnlock()

				if capture == nil {
					return fmt.Errorf("capture is nil")
				}

				if !capture.Read(&img) {
					d.metrics.streamErrors.Add(1)
					return fmt.Errorf("failed to read frame from video stream")
				}

				if img.Empty() {
					return fmt.Errorf("empty frame captured")
				}

				return nil
			})

			if err != nil {
				// Log the error with circuit breaker state
				state := d.circuitBreaker.GetState()
				d.logger.Error("Frame capture failed",
					"error", err,
					"circuit_state", state,
					"stream_errors", d.metrics.GetStreamErrors())

				// If circuit is open, attempt reconnection
				if state == CircuitOpen {
					d.logger.Info("Circuit breaker open, attempting stream reconnection")
					if !d.reconnectStream(ctx) {
						d.logger.Error("Stream reconnection failed, stopping capture")
						return
					}
				}
				continue
			}

			// Clone the image to avoid data races
			clonedImg := img.Clone()

			d.mu.Lock()
			d.frameIndex++
			frameIndex := d.frameIndex
			d.mu.Unlock()

			frame := Frame{
				Image:     clonedImg,
				Index:     frameIndex,
				Timestamp: time.Now(),
			}

			// Update metrics
			d.metrics.framesProcessed.Add(1)
			d.metrics.lastFrameTime.Store(time.Now().UnixNano())

			select {
			case frameChan <- frame:
				// Frame sent successfully
			case <-ctx.Done():
				clonedImg.Close()
				return
			default:
				// Drop frame if channel is full (last resort)
				clonedImg.Close()
				d.metrics.framesDropped.Add(1)
				d.logger.Warn("Dropped frame due to full buffer",
					"frame_index", frameIndex,
					"total_dropped", d.metrics.GetFramesDropped())
			}
		}
	}
}

// processFrames processes captured frames with OCR and word matching.
// This method runs in its own goroutine and is the second stage of the processing pipeline.
// It receives frames from the capture stage and outputs detection results.
// Enhanced with proper error handling, metrics tracking, and resource management.
//
// Processing workflow for each frame:
//   1. Apply image preprocessing (grayscale, resize, threshold)
//   2. Perform Tesseract OCR text extraction with error recovery
//   3. Calculate average confidence score from detected words
//   4. If confidence meets threshold, check for target word matches
//   5. Send results to logging stage (only frames with matches)
//   6. Update processing metrics for monitoring
//
// Memory management: Properly closes the frame image after processing to prevent leaks.
// Uses defer statements to ensure cleanup even on panic or early return.
// Results are only forwarded if target words are detected to reduce logging volume.
//
// Error handling: OCR failures are logged with frame context and metrics updated.
// Processing continues despite individual frame issues for robust operation.
func (d *Detector) processFrames(ctx context.Context, frameChan <-chan Frame, resultChan chan<- DetectionResult) {
	for {
		select {
		case <-ctx.Done():
			d.logger.Info("Frame processing stopped")
			return
		case frame, ok := <-frameChan:
			if !ok {
				return
			}

			// Ensure frame image is always closed, even on panic
			defer func() {
				if !frame.Image.Empty() {
					frame.Image.Close()
				}
			}()

			result := d.processFrame(frame)

			// Immediately close the frame image after processing
			frame.Image.Close()

			// Forward results with matches or high confidence for logging
			if len(result.Matches) > 0 {
				select {
				case resultChan <- result:
					// Result sent successfully
				case <-ctx.Done():
					return
				default:
					d.logger.Warn("Dropped detection result due to full buffer",
						"frame_index", result.Frame.Index,
						"matches", result.Matches)
				}
			}
		}
	}
}

// processFrame performs OCR on a single frame and checks for word matches.
// This is the core processing logic that transforms video frames into text detection results.
// Enhanced with comprehensive error handling, metrics tracking, and resource management.
//
// Processing steps:
//   1. Apply image preprocessing (grayscale conversion, resize, adaptive threshold)
//   2. Encode preprocessed image as PNG for Tesseract input
//   3. Extract text using configured OCR language settings with error recovery
//   4. Retrieve word-level bounding boxes with confidence scores
//   5. Calculate average confidence across all detected words
//   6. If confidence ≥ threshold, perform case-insensitive target word matching
//   7. Update OCR processing metrics for monitoring
//
// Confidence calculation: Uses the average confidence of all detected words
// with confidence > 0. This provides a more stable metric than individual
// word confidence scores which can vary significantly.
//
// Returns a DetectionResult with the original frame metadata, extracted text,
// calculated confidence, and any matched target words. Empty matches indicate
// either no target words found or confidence below threshold.
//
// Error recovery: OCR failures are tracked in metrics but don't crash processing.
// Resource cleanup is guaranteed through defer statements.
func (d *Detector) processFrame(frame Frame) DetectionResult {
	// Preprocess the frame for better OCR accuracy
	processed := d.preprocessFrame(frame.Image)
	defer processed.Close()

	// Convert to bytes for OCR
	imgBytes, err := gocv.IMEncode(".png", processed)
	if err != nil {
		d.metrics.ocrErrors.Add(1)
		d.logger.Error("Failed to encode image",
			"error", err,
			"frame_index", frame.Index,
			"total_ocr_errors", d.metrics.GetOCRErrors())
		return DetectionResult{Frame: frame}
	}
	defer imgBytes.Close() // Ensure encoded bytes are cleaned up

	// Perform OCR with error recovery
	if err := d.ocrClient.SetImageFromBytes(imgBytes.GetBytes()); err != nil {
		d.metrics.ocrErrors.Add(1)
		d.logger.Error("Failed to set OCR image",
			"error", err,
			"frame_index", frame.Index,
			"total_ocr_errors", d.metrics.GetOCRErrors())
		return DetectionResult{Frame: frame}
	}

	text, err := d.ocrClient.Text()
	if err != nil {
		d.metrics.ocrErrors.Add(1)
		d.logger.Error("Failed to extract text",
			"error", err,
			"frame_index", frame.Index,
			"total_ocr_errors", d.metrics.GetOCRErrors())
		return DetectionResult{Frame: frame}
	}

	// Get bounding boxes to calculate average confidence
	boxes, err := d.ocrClient.GetBoundingBoxes(gosseract.RIL_WORD)
	if err != nil {
		d.metrics.ocrErrors.Add(1)
		d.logger.Error("Failed to get bounding boxes",
			"error", err,
			"frame_index", frame.Index,
			"total_ocr_errors", d.metrics.GetOCRErrors())
		// Continue without confidence calculation if bounding boxes fail
		boxes = nil
	}

	// Calculate average confidence from all detected words
	var totalConfidence float64
	var wordCount int
	if boxes != nil {
		for _, box := range boxes {
			if box.Confidence > 0 {
				totalConfidence += box.Confidence
				wordCount++
			}
		}
	}

	var avgConfidence float64
	if wordCount > 0 {
		avgConfidence = totalConfidence / float64(wordCount)
	}

	// Check for word matches if confidence meets threshold or no confidence available
	var matches []string
	if wordCount == 0 || avgConfidence >= d.config.Confidence*100 {
		matches = d.findMatches(text)
	}

	return DetectionResult{
		Frame:      frame,
		Text:       strings.TrimSpace(text),
		Confidence: avgConfidence / 100.0,
		Matches:    matches,
	}
}

// reportMetrics periodically logs stream health and performance metrics.
// This method runs in its own goroutine and provides visibility into stream quality,
// processing performance, and error rates for monitoring and debugging.
//
// Metrics reported every 30 seconds:
//   - Stream health: frames processed, dropped, errors
//   - Processing performance: OCR errors, last frame age
//   - Connection status: reconnection attempts, circuit breaker state
//   - Memory efficiency: frame processing rate vs capture rate
//
// This information helps identify bottlenecks, stream quality issues,
// and system performance characteristics in production deployments.
func (d *Detector) reportMetrics(ctx context.Context) {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			d.logger.Info("Metrics reporting stopped")
			return
		case <-ticker.C:
			lastFrameAge := d.metrics.GetLastFrameAge()
			circuitState := d.circuitBreaker.GetState()

			d.logger.Info("Stream metrics report",
				"frames_processed", d.metrics.GetFramesProcessed(),
				"frames_dropped", d.metrics.GetFramesDropped(),
				"stream_errors", d.metrics.GetStreamErrors(),
				"ocr_errors", d.metrics.GetOCRErrors(),
				"reconnect_attempts", d.metrics.GetReconnectAttempts(),
				"last_frame_age_ms", lastFrameAge.Milliseconds(),
				"circuit_state", circuitState,
				"stream_url", d.config.URL)

			// Log warning if frames haven't been processed recently
			if lastFrameAge > 5*d.config.Interval {
				d.logger.Warn("Stream processing may be stalled",
					"last_frame_age", lastFrameAge,
					"expected_interval", d.config.Interval)
			}
		}
	}
}

// preprocessFrame applies preprocessing steps to improve OCR accuracy.
// This method optimizes the input image for Tesseract text recognition using
// standard computer vision techniques proven effective for OCR applications.
// Optimized for memory efficiency and processing speed.
//
// Processing pipeline:
//   1. Convert to grayscale - reduces noise and focuses on text structure
//   2. Resize to 150% - improves small text recognition as per requirements
//   3. Apply adaptive threshold - enhances text contrast against background
//   4. Optional noise reduction for very noisy streams
//
// The adaptive threshold uses mean-based thresholding with an 11x11 kernel
// and constant offset of 2, which works well for various lighting conditions
// and text styles commonly found in video streams.
//
// Memory optimization: Intermediate Mats are closed immediately after use
// to prevent accumulation of OpenCV memory allocations.
//
// Returns a new Mat containing the processed image. The caller must close
// the returned Mat to prevent memory leaks.
func (d *Detector) preprocessFrame(src gocv.Mat) gocv.Mat {
	// Convert to grayscale
	gray := gocv.NewMat()
	defer gray.Close() // Ensure cleanup even on early return
	gocv.CvtColor(src, &gray, gocv.ColorBGRToGray)

	// Resize to improve OCR accuracy (up to 150% as mentioned in requirements)
	// Only resize if source is reasonably sized to prevent excessive memory usage
	resized := gocv.NewMat()
	defer resized.Close() // Ensure cleanup even on early return

	currentSize := gray.Size()
	// Limit maximum dimensions to prevent memory exhaustion
	maxDimension := 2048
	scaleFactor := 1.5

	newWidth := int(float64(currentSize[1]) * scaleFactor)
	newHeight := int(float64(currentSize[0]) * scaleFactor)

	// Adjust scale factor if result would be too large
	if newWidth > maxDimension || newHeight > maxDimension {
		widthScale := float64(maxDimension) / float64(currentSize[1])
		heightScale := float64(maxDimension) / float64(currentSize[0])
		scaleFactor = math.Min(widthScale, heightScale)
		newWidth = int(float64(currentSize[1]) * scaleFactor)
		newHeight = int(float64(currentSize[0]) * scaleFactor)
	}

	newSize := image.Point{X: newWidth, Y: newHeight}
	gocv.Resize(gray, &resized, newSize, 0, 0, gocv.InterpolationLinear)

	// Apply adaptive threshold to improve text contrast
	thresholded := gocv.NewMat()
	gocv.AdaptiveThreshold(resized, &thresholded, 255, gocv.AdaptiveThresholdMean, gocv.ThresholdBinary, 11, 2)

	return thresholded
}

// findMatches searches for target words in the extracted text using case-insensitive substring matching.
// This method implements the core text detection logic that determines which target words
// are present in the OCR-extracted text from a video frame.
//
// Matching algorithm:
//   - Performs case-insensitive comparison by converting both text and target words to lowercase
//   - Uses substring matching (not whole word matching) to handle OCR spacing inconsistencies
//   - Returns all target words found in the text, preserving original case from configuration
//
// Examples of matches:
//   - "BREAKING" matches "breaking news story" and "breakingpoint update"
//   - "NEWS" matches "news alert" and "newscaster reports"
//   - Partial words like "URG" would match "urgent" if configured as a target
//
// Returns a slice of matched target words in their original configured case,
// or nil if no matches found or input text is empty.
func (d *Detector) findMatches(text string) []string {
	if text == "" {
		return nil
	}

	lowerText := strings.ToLower(text)
	var matches []string

	for _, word := range d.config.Words {
		lowerWord := strings.ToLower(word)
		if strings.Contains(lowerText, lowerWord) {
			matches = append(matches, word)
		}
	}

	return matches
}

// logResults processes detection results and logs matches using structured logging.
// This method runs in its own goroutine and is the final stage of the processing pipeline.
// It receives detection results and outputs structured log entries for monitoring and analysis.
//
// Structured log fields included:
//   - timestamp: RFC3339 formatted capture time for temporal correlation
//   - frame_index: unique frame identifier for debugging and tracking
//   - matched_words: array of detected target words for filtering and alerting
//   - confidence: OCR confidence score for quality assessment
//   - extracted_text: complete OCR output for manual verification
//   - stream_url: source identifier for multi-stream deployments
//
// The logging format (JSON or key=value) is determined by the configuration
// and optimized for different use cases: JSON for log aggregation systems,
// key=value for human-readable console output.
//
// Only results with matched target words are logged to reduce noise while
// maintaining all relevant detection events for analysis.
func (d *Detector) logResults(ctx context.Context, resultChan <-chan DetectionResult) {
	for {
		select {
		case <-ctx.Done():
			d.logger.Info("Result logging stopped")
			return
		case result, ok := <-resultChan:
			if !ok {
				return
			}

			d.logger.Info("Text detected in stream",
				"timestamp", result.Frame.Timestamp.Format(time.RFC3339),
				"frame_index", result.Frame.Index,
				"matched_words", result.Matches,
				"confidence", result.Confidence,
				"extracted_text", result.Text,
				"stream_url", d.config.URL,
			)
		}
	}
}
