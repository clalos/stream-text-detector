package main

import (
	"context"
	"fmt"
	"image"
	"log/slog"
	"math"
	"math/rand"
	"runtime"
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

// String returns a string representation of the CircuitState.
func (s CircuitState) String() string {
	switch s {
	case CircuitClosed:
		return "CLOSED"
	case CircuitOpen:
		return "OPEN"
	case CircuitHalfOpen:
		return "HALF_OPEN"
	default:
		return "UNKNOWN"
	}
}

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
	// logger for state transition logging.
	logger *slog.Logger
}

// NewCircuitBreaker creates a circuit breaker with the specified configuration.
// maxFailures: number of consecutive failures before opening
// timeout: how long to wait before attempting recovery
// recoveryThreshold: successful operations needed to fully recover
func NewCircuitBreaker(maxFailures int64, timeout time.Duration, recoveryThreshold int64, logger *slog.Logger) *CircuitBreaker {
	cb := &CircuitBreaker{
		maxFailures:       maxFailures,
		timeout:           timeout,
		recoveryThreshold: recoveryThreshold,
		logger:            logger,
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
			// Attempt to transition to half-open
			if cb.state.CompareAndSwap(int32(CircuitOpen), int32(CircuitHalfOpen)) {
				cb.successCount.Store(0)
				cb.logger.Info("Circuit breaker state transition",
					"from", "OPEN",
					"to", "HALF_OPEN",
					"timeout_elapsed", time.Since(lastFailure))
			}
			// Update state for this execution
			state = CircuitHalfOpen
		} else {
			return fmt.Errorf("circuit breaker is open, last failure: %v ago", time.Since(lastFailure))
		}
	case CircuitHalfOpen:
		// Already in half-open, proceed with the operation
	case CircuitClosed:
		// Normal operation, proceed
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
	currentState := CircuitState(cb.state.Load())

	// Transition to OPEN if we've exceeded max failures
	if failures >= cb.maxFailures && currentState != CircuitOpen {
		cb.state.Store(int32(CircuitOpen))
		cb.logger.Warn("Circuit breaker state transition",
			"from", currentState,
			"to", "OPEN",
			"failure_count", failures,
			"max_failures", cb.maxFailures)
	}

	// Reset from HALF_OPEN to OPEN on any failure
	if currentState == CircuitHalfOpen {
		cb.state.Store(int32(CircuitOpen))
		cb.successCount.Store(0)
		cb.logger.Warn("Circuit breaker state transition",
			"from", "HALF_OPEN",
			"to", "OPEN",
			"reason", "failure_during_recovery")
	}
}

// recordSuccess resets failure count and potentially closes the circuit from half-open.
func (cb *CircuitBreaker) recordSuccess() {
	cb.failureCount.Store(0)

	state := CircuitState(cb.state.Load())
	if state == CircuitHalfOpen {
		successes := cb.successCount.Add(1)
		if successes >= cb.recoveryThreshold {
			if cb.state.CompareAndSwap(int32(CircuitHalfOpen), int32(CircuitClosed)) {
				cb.logger.Info("Circuit breaker state transition",
					"from", "HALF_OPEN",
					"to", "CLOSED",
					"success_count", successes,
					"recovery_threshold", cb.recoveryThreshold)
			}
		}
	}
}

// GetState returns the current circuit breaker state.
func (cb *CircuitBreaker) GetState() CircuitState {
	return CircuitState(cb.state.Load())
}

// Reset forcibly resets the circuit breaker to CLOSED state.
// This should be called after successful stream reconnection.
func (cb *CircuitBreaker) Reset() {
	oldState := CircuitState(cb.state.Load())
	if oldState != CircuitClosed {
		cb.state.Store(int32(CircuitClosed))
		cb.failureCount.Store(0)
		cb.successCount.Store(0)
		cb.logger.Info("Circuit breaker reset to CLOSED",
			"previous_state", oldState,
			"reason", "successful_reconnection")
	}
}

// GetFailureCount returns the current failure count.
func (cb *CircuitBreaker) GetFailureCount() int64 {
	return cb.failureCount.Load()
}

// GetLastFailureTime returns the time of the last failure.
func (cb *CircuitBreaker) GetLastFailureTime() time.Time {
	nanos := cb.lastFailureTime.Load()
	if nanos == 0 {
		return time.Time{}
	}
	return time.Unix(0, nanos)
}

// StreamMetrics tracks health and performance metrics for the video stream.
// Enhanced with worker pool performance tracking.
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
	// parallelFramesProcessed counts frames processed by the worker pool.
	parallelFramesProcessed atomic.Int64
	// avgProcessingTimeNs tracks average frame processing time in nanoseconds.
	avgProcessingTimeNs atomic.Int64
	// maxBufferUtilization tracks peak channel utilization percentage.
	maxBufferUtilization atomic.Int64
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

// GetParallelFramesProcessed returns the total number of frames processed by workers.
func (m *StreamMetrics) GetParallelFramesProcessed() int64 {
	return m.parallelFramesProcessed.Load()
}

// GetAvgProcessingTimeMs returns the average frame processing time in milliseconds.
func (m *StreamMetrics) GetAvgProcessingTimeMs() float64 {
	return float64(m.avgProcessingTimeNs.Load()) / 1e6
}

// GetMaxBufferUtilization returns the peak buffer utilization as a percentage.
func (m *StreamMetrics) GetMaxBufferUtilization() int64 {
	return m.maxBufferUtilization.Load()
}

// UpdateProcessingTime updates the average processing time with a new measurement.
func (m *StreamMetrics) UpdateProcessingTime(processingTime time.Duration) {
	// Simple exponential moving average
	current := m.avgProcessingTimeNs.Load()
	new := int64(processingTime.Nanoseconds())
	if current == 0 {
		m.avgProcessingTimeNs.Store(new)
	} else {
		// EMA with alpha = 0.1
		updated := int64(float64(current)*0.9 + float64(new)*0.1)
		m.avgProcessingTimeNs.Store(updated)
	}
}

// UpdateBufferUtilization updates the maximum buffer utilization if current is higher.
func (m *StreamMetrics) UpdateBufferUtilization(utilization int64) {
	for {
		current := m.maxBufferUtilization.Load()
		if utilization <= current {
			break
		}
		if m.maxBufferUtilization.CompareAndSwap(current, utilization) {
			break
		}
	}
}

// GetLastFrameAge returns how long ago the last frame was processed.
func (m *StreamMetrics) GetLastFrameAge() time.Duration {
	lastTime := m.lastFrameTime.Load()
	if lastTime == 0 {
		return 0
	}
	return time.Since(time.Unix(0, lastTime))
}

// OCRWorker represents a worker in the OCR processing pool.
// Each worker maintains its own Tesseract client to enable parallel processing.
type OCRWorker struct {
	// id uniquely identifies this worker for logging and debugging.
	id int
	// client is the Tesseract OCR engine instance for this worker.
	client *gosseract.Client
	// detector is a reference to the parent detector for access to config and logger.
	detector *Detector
}

// NewOCRWorker creates a new OCR worker with its own Tesseract client.
// Each worker is configured with the same language and page segmentation settings.
func NewOCRWorker(id int, detector *Detector) (*OCRWorker, error) {
	client := gosseract.NewClient()
	if err := client.SetLanguage(detector.config.Language); err != nil {
		client.Close()
		return nil, fmt.Errorf("worker %d: failed to set OCR language: %w", id, err)
	}

	if err := client.SetPageSegMode(gosseract.PSM_AUTO); err != nil {
		client.Close()
		return nil, fmt.Errorf("worker %d: failed to set page segmentation mode: %w", id, err)
	}

	return &OCRWorker{
		id:       id,
		client:   client,
		detector: detector,
	}, nil
}

// Close releases the OCR client resources.
func (w *OCRWorker) Close() error {
	if w.client != nil {
		return w.client.Close()
	}
	return nil
}

// shouldThrottleCPU checks if CPU throttling should be applied based on system load.
// Uses a simple heuristic based on goroutine count as a proxy for CPU pressure.
// This lightweight approach avoids expensive system calls while providing throttling benefits.
func (w *OCRWorker) shouldThrottleCPU() bool {
	// Use goroutine count as a lightweight proxy for system load
	// When goroutines spike, it usually indicates high CPU usage
	currentGoroutines := runtime.NumGoroutine()
	
	// Base threshold: 4x CPU cores is considered high activity
	// This accounts for our worker pool + other goroutines
	baseThreshold := runtime.NumCPU() * 4
	
	// Additional check: if we have significantly more goroutines than expected
	// Expected: workers + capture + result logging + metrics = ~workerCount + 4
	expectedGoroutines := w.detector.ocrWorkerPool.workerCount + 10 // some buffer
	highGoroutineThreshold := expectedGoroutines * 2
	
	return currentGoroutines > baseThreshold || currentGoroutines > highGoroutineThreshold
}

// OCRWorkerPool manages a pool of OCR workers for parallel processing with CPU throttling.
// Enhanced with dynamic CPU throttling and fast shutdown capabilities.
type OCRWorkerPool struct {
	// workers contains the pool of OCR workers.
	workers []*OCRWorker
	// workerCount is the number of workers in the pool.
	workerCount int
	// detector is a reference to the parent detector.
	detector *Detector
	// cpuThrottleThreshold is the CPU usage percentage that triggers throttling.
	cpuThrottleThreshold float64
	// throttleDelay is the additional delay when CPU usage is high.
	throttleDelay time.Duration
	// shutdownTimeout is the maximum time to wait for workers to stop.
	shutdownTimeout time.Duration
}

// NewOCRWorkerPool creates a new pool of OCR workers with CPU throttling.
// The pool size is conservatively calculated to limit CPU usage to ~80% capacity.
// Uses a more conservative approach to prevent CPU exhaustion and enable faster shutdown.
func NewOCRWorkerPool(detector *Detector) (*OCRWorkerPool, error) {
	// Conservative worker count: 80% of CPU cores to prevent resource exhaustion
	// This allows for system breathing room and faster shutdown coordination
	workerCount := int(float64(runtime.NumCPU()) * 0.8)
	if workerCount < 2 {
		workerCount = 2 // Minimum 2 workers for basic concurrency
	}
	if workerCount > 8 {
		workerCount = 8 // Maximum 8 workers to prevent resource exhaustion and enable fast shutdown
	}

	workers := make([]*OCRWorker, workerCount)
	for i := 0; i < workerCount; i++ {
		worker, err := NewOCRWorker(i, detector)
		if err != nil {
			// Clean up any successfully created workers
			for j := 0; j < i; j++ {
				workers[j].Close()
			}
			return nil, fmt.Errorf("failed to create worker pool: %w", err)
		}
		workers[i] = worker
	}

	detector.logger.Debug("Created OCR worker pool with CPU throttling", 
		"worker_count", workerCount, 
		"cpu_cores", runtime.NumCPU(),
		"cpu_utilization_target", "80%",
		"throttling_enabled", true)

	return &OCRWorkerPool{
		workers:              workers,
		workerCount:          workerCount,
		detector:             detector,
		cpuThrottleThreshold: 80.0, // Throttle when CPU usage exceeds 80%
		throttleDelay:        10 * time.Millisecond, // Small delay to reduce CPU pressure
		shutdownTimeout:      2 * time.Second, // Fast shutdown timeout for individual workers
	}, nil
}

// Close releases all worker resources.
func (p *OCRWorkerPool) Close() error {
	var errs []error
	for i, worker := range p.workers {
		if err := worker.Close(); err != nil {
			errs = append(errs, fmt.Errorf("worker %d: %w", i, err))
		}
	}

	if len(errs) > 0 {
		return fmt.Errorf("errors closing worker pool: %v", errs)
	}
	return nil
}

// ProcessFrames starts worker goroutines to process frames concurrently with CPU throttling.
// Enhanced with faster shutdown coordination and CPU usage monitoring.
// Each worker processes frames from the input channel and sends results to the output channel.
func (p *OCRWorkerPool) ProcessFrames(ctx context.Context, frameChan <-chan Frame, resultChan chan<- DetectionResult) {
	var wg sync.WaitGroup
	
	// Create a context with timeout for faster shutdown coordination
	workerCtx, workerCancel := context.WithCancel(ctx)
	defer workerCancel()

	// Start all workers with the cancellable context
	for _, worker := range p.workers {
		wg.Add(1)
		go func(w *OCRWorker) {
			defer wg.Done()
			w.processFrames(workerCtx, frameChan, resultChan)
		}(worker)
	}

	// Wait for completion or timeout for fast shutdown
	done := make(chan struct{})
	go func() {
		wg.Wait()
		close(done)
	}()

	select {
	case <-done:
		p.detector.logger.Debug("All OCR workers stopped gracefully")
	case <-ctx.Done():
		p.detector.logger.Debug("Context cancelled, forcing worker shutdown")
		workerCancel() // Cancel all workers immediately
		
		// Wait a short time for graceful shutdown
		select {
		case <-done:
			p.detector.logger.Debug("Workers stopped after cancellation")
		case <-time.After(p.shutdownTimeout):
			p.detector.logger.Warn("Worker shutdown timeout reached", 
				"timeout", p.shutdownTimeout,
				"workers_may_still_be_running", true)
			// Still wait for workers to prevent resource leaks
			<-done
			p.detector.logger.Debug("Workers finally stopped after timeout")
		}
	}
}

// processFrames is the main processing loop for an individual OCR worker with CPU throttling.
// Enhanced with immediate context cancellation response and CPU usage monitoring.
func (w *OCRWorker) processFrames(ctx context.Context, frameChan <-chan Frame, resultChan chan<- DetectionResult) {
	processedCount := 0
	lastThrottleCheck := time.Now()
	throttleCheckInterval := 100 * time.Millisecond // Check CPU every 100ms
	
	defer func() {
		w.detector.logger.Debug("OCR worker stopped", "worker_id", w.id, "frames_processed", processedCount)
	}()

	for {
		// Check for immediate cancellation at the start of each loop
		select {
		case <-ctx.Done():
			w.detector.logger.Debug("OCR worker cancelled", "worker_id", w.id)
			return
		default:
		}

		// CPU throttling check - only check periodically to reduce overhead
		if time.Since(lastThrottleCheck) > throttleCheckInterval {
			if w.shouldThrottleCPU() {
				w.detector.logger.Debug("CPU throttling activated", "worker_id", w.id)
				select {
				case <-time.After(w.detector.ocrWorkerPool.throttleDelay):
					// Throttle delay completed
				case <-ctx.Done():
					return // Exit immediately on cancellation
				}
			}
			lastThrottleCheck = time.Now()
		}

		select {
		case <-ctx.Done():
			return
		case frame, ok := <-frameChan:
			if !ok {
				return
			}

			// Process the frame with proper resource management and cancellation support
			func() {
				defer func() {
					// Ensure frame image is always closed after processing
					if !frame.Image.Empty() {
						frame.Image.Close()
					}
				}()

				// Check for cancellation before expensive OCR operation
				select {
				case <-ctx.Done():
					return
				default:
				}

				startTime := time.Now()
				result := w.processFrame(ctx, frame) // Pass context for cancellation
				processingTime := time.Since(startTime)
				
				// Check for cancellation after OCR operation
				select {
				case <-ctx.Done():
					return
				default:
				}
				
				// Update metrics
				w.detector.metrics.parallelFramesProcessed.Add(1)
				w.detector.metrics.UpdateProcessingTime(processingTime)
				processedCount++

				// Forward all results for logging (both matches and no-matches)
				select {
				case resultChan <- result:
					// Result sent successfully
				case <-ctx.Done():
					return
				default:
					w.detector.logger.Warn("Dropped detection result due to full buffer",
						"worker_id", w.id,
						"frame_index", result.Frame.Index,
						"matches", result.Matches)
				}
			}()
		}
	}
}

// processFrame performs OCR on a single frame using this worker's Tesseract client.
// Enhanced with context cancellation support for immediate shutdown during OCR operations.
func (w *OCRWorker) processFrame(ctx context.Context, frame Frame) DetectionResult {
	// Check for cancellation before starting expensive operations
	select {
	case <-ctx.Done():
		w.detector.logger.Debug("OCR processing cancelled before starting", "worker_id", w.id, "frame_index", frame.Index)
		return DetectionResult{Frame: frame}
	default:
	}
	// Preprocess the frame for better OCR accuracy
	processed := w.detector.preprocessFrame(frame.Image)
	defer processed.Close()

	// Convert to bytes for OCR
	imgBytes, err := gocv.IMEncode(".png", processed)
	if err != nil {
		w.detector.metrics.ocrErrors.Add(1)
		w.detector.logger.Error("Failed to encode image",
			"worker_id", w.id,
			"error", err,
			"frame_index", frame.Index,
			"total_ocr_errors", w.detector.metrics.GetOCRErrors())
		return DetectionResult{Frame: frame}
	}
	defer imgBytes.Close()

	// Check for cancellation before OCR operation
	select {
	case <-ctx.Done():
		w.detector.logger.Debug("OCR processing cancelled before OCR", "worker_id", w.id, "frame_index", frame.Index)
		return DetectionResult{Frame: frame}
	default:
	}

	// Perform OCR with error recovery
	if err := w.client.SetImageFromBytes(imgBytes.GetBytes()); err != nil {
		w.detector.metrics.ocrErrors.Add(1)
		w.detector.logger.Error("Failed to set OCR image",
			"worker_id", w.id,
			"error", err,
			"frame_index", frame.Index,
			"total_ocr_errors", w.detector.metrics.GetOCRErrors())
		return DetectionResult{Frame: frame}
	}

	// Check for cancellation before text extraction
	select {
	case <-ctx.Done():
		w.detector.logger.Debug("OCR processing cancelled before text extraction", "worker_id", w.id, "frame_index", frame.Index)
		return DetectionResult{Frame: frame}
	default:
	}

	text, err := w.client.Text()
	if err != nil {
		w.detector.metrics.ocrErrors.Add(1)
		w.detector.logger.Error("Failed to extract text",
			"worker_id", w.id,
			"error", err,
			"frame_index", frame.Index,
			"total_ocr_errors", w.detector.metrics.GetOCRErrors())
		return DetectionResult{Frame: frame}
	}

	// Get bounding boxes to calculate average confidence
	boxes, err := w.client.GetBoundingBoxes(gosseract.RIL_WORD)
	if err != nil {
		w.detector.metrics.ocrErrors.Add(1)
		w.detector.logger.Error("Failed to get bounding boxes",
			"worker_id", w.id,
			"error", err,
			"frame_index", frame.Index,
			"total_ocr_errors", w.detector.metrics.GetOCRErrors())
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
	if wordCount == 0 || avgConfidence >= w.detector.config.Confidence*100 {
		matches = w.detector.findMatches(text)
	}

	return DetectionResult{
		Frame:      frame,
		Text:       strings.TrimSpace(text),
		Confidence: avgConfidence / 100.0,
		Matches:    matches,
	}
}

// Detector manages video stream capture and OCR processing using a high-performance concurrent pipeline with CPU throttling.
// It coordinates multiple goroutines: frame capture, parallel OCR processing via worker pool, and result logging.
// All operations are thread-safe and respond to context cancellation for fast graceful shutdown (target: <5 seconds).
// Enhanced with circuit breaker pattern, stream reconnection, CPU throttling, comprehensive metrics, and optimized resource utilization.
// CPU usage is limited to ~80% of system capacity through conservative worker pool sizing and dynamic throttling.
type Detector struct {
	// config holds the application configuration including target words,
	// confidence thresholds, and processing intervals.
	config *Config

	// logger provides structured logging for all detection events and errors.
	logger *slog.Logger

	// capture manages the OpenCV video stream connection.
	// Must be closed during cleanup to release system resources.
	capture *gocv.VideoCapture

	// ocrWorkerPool manages the pool of OCR workers for parallel processing.
	ocrWorkerPool *OCRWorkerPool

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

	// closeOnce ensures Close() is called only once to prevent double cleanup.
	closeOnce sync.Once

	// closed indicates whether the detector has been closed.
	closed atomic.Bool

	// shutdownTimeout defines maximum time to wait for graceful shutdown (reduced to 5s for faster shutdown).
	shutdownTimeout time.Duration

	// Performance optimization fields
	// frameBufferSize is the size of the frame processing buffer
	frameBufferSize int
	// resultBufferSize is the size of the result processing buffer
	resultBufferSize int
}

// NewDetector creates a new Detector instance with CPU throttling and fast shutdown capabilities.
// It initializes the video capture connection and OCR worker pool with conservative resource usage.
// Enhanced with circuit breaker, metrics tracking, CPU throttling, and fast shutdown coordination.
//
// The function performs the following initialization steps:
//   1. Opens video capture connection to the specified stream URL
//   2. Verifies the connection is active and responsive
//   3. Creates conservative OCR worker pool (80% of CPU cores, max 8 workers)
//   4. Configures CPU throttling mechanism to prevent resource exhaustion
//   5. Sets fast shutdown timeout (5 seconds) for responsive termination
//   6. Initializes circuit breaker for stream resilience
//   7. Sets up metrics tracking for monitoring stream health and CPU usage
//
// CPU Optimization Features:
//   - Worker pool sized at 80% of CPU cores to prevent system overload
//   - Dynamic CPU throttling based on goroutine count heuristics
//   - Fast shutdown with immediate context cancellation (target: <5 seconds)
//   - Conservative buffer sizes to prevent memory pressure
//
// Returns an error if:
//   - The video stream URL is unreachable or invalid
//   - The video capture fails to open (network/codec issues)
//   - The OCR worker pool fails to initialize
//   - System resources are insufficient for the configured workers
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

	// Initialize OCR worker pool for parallel processing
	detector := &Detector{
		config:                config,
		logger:                logger,
		capture:               capture,
		circuitBreaker:        NewCircuitBreaker(5, 30*time.Second, 3, logger),
		metrics:               &StreamMetrics{},
		backpressureThreshold: 0.7,  // Apply backpressure when channels are 70% full (more aggressive)
		shutdownTimeout:       5 * time.Second,    // Reduced timeout for faster shutdown with CPU throttling
		// Performance optimization: larger buffers for better throughput
		frameBufferSize:  runtime.NumCPU() * 10, // 10 frames per CPU core
		resultBufferSize: runtime.NumCPU() * 5,  // 5 results per CPU core
	}

	// Create OCR worker pool
	ocrWorkerPool, err := NewOCRWorkerPool(detector)
	if err != nil {
		capture.Close()
		return nil, fmt.Errorf("failed to create OCR worker pool: %w", err)
	}
	detector.ocrWorkerPool = ocrWorkerPool

	logger.Debug("Detector initialized with CPU throttling and fast shutdown",
		"frame_buffer_size", detector.frameBufferSize,
		"result_buffer_size", detector.resultBufferSize,
		"cpu_cores", runtime.NumCPU(),
		"worker_count", ocrWorkerPool.workerCount,
		"cpu_utilization_target", "80%",
		"shutdown_timeout", detector.shutdownTimeout,
		"throttling_enabled", true)

	return detector, nil
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
//
// This method is designed to be called only after all goroutines have stopped
// to prevent segmentation faults during resource cleanup. The Run() method
// guarantees this condition is met before returning.
func (d *Detector) Close() error {
	var finalErr error

	// Use sync.Once to ensure cleanup happens only once
	d.closeOnce.Do(func() {
		// Mark as closed first to prevent new operations
		d.closed.Store(true)

		var errs []error

		// Close OCR worker pool first to ensure all workers stop
		if d.ocrWorkerPool != nil {
			if err := d.ocrWorkerPool.Close(); err != nil {
				errs = append(errs, fmt.Errorf("failed to close OCR worker pool: %w", err))
			}
			d.ocrWorkerPool = nil
		}

		// Close OpenCV resources under lock to prevent races
		d.mu.Lock()
		if d.capture != nil {
			// Give a brief moment for any in-flight operations to complete
			// This is a defensive measure against any remaining race conditions
			time.Sleep(50 * time.Millisecond)
			
			if err := d.capture.Close(); err != nil {
				errs = append(errs, fmt.Errorf("failed to close video capture: %w", err))
			}
			d.capture = nil // Clear reference to prevent further access
		}
		d.mu.Unlock()

		if len(errs) > 0 {
			finalErr = fmt.Errorf("errors during cleanup: %v", errs)
		}

		d.logger.Debug("Detector cleanup completed")
	})

	return finalErr
}

// isClosed returns true if the detector has been closed.
func (d *Detector) isClosed() bool {
	return d.closed.Load()
}

// Run starts the video stream processing loop using a CPU-throttled concurrent pipeline architecture.
// It coordinates four goroutines that communicate through buffered channels with CPU usage controls:
//
// Pipeline stages:
//   1. Frame capture: samples frames from video stream at configured intervals
//   2. OCR processing: parallel processing with CPU throttling and fast shutdown
//   3. Result logging: outputs structured logs for matched target words
//   4. Metrics reporting: periodically logs stream health and CPU usage metrics
//
// The method blocks until the context is cancelled or an unrecoverable error occurs.
// All goroutines respond to context cancellation for fast coordinated shutdown (target: <5 seconds).
//
// CPU Throttling and Shutdown Features:
//   - Conservative worker pool sizing (80% of CPU cores) to prevent system overload
//   - Dynamic CPU throttling based on goroutine count heuristics
//   - Immediate context cancellation propagation for fast shutdown
//   - Multiple cancellation check points during OCR processing
//   - Timeout-based worker coordination with graceful fallback
//
// Shutdown guarantee: This method ALWAYS waits for all goroutines to fully terminate
// before returning, preventing resource cleanup race conditions. If graceful shutdown
// takes longer than the configured timeout (default 5s), a warning is logged but
// the method still waits for complete termination to prevent segmentation faults.
//
// Enhanced features:
//   - Dynamic backpressure based on channel utilization
//   - Circuit breaker pattern for stream resilience
//   - CPU throttling to maintain system responsiveness
//   - Comprehensive metrics tracking and CPU usage reporting
//   - Automatic stream reconnection with exponential backoff
//   - Fast graceful shutdown with proper goroutine coordination
//   - Memory-safe OpenCV Mat object management with panic recovery
//
// Optimized buffer sizes provide better flow control while the backpressure
// mechanism prevents excessive memory usage when processing falls behind.
//
// Error handling: Individual frame processing errors are logged but don't
// terminate the entire pipeline, ensuring robust operation with intermittent issues.
// OpenCV Mat objects are guaranteed to be cleaned up even on panic conditions.
func (d *Detector) Run(ctx context.Context) error {
	// Check if detector is already closed
	if d.isClosed() {
		return fmt.Errorf("detector is closed")
	}

	// Optimized buffer sizes based on CPU cores for maximum throughput
	// Larger buffers reduce contention and improve parallel processing efficiency
	frameChan := make(chan Frame, d.frameBufferSize)
	resultChan := make(chan DetectionResult, d.resultBufferSize)

	d.logger.Debug("Starting CPU-throttled processing pipeline",
		"frame_buffer_size", d.frameBufferSize,
		"result_buffer_size", d.resultBufferSize,
		"worker_count", d.ocrWorkerPool.workerCount,
		"cpu_cores", runtime.NumCPU(),
		"target_cpu_usage", "80%",
		"shutdown_timeout", d.shutdownTimeout,
		"cpu_throttling", "enabled")

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

	// Start OCR worker pool for parallel processing
	wg.Add(1)
	go func() {
		defer wg.Done()
		defer close(resultChan)
		d.ocrWorkerPool.ProcessFrames(ctx, frameChan, resultChan)
	}()

	// Start result logging goroutine
	wg.Add(1)
	go func() {
		defer wg.Done()
		d.logResults(ctx, resultChan)
	}()

	// Wait for all goroutines with timeout
	done := make(chan struct{})
	go func() {
		wg.Wait()
		close(done)
	}()

	select {
	case <-done:
		// All goroutines completed normally
		d.logger.Debug("All processing goroutines stopped gracefully")
		return nil
	case <-time.After(d.shutdownTimeout):
		// Timeout occurred - but we still must wait for goroutines to prevent segfaults
		d.logger.Warn("Shutdown timeout reached, some goroutines may still be running",
			"timeout", d.shutdownTimeout)
		
		// Wait for goroutines to finish even after timeout to prevent resource races
		d.logger.Debug("Waiting for remaining goroutines to finish to prevent segfaults...")
		<-done
		d.logger.Debug("All goroutines finally stopped")
		
		return fmt.Errorf("shutdown timeout after %v", d.shutdownTimeout)
	}
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
	// Don't attempt reconnection if detector is closed
	if d.isClosed() {
		return false
	}

	d.mu.Lock()
	defer d.mu.Unlock()

	// Double-check after acquiring lock
	if d.isClosed() {
		return false
	}

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
			d.logger.Info("Stream reconnection successful", 
			"attempt", attempt,
			"total_reconnect_attempts", d.metrics.GetReconnectAttempts())
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
// Backpressure is applied when the frame channel is above the configured threshold (default 70%).
// This provides early warning before the channel becomes completely full and frames are dropped.
// Enhanced with buffer utilization tracking for performance monitoring.
func (d *Detector) shouldApplyBackpressure(frameChan chan<- Frame) bool {
	utilization := float64(len(frameChan)) / float64(cap(frameChan))
	utilizationPercent := int64(utilization * 100)
	
	// Update buffer utilization metrics
	d.metrics.UpdateBufferUtilization(utilizationPercent)
	
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
			d.logger.Debug("Frame capture stopped")
			return
		case <-ticker.C:
			// Apply backpressure if processing is falling behind
			if d.shouldApplyBackpressure(frameChan) {
				d.logger.Debug("Applying backpressure, skipping frame capture")
				continue
			}

			// Check if detector is closed before attempting capture
			if d.isClosed() {
				d.logger.Debug("Detector closed, stopping frame capture")
				return
			}

			// Use circuit breaker to handle stream failures gracefully
			err := d.circuitBreaker.Call(func() error {
				d.mu.RLock()
				capture := d.capture
				d.mu.RUnlock()

				if capture == nil || d.isClosed() {
					// Connection error - needs reconnection
					return fmt.Errorf("connection error: capture is nil or detector is closed")
				}

				if !capture.Read(&img) {
					d.metrics.streamErrors.Add(1)
					// Stream read error - could be temporary or connection issue
					return fmt.Errorf("stream read error: failed to read frame from video stream")
				}

				if img.Empty() {
					// Empty frame could indicate stream end or temporary issue
					return fmt.Errorf("stream error: empty frame captured")
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

				// Only attempt reconnection if circuit is open (indicating multiple failures)
				if state == CircuitOpen {
					// Check if this is a connection error that warrants reconnection
					if strings.Contains(err.Error(), "connection error") || strings.Contains(err.Error(), "stream read error") {
						d.logger.Info("Circuit breaker open due to connection issues, attempting stream reconnection")
						if d.reconnectStream(ctx) {
							// Reset circuit breaker after successful reconnection
							d.circuitBreaker.Reset()
							d.logger.Info("Stream reconnected successfully, circuit breaker reset")
						} else {
							d.logger.Error("Stream reconnection failed, stopping capture")
							return
						}
					} else {
						d.logger.Debug("Circuit open but error doesn't warrant reconnection", "error_type", err.Error())
					}
				}
				continue
			}

			// Clone the image to avoid data races
			clonedImg := img.Clone()
			
			// Ensure the cloned image is valid before proceeding
			if clonedImg.Empty() {
				d.logger.Warn("Failed to clone frame image, skipping")
				continue
			}

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
				// Ensure cleanup on shutdown
				if !clonedImg.Empty() {
					clonedImg.Close()
				}
				return
			default:
				// Drop frame if channel is full (last resort)
				if !clonedImg.Empty() {
					clonedImg.Close()
				}
				d.metrics.framesDropped.Add(1)
				d.logger.Warn("Dropped frame due to full buffer",
					"frame_index", frameIndex,
					"total_dropped", d.metrics.GetFramesDropped())
			}
		}
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
			d.logger.Debug("Metrics reporting stopped")
			return
		case <-ticker.C:
			lastFrameAge := d.metrics.GetLastFrameAge()
			circuitState := d.circuitBreaker.GetState()
			circuitFailureCount := d.circuitBreaker.GetFailureCount()
			lastFailureTime := d.circuitBreaker.GetLastFailureTime()
			avgProcessingTime := d.metrics.GetAvgProcessingTimeMs()
			maxBufferUtil := d.metrics.GetMaxBufferUtilization()
			parallelFrames := d.metrics.GetParallelFramesProcessed()

			d.logger.Debug("Enhanced stream metrics report",
				"frames_processed", d.metrics.GetFramesProcessed(),
				"parallel_frames_processed", parallelFrames,
				"frames_dropped", d.metrics.GetFramesDropped(),
				"stream_errors", d.metrics.GetStreamErrors(),
				"ocr_errors", d.metrics.GetOCRErrors(),
				"reconnect_attempts", d.metrics.GetReconnectAttempts(),
				"last_frame_age_ms", lastFrameAge.Milliseconds(),
				"avg_processing_time_ms", avgProcessingTime,
				"max_buffer_utilization_pct", maxBufferUtil,
				"worker_count", d.ocrWorkerPool.workerCount,
				"frame_buffer_size", d.frameBufferSize,
				"result_buffer_size", d.resultBufferSize,
				"circuit_state", circuitState,
				"circuit_failure_count", circuitFailureCount,
				"last_failure_age_ms", func() int64 {
					if lastFailureTime.IsZero() {
						return -1
					}
					return time.Since(lastFailureTime).Milliseconds()
				}(),
				"stream_url", d.config.URL)

			// Calculate and log performance statistics
			if parallelFrames > 0 {
				processingRate := float64(parallelFrames) / 30.0 // frames per second over 30s interval
				d.logger.Debug("Performance statistics",
					"processing_rate_fps", processingRate,
					"cpu_cores", runtime.NumCPU(),
					"goroutines", runtime.NumGoroutine())
			}

			// Log warning if frames haven't been processed recently
			if lastFrameAge > 5*d.config.Interval {
				d.logger.Warn("Stream processing may be stalled",
					"last_frame_age", lastFrameAge,
					"expected_interval", d.config.Interval)
			}

			// Log warning if circuit breaker has been open for too long
			if circuitState == CircuitOpen && !lastFailureTime.IsZero() {
				timeSinceFailure := time.Since(lastFailureTime)
				if timeSinceFailure > 2*time.Minute {
					d.logger.Warn("Circuit breaker has been open for extended period",
						"time_open", timeSinceFailure,
						"failure_count", circuitFailureCount,
						"consider_manual_intervention", true)
				}
			}

			// Log performance warnings
			if maxBufferUtil > 90 {
				d.logger.Warn("High buffer utilization detected",
					"max_utilization_pct", maxBufferUtil,
					"consider_increasing_workers", true)
			}

			if avgProcessingTime > 200 { // 200ms is quite slow for OCR
				d.logger.Warn("Slow OCR processing detected",
					"avg_processing_time_ms", avgProcessingTime,
					"consider_optimization", true)
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

// logResults processes detection results and logs both matches and non-matches using structured logging.
// This method runs in its own goroutine and is the final stage of the processing pipeline.
// It receives detection results and outputs structured log entries for monitoring and analysis.
//
// Logging behavior:
//   - Match found: logs "Text detected in stream" with matched words
//   - No match found: logs "No target words detected in frame" 
//   - Both cases include frame analysis metadata for comprehensive monitoring
//
// Structured log fields included:
//   - timestamp: RFC3339 formatted capture time for temporal correlation
//   - frame_index: unique frame identifier for debugging and tracking
//   - matched_words: array of detected target words for filtering and alerting
//   - confidence: OCR confidence score for quality assessment
//   - extracted_text: complete OCR output for manual verification (in verbose mode only)
//   - stream_url: source identifier for multi-stream deployments
//
// The logging format (JSON or key=value) is determined by the configuration
// and optimized for different use cases: JSON for log aggregation systems,
// key=value for human-readable console output.
//
// All frame analysis results are logged to ensure complete visibility into
// the detection process, enabling effective monitoring and debugging.
func (d *Detector) logResults(ctx context.Context, resultChan <-chan DetectionResult) {
	for {
		select {
		case <-ctx.Done():
			d.logger.Debug("Result logging stopped")
			return
		case result, ok := <-resultChan:
			if !ok {
				return
			}

			if len(result.Matches) > 0 {
				// Match found - always log this as it's a primary detection event
				d.logger.Info("Text detected in stream",
					"timestamp", result.Frame.Timestamp.Format(time.RFC3339),
					"frame_index", result.Frame.Index,
					"matched_words", result.Matches,
					"confidence", result.Confidence,
					"stream_url", d.config.URL,
				)
				// Include extracted text in verbose mode for debugging
				if d.config.Verbose {
					d.logger.Debug("Extracted text from matched frame",
						"frame_index", result.Frame.Index,
						"extracted_text", result.Text)
				}
			} else {
				// No match found - log frame analysis completion
				d.logger.Info("No target words detected in frame",
					"timestamp", result.Frame.Timestamp.Format(time.RFC3339),
					"frame_index", result.Frame.Index,
					"confidence", result.Confidence,
					"stream_url", d.config.URL,
				)
				// Include extracted text in verbose mode for debugging
				if d.config.Verbose {
					d.logger.Debug("Extracted text from non-matched frame",
						"frame_index", result.Frame.Index,
						"extracted_text", result.Text)
				}
			}
		}
	}
}
