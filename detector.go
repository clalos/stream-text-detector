package main

import (
	"context"
	"fmt"
	"image"
	"log/slog"
	"strings"
	"sync"
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

// Detector manages video stream capture and OCR processing using a concurrent pipeline.
// It coordinates three goroutines: frame capture, OCR processing, and result logging.
// All operations are thread-safe and respond to context cancellation for graceful shutdown.
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

	// mu protects frameIndex during concurrent updates from the capture goroutine.
	mu sync.RWMutex
}

// NewDetector creates a new Detector instance with the given configuration.
// It initializes the video capture connection and OCR client with proper validation.
//
// The function performs the following initialization steps:
//   1. Opens video capture connection to the specified stream URL
//   2. Verifies the connection is active and responsive
//   3. Creates and configures Tesseract OCR client with specified language
//   4. Sets OCR page segmentation mode to PSM_AUTO for optimal text detection
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

	return &Detector{
		config:    config,
		logger:    logger,
		capture:   capture,
		ocrClient: ocrClient,
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
//
// The method blocks until the context is cancelled or an unrecoverable error occurs.
// All goroutines respond to context cancellation for coordinated shutdown.
//
// Channel buffer sizes (10 each) provide flow control to handle processing
// speed variations between stages while preventing excessive memory usage.
//
// Error handling: Individual frame processing errors are logged but don't
// terminate the entire pipeline, ensuring robust operation with intermittent issues.
func (d *Detector) Run(ctx context.Context) error {
	frameChan := make(chan Frame, 10)
	resultChan := make(chan DetectionResult, 10)

	var wg sync.WaitGroup

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

// captureFrames captures frames from the video stream at the configured interval.
// This method runs in its own goroutine and is the first stage of the processing pipeline.
//
// Behavior:
//   - Uses a ticker to sample frames at the configured interval
//   - Clones each frame to prevent data races with OpenCV Mat objects
//   - Assigns monotonically increasing frame indices for tracking
//   - Drops frames if the processing pipeline is backlogged (non-blocking send)
//   - Responds immediately to context cancellation for shutdown
//
// Frame memory management: Each captured frame is cloned to ensure thread safety.
// The clone is closed by the OCR processing stage after use.
//
// Error handling: Individual frame read failures are logged but don't terminate
// the capture loop, allowing recovery from temporary stream issues.
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
			if !d.capture.Read(&img) {
				d.logger.Error("Failed to read frame from video stream")
				continue
			}

			if img.Empty() {
				d.logger.Warn("Empty frame captured")
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

			select {
			case frameChan <- frame:
			case <-ctx.Done():
				clonedImg.Close()
				return
			default:
				// Drop frame if channel is full
				clonedImg.Close()
				d.logger.Warn("Dropped frame due to full buffer", "frame_index", frameIndex)
			}
		}
	}
}

// processFrames processes captured frames with OCR and word matching.
// This method runs in its own goroutine and is the second stage of the processing pipeline.
// It receives frames from the capture stage and outputs detection results.
//
// Processing workflow for each frame:
//   1. Apply image preprocessing (grayscale, resize, threshold)
//   2. Perform Tesseract OCR text extraction
//   3. Calculate average confidence score from detected words
//   4. If confidence meets threshold, check for target word matches
//   5. Send results to logging stage (only frames with matches)
//
// Memory management: Properly closes the frame image after processing to prevent leaks.
// Results are only forwarded if target words are detected to reduce logging volume.
//
// Error handling: OCR failures are logged with frame context but don't stop processing,
// ensuring the pipeline continues operating despite individual frame issues.
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

			result := d.processFrame(frame)
			frame.Image.Close() // Clean up the frame image

			if len(result.Matches) > 0 {
				select {
				case resultChan <- result:
				case <-ctx.Done():
					return
				default:
					d.logger.Warn("Dropped detection result due to full buffer", "frame_index", result.Frame.Index)
				}
			}
		}
	}
}

// processFrame performs OCR on a single frame and checks for word matches.
// This is the core processing logic that transforms video frames into text detection results.
//
// Processing steps:
//   1. Apply image preprocessing (grayscale conversion, resize, adaptive threshold)
//   2. Encode preprocessed image as PNG for Tesseract input
//   3. Extract text using configured OCR language settings
//   4. Retrieve word-level bounding boxes with confidence scores
//   5. Calculate average confidence across all detected words
//   6. If confidence ≥ threshold, perform case-insensitive target word matching
//
// Confidence calculation: Uses the average confidence of all detected words
// with confidence > 0. This provides a more stable metric than individual
// word confidence scores which can vary significantly.
//
// Returns a DetectionResult with the original frame metadata, extracted text,
// calculated confidence, and any matched target words. Empty matches indicate
// either no target words found or confidence below threshold.
func (d *Detector) processFrame(frame Frame) DetectionResult {
	// Preprocess the frame for better OCR accuracy
	processed := d.preprocessFrame(frame.Image)
	defer processed.Close()

	// Convert to bytes for OCR
	imgBytes, err := gocv.IMEncode(".png", processed)
	if err != nil {
		d.logger.Error("Failed to encode image", "error", err, "frame_index", frame.Index)
		return DetectionResult{Frame: frame}
	}

	// Perform OCR
	if err := d.ocrClient.SetImageFromBytes(imgBytes.GetBytes()); err != nil {
		d.logger.Error("Failed to set OCR image", "error", err, "frame_index", frame.Index)
		return DetectionResult{Frame: frame}
	}

	text, err := d.ocrClient.Text()
	if err != nil {
		d.logger.Error("Failed to extract text", "error", err, "frame_index", frame.Index)
		return DetectionResult{Frame: frame}
	}

	// Get bounding boxes to calculate average confidence
	boxes, err := d.ocrClient.GetBoundingBoxes(gosseract.RIL_WORD)
	if err != nil {
		d.logger.Error("Failed to get bounding boxes", "error", err, "frame_index", frame.Index)
		return DetectionResult{Frame: frame}
	}

	// Calculate average confidence from all detected words
	var totalConfidence float64
	var wordCount int
	for _, box := range boxes {
		if box.Confidence > 0 {
			totalConfidence += box.Confidence
			wordCount++
		}
	}

	var avgConfidence float64
	if wordCount > 0 {
		avgConfidence = totalConfidence / float64(wordCount)
	}

	// Check for word matches if confidence meets threshold
	var matches []string
	if avgConfidence >= d.config.Confidence*100 {
		matches = d.findMatches(text)
	}

	return DetectionResult{
		Frame:      frame,
		Text:       strings.TrimSpace(text),
		Confidence: avgConfidence / 100.0,
		Matches:    matches,
	}
}

// preprocessFrame applies preprocessing steps to improve OCR accuracy.
// This method optimizes the input image for Tesseract text recognition using
// standard computer vision techniques proven effective for OCR applications.
//
// Processing pipeline:
//   1. Convert to grayscale - reduces noise and focuses on text structure
//   2. Resize to 150% - improves small text recognition as per requirements
//   3. Apply adaptive threshold - enhances text contrast against background
//
// The adaptive threshold uses mean-based thresholding with an 11x11 kernel
// and constant offset of 2, which works well for various lighting conditions
// and text styles commonly found in video streams.
//
// Returns a new Mat containing the processed image. The caller must close
// the returned Mat to prevent memory leaks.
func (d *Detector) preprocessFrame(src gocv.Mat) gocv.Mat {
	// Convert to grayscale
	gray := gocv.NewMat()
	gocv.CvtColor(src, &gray, gocv.ColorBGRToGray)

	// Resize to improve OCR accuracy (up to 150% as mentioned in requirements)
	resized := gocv.NewMat()
	newSize := image.Point{
		X: int(float64(gray.Cols()) * 1.5),
		Y: int(float64(gray.Rows()) * 1.5),
	}
	gocv.Resize(gray, &resized, newSize, 0, 0, gocv.InterpolationLinear)
	gray.Close()

	// Apply adaptive threshold to improve text contrast
	thresholded := gocv.NewMat()
	gocv.AdaptiveThreshold(resized, &thresholded, 255, gocv.AdaptiveThresholdMean, gocv.ThresholdBinary, 11, 2)
	resized.Close()

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
