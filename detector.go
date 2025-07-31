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

// Frame represents a captured video frame with metadata.
type Frame struct {
	Image     gocv.Mat
	Index     int64
	Timestamp time.Time
}

// DetectionResult represents the result of OCR processing on a frame.
type DetectionResult struct {
	Frame      Frame
	Text       string
	Confidence float64
	Matches    []string
}

// Detector manages video stream capture and OCR processing.
type Detector struct {
	config     *Config
	logger     *slog.Logger
	capture    *gocv.VideoCapture
	ocrClient  *gosseract.Client
	frameIndex int64
	mu         sync.RWMutex
}

// NewDetector creates a new Detector instance with the given configuration.
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

// Run starts the video stream processing loop.
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

// findMatches searches for target words in the extracted text (case-insensitive).
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

// logResults processes detection results and logs matches.
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
