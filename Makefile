# Makefile for Stream Text Detector

BINARY_NAME=std
BINARY_UNIX=$(BINARY_NAME)_unix
BINARY_WINDOWS=$(BINARY_NAME).exe
BINARY_DARWIN=$(BINARY_NAME)_darwin

# Go parameters
GOCMD=go
GOBUILD=$(GOCMD) build
GOCLEAN=$(GOCMD) clean
GOTEST=$(GOCMD) test
GOGET=$(GOCMD) get
GOMOD=$(GOCMD) mod

# Build flags
LDFLAGS=-ldflags="-s -w"

.PHONY: all build clean test deps help

all: test build

build: ## Build the binary
	$(GOBUILD) $(LDFLAGS) -o $(BINARY_NAME) -v ./...

build-linux: ## Build for Linux
	CGO_ENABLED=1 GOOS=linux GOARCH=amd64 $(GOBUILD) $(LDFLAGS) -o $(BINARY_UNIX) -v ./...

build-windows: ## Build for Windows
	CGO_ENABLED=1 GOOS=windows GOARCH=amd64 $(GOBUILD) $(LDFLAGS) -o $(BINARY_WINDOWS) -v ./...

build-darwin: ## Build for macOS
	CGO_ENABLED=1 GOOS=darwin GOARCH=amd64 $(GOBUILD) $(LDFLAGS) -o $(BINARY_DARWIN) -v ./...

build-all: build-linux build-windows build-darwin ## Build for all platforms

test: ## Run tests (requires OpenCV and Tesseract)
	$(GOTEST) -v ./...

test-coverage: ## Run tests with coverage
	$(GOTEST) -cover -v ./...

clean: ## Clean build artifacts
	$(GOCLEAN)
	rm -f $(BINARY_NAME)
	rm -f $(BINARY_UNIX)
	rm -f $(BINARY_WINDOWS)
	rm -f $(BINARY_DARWIN)

deps: ## Download dependencies
	$(GOMOD) download
	$(GOMOD) verify

deps-update: ## Update dependencies
	$(GOMOD) tidy
	$(GOGET) -u ./...

run: ## Run the application (requires -url and -word flags)
	$(GOBUILD) -o $(BINARY_NAME) -v ./... && ./$(BINARY_NAME) $(ARGS)

install: ## Install the binary to GOPATH/bin
	$(GOCMD) install $(LDFLAGS) ./...

help: ## Display this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'