# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-05-15

### Added

- Backend stability enhancements (v1.1.0)
- Production logging system with file rotation
- Request timeout handling (5 minute limit per request)
- Concurrent request limiting (max 3 simultaneous requests)
- Request ID tracking for debugging
- Automatic GPU/CPU fallback with logging
- `/stats` endpoint for monitoring active requests

### Fixed

- Improved error handling across all endpoints
- Proper cleanup of temporary video files

## [1.0.0] - 2026-05-13

### Added

- Photo comparison - Upload two photos and calculate face similarity
- Face detection - Detect all faces in photos, identify age and gender
- Video comparison - Search for target faces in videos with real-time progress bar
- Export reports - Export HTML analysis reports or JSON raw data
- Offline operation - All computation done locally, data stays on machine
- Single file deployment - Packaged as a single exe for easy distribution
- GPU acceleration support via ctx_id configuration
- CPU fallback mode for low-end computers
