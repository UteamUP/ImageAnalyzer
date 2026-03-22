# Changelog

All notable changes to the UteamUP Image Analyzer will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] — 2026-03-22

### Added
- Multi-entity detection per image (assets, parts, tools, chemicals in one photo)
- Asset hierarchy support with `related_to` field linking child entities to parents
- `related_to` column in all CSV exports
- Cross-image deduplication with exact name match grouping
- Description fuzzy similarity signal in grouper (weight 0.10)
- Field merging: representative's null fields filled from higher-confidence members
- Separated output folders: CSVs/reports → `./Output`, renamed images → `./Images/Updated`

### Changed
- Prompt restructured to return `entities` array instead of single classification
- Analyzer returns list of results per image
- Grouper similarity weights rebalanced (serial 0.40, model 0.20, name 0.20, desc 0.10, phash 0.05, brand 0.05)
- Default model updated to `gemini-3.1-flash-lite-preview`
- Pipeline handles multi-entity results and list-format checkpoints

## [0.1.0] — 2026-03-22

### Added
- Initial image analyzer with Gemini Vision AI integration
- 4-phase pipeline: scan → analyze → group → export
- Entity classification: asset, tool, part, chemical
- HEIC/HEIF conversion and image resizing
- iPhone edit pair detection (IMG_E* variants)
- Duplicate detection via SHA-256 and perceptual hashing
- 5-stage image grouping with agglomerative clustering
- CSV export per entity type with CMMS-aligned columns
- Image renaming with descriptive filenames
- Checkpoint/resume support for long-running batches
- Dry-run mode with cost estimation
- Rate limiting with token bucket and exponential backoff retry
- 41 unit tests (scanner, analyzer, grouper, exporter)
- Click CLI with `analyze` and `status` commands
- Configurable via YAML + .env + CLI flags
