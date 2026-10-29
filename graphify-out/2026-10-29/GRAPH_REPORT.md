# Graph Report - NuovoProgetto  (2026-10-29)

## Corpus Check
- 6 files · ~4,464 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 66 nodes · 97 edges · 12 communities (2 shown, 10 thin omitted)
- Extraction: 92% EXTRACTED · 8% INFERRED · 0% AMBIGUOUS · INFERRED: 8 edges (avg confidence: 0.88)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- FaceTracker
- app.py
- ReelForge - YouTube to Vertical 9:16 Converter
- app.py
- on_start_processing
- Transcriber
- VideoProcessor
- process_job

## God Nodes (most connected - your core abstractions)
1. `VideoProcessor` - 15 edges
2. `FaceTracker` - 8 edges
3. `Transcriber` - 7 edges
4. `on_start_processing()` - 6 edges
5. `ReelForge - YouTube to Vertical 9:16 Converter` - 6 edges
6. `convert()` - 4 edges
7. `process_job()` - 3 edges
8. `upload_video()` - 3 edges
9. `index()` - 2 edges
10. `download()` - 2 edges

## Surprising Connections (you probably didn't know these)
- `VideoProcessor` --uses--> `FaceTracker`  [INFERRED]
  utils/video_processor.py → utils/face_tracker.py
- `VideoProcessor` --uses--> `Transcriber`  [INFERRED]
  utils/video_processor.py → utils/transcriber.py

## Import Cycles
- None detected.

## Communities (12 total, 10 thin omitted)

### Community 2 - "app.py"
Cohesion: 0.40
Nodes (3): convert(), download(), index()

### Community 6 - "ReelForge - YouTube to Vertical 9:16 Converter"
Cohesion: 0.29
Nodes (6): Features, Installation, Project Structure, ReelForge - YouTube to Vertical 9:16 Converter, Requirements, Usage

## Knowledge Gaps
- **10 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `VideoProcessor` connect `VideoProcessor` to `FaceTracker`, `video_processor.py`, `.split_by_duration`, `.cleanup`, `app.py`, `Transcriber`?**
  _High betweenness centrality (0.400) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `VideoProcessor` (e.g. with `FaceTracker` and `Transcriber`) actually correct?**
  _`VideoProcessor` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Why does `FaceTracker` connect `FaceTracker` to `video_processor.py`, `VideoProcessor`, `Transcriber`?**
  _High betweenness centrality (0.103) - this node is a cross-community bridge._
- **Why does `Transcriber` connect `Transcriber` to `video_processor.py`, `VideoProcessor`, `.split_by_duration`?**
  _High betweenness centrality (0.071) - this node is a cross-community bridge._