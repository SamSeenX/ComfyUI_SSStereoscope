# Changelog

## [2.1.8] - 2026-03-02

### Added

- **Audio Output for SBS Video Uploader** (Issue #20):
  - New `AUDIO` output that extracts the audio track from the uploaded video.
  - Audio is automatically trimmed to stay in sync with frame selection (`skip_first_frames`, `select_every_nth`, `frame_load_cap`).
  - Can be wired directly to the `SBS Video Combiner`'s audio input for seamless audio pass-through.
- **FPS Outputs for SBS Video Uploader**:
  - `source_fps` (`FLOAT`): The original video frame rate.
  - `target_fps` (`FLOAT`): Effective fps adjusted for `select_every_nth` (calculated as `source_fps / select_every_nth`). Wire this directly to the combiner's `frame_rate` for perfect sync.

### Changed

- **SBS Video Combiner `frame_rate` now accepts FLOAT**:
  - Changed from `INT` to `FLOAT` for precise frame rate control (e.g., 23.976, 29.97).
  - Can now be directly wired from the Video Uploader's `fps` output.

### Fixed

- **Concurrent workflow safety**: Temp files now use unique names (UUID-based) to prevent collisions when multiple workflows run simultaneously.

## [2.1.7] - 2026-02-14

### Fixed

- **Progress Bar Throttling**:
  - Reduced the number of progress update messages sent to the browser.
  - Fixes the issue where the progress bar would "lag" for minutes after processing high-speed images.
  - Improves UI responsiveness and prevents browser freezes during batch processing.

## [2.1.6] - 2026-02-14

### Added

- **Gap Fill Modes & Inpainting (SBS V2.1)**:
- Added `gap_fill_mode` dropdown to handle occlusion gaps ("holes") caused by high depth.
- **Inpaint (Telea)**: (Default) Uses internal OpenCV fast inpainting to fill gaps with smooth textures. Removes streaks without leaving holes.
- **Stretch**: Legacy behavior. Stretches the last pixel to cover the gap. Fast but creates "streaking" artifacts.
- **None**: Leaves gaps as **Black Holes**. Useful if you want to use the Mask output for external inpainting.
- **Output Layout Options**:
  - Added `stereo_layout` dropdown.
  - **Side by Side**: (Default) Standard horizontal layout.
  - **Top Bottom**: Vertical layout (Left/Top, Right/Bottom or inverted for cross-eyed).
- **Gap Mask Output**:
  - New `MASK` return output (`gap_mask`).
  - Provides a white mask (1.0) where gaps/holes are.
  - Useful for advanced workflows using generative inpainting (KSampler) to fix disocclusions.

## [2.1.5] - 2026-02-13

### Changed

- **Refined Depth Scale (SBS V2.1)**:
  - The `depth_scale` slider (0-100) now maps to **0% - 20%** of the image width.
  - This provides finer control for realistic 3D effects, as values >20% usually break the stereoscopic effect.
  - **Formula**: `Max Shift = Width * (SliderValue / 500)`

## [2.1.4] - 2026-02-13

### Changed

- **Resolution-Relative Depth Scaling (SBS V2.1)**:
  - `depth_scale` is now a **percentage of the image width** (0-100).
  - Example: A scale of `10` is always "10% width separation", whether the image is 512px or 4000px.
  - **Breaking Change**: Default value changed from `30` to `5` to match this new sensitivity.
  - Solves the issue where high-res images needed massive scale values (e.g., 2600).

## [2.1.3] - 2026-02-13

### Fixed

- Fixed repository URL configuration to point to correct `SamSeenX` repository.

## [2.1.1] - 2026-02-13

### Added

- **New Node: SBS V2.1 (External Depth)**
  - Allows using **Custom/External Depth Maps** with the modern V2 rendering engine.
  - Uses **HighSodium Optimization** (Right-to-Left vectorization) to fix "reducing/eating" artifacts seen in the legacy node.
  - Significantly faster than the Legacy node (runs on CPU, compatible with M1/M2/M3 Macs).
  - Increased `depth_scale` limit to **200.0**.

### Fixed

- Fixed artifacts where foreground objects would appear thinner or have "holes" when using the legacy Left-to-Right algorithm.
