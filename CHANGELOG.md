# Changelog

# Changelog

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
