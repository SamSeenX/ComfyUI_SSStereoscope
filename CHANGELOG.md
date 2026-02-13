# Changelog

## [2.1.1] - 2026-02-13

### Added

- **New Node: SBS V2.1 (External Depth)**
  - Allows using **Custom/External Depth Maps** with the modern V2 rendering engine.
  - Uses **HighSodium Optimization** (Right-to-Left vectorization) to fix "reducing/eating" artifacts seen in the legacy node.
  - Significantly faster than the Legacy node (runs on CPU, compatible with M1/M2/M3 Macs).
  - Increased `depth_scale` limit to **200.0**.

### Fixed

- Fixed artifacts where foreground objects would appear thinner or have "holes" when using the legacy Left-to-Right algorithm.
