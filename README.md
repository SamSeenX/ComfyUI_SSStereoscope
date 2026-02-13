# ComfyUI SS Stereoscope

[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-ComfyUI-blue.svg)]()

> Transform any 2D image into immersive stereoscopic 3D content in ComfyUI.

The SideBySide Stereoscope Node is a cutting-edge 2D-to-3D conversion tool designed for ComfyUI. Powered by Depth-Anything (V2 & V3), it automatically creates high-quality depth maps and generates side-by-side (SBS) 3D images and videos perfect for VR headsets and 3D displays.

## ✨ Features

- 🧠 **AI-Powered Depth** - Uses Depth-Anything (V2 & V3) for automatic, high-quality depth estimation.
- 👓 **Stereoscopic Modes** - Supports Parallel and Cross-eyed viewing techniques.
- 🎥 **Video Support** - Full workflow for converting videos to 3D (upload, process, combine).
- 🎛️ **Fine Control** - Adjust depth scale, blur radius, and depth inversion.
- ⚡ **Integrated Workflow** - No external tools needed; handles everything within ComfyUI.
- 🚀 **Performance Mode** - Optional "HighSodium Optimization" for up to 4x faster processing.

## 📸 Demo

![Workflow Example](https://github.com/SamSeenX/ComfyUI_SSStereoscope/assets/8541558/6b972838-07ca-4e64-a3a4-32277ddcf4c7)

## 🚀 Quick Start

### Prerequisites

- ComfyUI installed
- Python >= 3.8
- **FFmpeg** (Recommended for video support)
  - The node attempts to use your system's FFmpeg first.
  - If missing, it will fallback to the `imageio-ffmpeg` binary (included in dependencies).
  - **Windows Users**: For best performance, install FFmpeg manually from [ffmpeg.org](https://ffmpeg.org/download.html) and ensure it is added to your System PATH.

### Installation

1.  Navigate to your custom nodes directory:
    ```bash
    cd ComfyUI/custom_nodes
    ```
2.  Clone the repository:
    ```bash
    git clone https://github.com/SamSeenX/ComfyUI_SSStereoscope.git
    ```
3.  Install dependencies:
    ```bash
    pip install -r ComfyUI_SSStereoscope/requirements.txt
    ```
4.  Restart ComfyUI.

_(Note: The AI model will download automatically on first use.)_

## 📖 Documentation

### Core Nodes

- **👀 SBS V2**: The main node. Input a 2D image, get a 3D SBS image + depth map.
  - `depth_scale`: Intensity of the 3D effect.
  - `mode`: Cross-eyed vs Parallel.
  - `highsodium_optimization`: Enable vectorized algorithm for faster processing (may produce slightly different results).
- **👀 SBS V2.1 (External Depth)**: _New in v2.1!_ Uses the fast V2 engine (HighSodium) but accepts your custom Depth Maps.
  - Fixes "reducing" artifacts found in the legacy node.
  - **Resolution-Relative Scaling**: `depth_scale` (0-100) now maps to **0-20% of image width**. This gives you fine-grained control while preventing output-breaking values.
    - Scale 10 = 2% width separation.
    - Scale 50 = 10% width separation.
    - Scale 100 = 20% width separation (Max).
- **👀 SBS Video Uploader**: Converts input video to image sequence.
- **👀 SBS Video Combiner**: Merges processed frames back into a video.

### Workflow Example

1.  **Video Input**: Use `SBS Video Uploader` to load a video.
2.  **Process**: Connect frames to `SBS V2` node.
3.  **Output**: Connect results to `SBS Video Combiner` to save the 3D video.

## 🏗️ Project Structure

```
ComfyUI_SSStereoscope/
├── sbs_v2.py            # Main logic
├── requirements.txt     # Dependencies
├── video_utils.py       # Video processing helpers
└── README.md
```

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

### Contributors

- [SamSeen](https://github.com/SamSeenX) - Original Author
- [HighSodium](https://github.com/HighSodium) - Performance optimization ([PR #16](https://github.com/SamSeenX/ComfyUI_SSStereoscope/pull/16))

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ☕ Support

If you find this project useful, please consider supporting me:

- ⭐ Starring this repository
- 🐛 Reporting issues
- ☕ [Buy me a coffee](https://buymeacoffee.com/samseen)

---

Created with ❤️ by [SamSeen](https://buymeacoffee.com/samseen)
