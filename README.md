# DepthFX

Real-Time AI Depth-Aware Visual Effects

## Project Status

Early development — Phase 1: Project Setup

## Overview

DepthFX is a computer vision and computer graphics project that uses a pretrained monocular depth-estimation model to estimate relative depth from a normal RGB image or webcam stream.

The estimated depth information will later be used to create GPU-based visual effects such as:

- Depth-map visualization
- Depth-aware fog
- Depth-aware blur
- Depth-aware lighting
- Pseudo-3D / parallax effects

## Planned Pipeline

Webcam / Image
↓
RGB Frame
↓
Monocular Depth Estimation
↓
Depth Map
↓
GPU / OpenGL Processing
↓
Depth-Aware Effect
↓
Final Output

## Technology Stack

- Python
- PyTorch
- CUDA
- OpenCV
- OpenGL
- GLSL
- PyOpenGL
- Pyglet
- Depth Anything V2

## Hardware

Development hardware:

- NVIDIA GeForce RTX 4070 Laptop GPU
- 8 GB VRAM
- Windows 11

## Development Approach

The project is being developed incrementally.

Each phase is implemented, tested, and verified before moving to the next phase.

## Current Phase

Phase 1 — Project Setup

Completed:

- Python environment created
- Python 3.11.9 verified
- CUDA-enabled PyTorch verified
- RTX 4070 GPU computation verified
- OpenCV verified
- Webcam access verified
- OpenGL 4.6 context verified
- Initial project structure created

## Planned Phases

1. Environment Verification
2. Project Setup
3. Webcam Pipeline
4. Depth Estimation
5. Depth Normalization
6. Depth-Aware Fog
7. Depth-Aware Blur
8. Depth-Aware Lighting
9. Pseudo-3D / Parallax
10. Real-Time Pipeline
11. Performance Monitoring
12. Optimization
13. Final UI
14. Testing
15. Benchmarking
16. GitHub Documentation

## Limitations

The depth model will initially provide relative depth rather than guaranteed metric physical distance.

Advanced features such as zero-copy CUDA/OpenGL interoperability, full 3D reconstruction, NeRF, SLAM, and physically accurate rendering are outside the initial MVP scope.

## License

To be decided.