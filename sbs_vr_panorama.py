import torch
import numpy as np
import cv2
from PIL import Image
import math
from comfy.utils import ProgressBar
import gc
from typing import Tuple
import os
import sys

# Add the current directory to the path for depth estimator import
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

class SBS_VR_Panorama_by_SamSeen:
    """
    Create high-quality VR-compatible stereoscopic 360° panoramas from equirectangular images.
    Simplified version focused on quality over complexity.
    """
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.depth_model = None
        
    def _load_depth_model(self):
        """Load depth estimation model"""
        if self.depth_model is None:
            try:
                from depth_estimator import DepthEstimator
                self.depth_model = DepthEstimator()
                self.depth_model.load_model()
                print("Loaded depth model for VR panorama processing")
                return True
            except Exception as e:
                print(f"Warning: Could not load depth model: {e}")
                return False
        return True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "panorama_image": ("IMAGE", {"tooltip": "Input 360° panoramic image in equirectangular format (2:1 aspect ratio)"}),
                "depth_scale": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 20.0, "step": 0.5, "tooltip": "Controls stereo separation intensity. Start at 10 for noticeable depth, reduce if uncomfortable in VR."}),
                "ipd_mm": ("FLOAT", {"default": 65.0, "min": 50.0, "max": 80.0, "step": 1.0, "tooltip": "Interpupillary distance in millimeters. Average adult: 62-68mm."}),
                "format": (["over_under", "side_by_side"], {"default": "over_under", "tooltip": "VR output format. Over-under recommended for Quest/Meta headsets."}),
                "invert_depth": ("BOOLEAN", {"default": True, "tooltip": "Reverse depth perception. Usually needed since depth models often produce inverted depth."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("vr_panorama", "depth_panorama")
    FUNCTION = "create_vr_panorama"
    CATEGORY = "👀 SamSeen"
    DESCRIPTION = "Create high-quality VR-compatible stereoscopic 360° panoramas with optimized depth processing"

    def _generate_panorama_depth(self, panorama_tensor: torch.Tensor) -> np.ndarray:
        """Generate high-quality depth map optimized for panoramic content"""
        
        if not self._load_depth_model():
            h, w = panorama_tensor.shape[1:3]
            print("Using gradient fallback depth map")
            return self._create_gradient_depth(h, w)
        
        try:
            # Convert to numpy
            panorama_np = panorama_tensor.cpu().numpy() * 255.0
            panorama_np = panorama_np.astype(np.uint8)
            
            original_h, original_w = panorama_np.shape[:2]
            print(f"Processing panorama: {original_w}x{original_h}")
            
            # For large images, process in tiles for memory efficiency
            if original_w > 4096 or original_h > 2048:
                depth_map = self._process_panorama_tiles(panorama_np)
            else:
                # Process entire image
                depth_map = self.depth_model.predict_depth(panorama_np)
            
            # Apply panorama-specific depth enhancements
            depth_map = self._enhance_panorama_depth(depth_map, original_w, original_h)
            
            return depth_map
            
        except Exception as e:
            print(f"Error generating depth map: {e}")
            return self._create_gradient_depth(panorama_tensor.shape[1], panorama_tensor.shape[2])

    def _process_panorama_tiles(self, panorama_np: np.ndarray, tile_size: int = 1024, overlap: int = 128) -> np.ndarray:
        """Process large panoramas in overlapping tiles"""
        height, width = panorama_np.shape[:2]
        depth_map = np.zeros((height, width), dtype=np.float32)
        weight_map = np.zeros((height, width), dtype=np.float32)
        
        # Calculate number of tiles
        num_tiles_x = math.ceil(width / (tile_size - overlap))
        num_tiles_y = math.ceil(height / (tile_size - overlap))
        
        print(f"Processing {num_tiles_x}x{num_tiles_y} tiles")
        
        for ty in range(num_tiles_y):
            for tx in range(num_tiles_x):
                # Calculate tile boundaries
                start_x = tx * (tile_size - overlap)
                start_y = ty * (tile_size - overlap)
                end_x = min(start_x + tile_size, width)
                end_y = min(start_y + tile_size, height)
                
                # Extract tile
                tile = panorama_np[start_y:end_y, start_x:end_x]
                
                # Process tile
                try:
                    tile_depth = self.depth_model.predict_depth(tile)
                except Exception as e:
                    print(f"Error processing tile ({tx}, {ty}): {e}")
                    tile_depth = np.ones((end_y-start_y, end_x-start_x), dtype=np.float32) * 0.5
                
                # Create weight map for blending (cosine weighting)
                tile_h, tile_w = tile_depth.shape
                weights = np.ones((tile_h, tile_w), dtype=np.float32)
                
                # Apply edge feathering for seamless blending
                if overlap > 0:
                    feather = overlap // 4
                    for i in range(tile_h):
                        for j in range(tile_w):
                            # Distance to edges
                            dist_left = j
                            dist_right = tile_w - j - 1
                            dist_top = i
                            dist_bottom = tile_h - i - 1
                            
                            # Apply feathering
                            if start_x > 0 and dist_left < feather:
                                weights[i, j] *= dist_left / feather
                            if end_x < width and dist_right < feather:
                                weights[i, j] *= dist_right / feather
                            if start_y > 0 and dist_top < feather:
                                weights[i, j] *= dist_top / feather
                            if end_y < height and dist_bottom < feather:
                                weights[i, j] *= dist_bottom / feather
                
                # Accumulate depth and weights
                depth_map[start_y:end_y, start_x:end_x] += tile_depth * weights
                weight_map[start_y:end_y, start_x:end_x] += weights
                
                # Memory cleanup
                del tile_depth, weights
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Normalize by weights
        valid_mask = weight_map > 0.01
        depth_map[valid_mask] /= weight_map[valid_mask]
        
        return depth_map

    def _enhance_panorama_depth(self, depth_map: np.ndarray, width: int, height: int) -> np.ndarray:
        """Apply panorama-specific depth enhancements optimized for realistic VR depth"""
        
        # Normalize depth to [0,1]
        depth_min, depth_max = np.min(depth_map), np.max(depth_map)
        if depth_max - depth_min > 1e-6:
            depth_map = (depth_map - depth_min) / (depth_max - depth_min)
        
        # Lighter filtering to preserve resolution and detail
        depth_8bit = (depth_map * 255).astype(np.uint8)
        # Single bilateral filter pass - preserve more detail
        filtered = cv2.bilateralFilter(depth_8bit, 5, 50, 50)
        depth_map = filtered.astype(np.float32) / 255.0
        
        # Much lighter gaussian blur to preserve sharpness
        blur_size = max(3, min(5, width // 1000))  # Smaller blur relative to size
        if blur_size % 2 == 0:
            blur_size += 1
        depth_map = cv2.GaussianBlur(depth_map, (blur_size, blur_size), 0)
        
        # Improve depth realism with better curve
        # Instead of power 0.7, use a more realistic depth curve
        # This creates more gradual transitions and less "pop-up" effect
        depth_map = np.sqrt(depth_map)  # Square root gives more natural depth falloff
        
        # Add subtle depth gradients to reduce flat "cardboard" effect
        # Create small local variations that make surfaces feel more volumetric
        kernel = np.array([[-0.1, -0.1, -0.1],
                          [-0.1,  1.8, -0.1], 
                          [-0.1, -0.1, -0.1]])
        enhanced = cv2.filter2D(depth_map, -1, kernel)
        depth_map = np.clip(0.8 * depth_map + 0.2 * enhanced, 0.0, 1.0)
        
        # Light pole correction only
        for y in range(height):
            lat_norm = abs(y / height - 0.5) * 2
            if lat_norm > 0.8:  # Only very close to poles
                correction = 1.0 - (lat_norm - 0.8) * 0.2  # Much lighter correction
                depth_map[y, :] *= correction
        
        # Ensure good contrast but don't over-enhance
        depth_range = np.max(depth_map) - np.min(depth_map)
        if depth_range < 0.4:  # Only enhance if really flat
            depth_center = np.mean(depth_map)
            depth_map = np.clip((depth_map - depth_center) * 1.2 + depth_center, 0.0, 1.0)
        
        return depth_map

    def _create_gradient_depth(self, height: int, width: int) -> np.ndarray:
        """Create a reasonable fallback depth map"""
        depth = np.zeros((height, width), dtype=np.float32)
        
        for y in range(height):
            for x in range(width):
                # Convert to spherical coordinates
                lat = (y / height - 0.5) * math.pi
                lon = (x / width - 0.5) * 2 * math.pi
                
                # Create depth based on spherical coordinates with some variation
                depth_val = 0.5 + 0.2 * math.cos(lat * 2) + 0.15 * math.sin(lon * 3)
                depth[y, x] = np.clip(depth_val, 0.0, 1.0)
        
        return depth

    def _create_stereo_displacement(self, depth_map: np.ndarray, ipd_mm: float, depth_scale: float) -> np.ndarray:
        """Calculate stereo displacement for panoramic content"""
        h, w = depth_map.shape
        
        # More conservative and predictable displacement calculation
        base_displacement = depth_scale * 0.5
        
        displacement = depth_map * base_displacement
        
        # Apply latitude correction
        lat_factors = np.cos((np.arange(h) / h - 0.5) * math.pi)
        lat_factors = 0.7 + 0.3 * lat_factors  # Range from 0.7 to 1.0
        displacement = displacement * lat_factors[:, np.newaxis]
        
        print(f"Displacement range: {np.min(displacement):.2f} to {np.max(displacement):.2f} pixels")
        
        return displacement

    def _apply_stereo_shift(self, image: np.ndarray, displacement: np.ndarray, eye: str = 'left') -> np.ndarray:
        """Apply stereo shift with high-quality interpolation to preserve resolution"""
        h, w = image.shape[:2]
        
        # Correct stereo direction for VR
        shift_factor = -1.0 if eye == 'left' else 1.0
        
        # Create coordinate grids
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        
        # Apply shift with sub-pixel precision
        shifted_x = (x_coords + displacement * shift_factor) % w
        
        # Use cubic interpolation for maximum quality
        map_x = shifted_x.astype(np.float32)
        map_y = y_coords.astype(np.float32)
        
        shifted_image = cv2.remap(image, map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)
        
        return shifted_image

    def create_vr_panorama(self, panorama_image, depth_scale, ipd_mm, format, invert_depth):
        """Create VR-compatible stereoscopic panorama with simplified, high-quality processing"""
        
        # Handle batch dimension
        if len(panorama_image.shape) == 4:
            panorama_tensor = panorama_image[0]
        else:
            panorama_tensor = panorama_image
        
        # Validate aspect ratio
        h, w = panorama_tensor.shape[:2]
        aspect_ratio = w / h
        if not (1.8 <= aspect_ratio <= 2.2):
            print(f"Warning: Aspect ratio {aspect_ratio:.2f} is not typical for equirectangular (should be ~2:1)")
        
        print(f"Processing {w}x{h} panorama for VR")
        
        try:
            # Generate high-quality depth map
            depth_map = self._generate_panorama_depth(panorama_tensor)
            
            # Apply depth inversion if requested
            if invert_depth:
                depth_map = 1.0 - depth_map
            
            # Convert panorama to numpy for processing
            panorama_np = panorama_tensor.cpu().numpy() * 255.0
            panorama_np = panorama_np.astype(np.uint8)
            
            # Create stereo displacement map
            displacement = self._create_stereo_displacement(depth_map, ipd_mm, depth_scale)
            
            # Generate left and right eye views
            print("Generating stereo views...")
            left_eye = self._apply_stereo_shift(panorama_np, displacement, 'left')
            right_eye = self._apply_stereo_shift(panorama_np, displacement, 'right')
            
            # Combine into selected stereo format
            if format == "side_by_side":
                stereo_panorama = np.concatenate([left_eye, right_eye], axis=1)
            else:  # over_under
                stereo_panorama = np.concatenate([left_eye, right_eye], axis=0)
            
            # Convert back to tensors
            stereo_tensor = torch.tensor(stereo_panorama.astype(np.float32) / 255.0).unsqueeze(0)
            
            # Create depth visualization (3-channel for ComfyUI compatibility)
            depth_vis = np.stack([depth_map, depth_map, depth_map], axis=-1)
            depth_tensor = torch.tensor(depth_vis).unsqueeze(0)
            
            # Memory cleanup
            del panorama_np, left_eye, right_eye, displacement, depth_map
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print(f"VR panorama created successfully: {stereo_tensor.shape}")
            return (stereo_tensor, depth_tensor)
            
        except Exception as e:
            print(f"Error creating VR panorama: {e}")
            import traceback
            traceback.print_exc()
            
            # Return fallback result
            if format == "side_by_side":
                fallback_shape = (1, h, w * 2, 3)
            else:
                fallback_shape = (1, h * 2, w, 3)
            
            fallback_stereo = torch.zeros(fallback_shape)
            fallback_depth = torch.zeros((1, h, w, 3))
            
            return (fallback_stereo, fallback_depth)
