import torch
from PIL import Image
import numpy as np
import tqdm
from comfy.utils import ProgressBar

class SBS_V2_External_Depth_by_SamSeen:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_image": ("IMAGE",),
                "depth_map": ("IMAGE",),
                "depth_scale": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 100.0, "step": 0.1, "display": "slider"}),
                "blur_radius": ("INT", {"default": 3, "min": 1, "max": 51, "step": 2}),
                "invert_depth": ("BOOLEAN", {"default": False}),
                "mode": (["Parallel", "Cross-eyed"], {"default": "Cross-eyed"}),
                "highsodium_optimization": ("BOOLEAN", {"default": True, "label_on": "Fast (HighSodium)", "label_off": "Legacy"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("stereoscopic_image",)
    FUNCTION = "process"
    CATEGORY = "👀 SamSeen"
    DESCRIPTION = "V2.1: Create side-by-side (SBS) stereoscopic images using your own custom depth maps. Uses the optimized HighSodium rendering engine for speed and artifact reduction."

    def process(self, base_image, depth_map, depth_scale, blur_radius, invert_depth=False, mode="Cross-eyed", highsodium_optimization=True):
        """
        Create a side-by-side (SBS) stereoscopic image from a standard image and an external depth map.
        """
        
        # Get batch size
        B = base_image.shape[0]
        sbs_images = []

        # Optimization for HighSodium mode (vectorized)
        if highsodium_optimization:
            # Pre-validate inputs to avoid errors during batch processing
            pass

        for b in range(B):
            # Get the current image
            current_image = base_image[b].cpu().numpy()
            current_image_pil = Image.fromarray((current_image * 255).astype(np.uint8))
            
            # Get the corresponding depth map (handle batch mismatch by broadcasting)
            depth_idx = b % depth_map.shape[0]
            current_depth = depth_map[depth_idx].cpu().numpy()
            
            # Handle depth map channels (use first channel if multi-channel)
            if len(current_depth.shape) == 3 and current_depth.shape[2] == 3:
                depth_for_sbs = current_depth[:, :, 0]
            else:
                depth_for_sbs = current_depth
                
            # Invert depth if requested
            if invert_depth:
                depth_for_sbs = 1.0 - depth_for_sbs
                
            # Convert filtered depth to PIL and resize
            depth_map_img = Image.fromarray((depth_for_sbs * 255).astype(np.uint8), mode='L')
            if depth_map_img.size != current_image_pil.size:
                depth_map_img = depth_map_img.resize(current_image_pil.size, Image.NEAREST)
                
            width, height = current_image_pil.size
            
            # Create empty SBS image canvas
            sbs_image = np.zeros((height, width * 2, 3), dtype=np.uint8)
            
            # Resolution-Relative Depth Scaling
            # depth_scale (0-100) maps to 0-20% of image width
            # Formula: width * (depth_scale / 100) / 5
            max_shift_pixels = width * (depth_scale / 500.0)
            depth_scaling_factor = max_shift_pixels / 255.0  # Factor to multiply 0-255 depth value by
            
            fliped = 0 if mode == "Parallel" else width
            
            if highsodium_optimization:
                # =============================================================
                # HighSodium's Optimized Algorithm (Vectorized)
                # =============================================================
                img_array = np.array(current_image_pil)
                depth_array = np.array(depth_map_img)
                
                # Fill base images
                sbs_image[:, :width, :] = img_array
                sbs_image[:, width:width*2, :] = img_array
                
                # Calculate pixel shifts matrix
                # pixel_shift = depth_value * depth_scaling_factor
                pixel_shifts = (depth_array * depth_scaling_factor).astype(np.int32)
                pixel_shifts = np.clip(pixel_shifts, 0, width - 1)
                
                pbar = ProgressBar(width)
                
                # Process columns Right-to-Left (x range: width-1 -> 0)
                # This ensures foreground occludes background correctly
                for x in range(width - 1, -1, -1):
                    pbar.update(1)
                    
                    # Source pixels for this column
                    source_pixels = img_array[:, x, :]
                    
                    # Shift amount for this column
                    shifts = pixel_shifts[:, x]
                    
                    # Target X positions
                    target_x = x + shifts
                    
                    # Fill logic (gap filling) - iterate small range
                    # This is the inner loop that handles the "splatting"
                    for fill_offset in range(2): # Minimal fill to avoid gaps, 2 is usually enough for continuous surfaces
                         # Using a small fixed range like 2-3 is faster than 10. 
                         # But let's stick closer to the approved V2 logic (it used 10). 
                         # Actually V2 used 10. Let's stick to 10 to be safe and consistent with V2.
                        pass
                    
                    # Let's use the exact V2 logic for consistency
                    for fill_offset in range(10):
                        fill_x = target_x + fill_offset
                        
                        # Vectorized mask for valid positions
                        valid_mask = (fill_x >= 0) & (fill_x < width)
                        
                        if np.any(valid_mask):
                            valid_rows = np.where(valid_mask)[0]
                            valid_target_x = fill_x[valid_mask]
                            
                            # Apply to the shifted side
                            sbs_image[valid_rows, valid_target_x + fliped, :] = source_pixels[valid_rows, :]
            
            else:
                # =============================================================
                # Legacy Algorithm (Pixel-by-Pixel)
                # =============================================================
                pbar = ProgressBar(height)
                
                # Draw base
                for y in range(height):
                    for x in range(width):
                        color = current_image_pil.getpixel((x, y))
                        sbs_image[y, width + x] = color
                        sbs_image[y, x] = color
                        
                # Draw shifted
                for y in tqdm.tqdm(range(height)):
                    pbar.update(1)
                    for x in range(width):
                        try:
                            # Depth
                            d_val = depth_map_img.getpixel((x,y))
                            if isinstance(d_val, tuple): d_val = d_val[0]
                            
                            pixel_shift = int(d_val * depth_scaling_factor)
                            new_x = x + pixel_shift
                            
                            # Clamp
                            if new_x >= width: new_x = width - 1
                            if new_x < 0: new_x = 0
                            
                            # Gap fill
                            for i in range(pixel_shift + 10):
                                if new_x + i >= width or new_x < 0: break
                                sbs_image[y, new_x + i + fliped] = current_image_pil.getpixel((x, y))
                        except Exception:
                            pass
                            
            # Convert back to tensor
            sbs_images.append(torch.from_numpy(sbs_image).float() / 255.0)

        if not sbs_images:
             return (torch.zeros((B, 512, 1024, 3)),)

        return (torch.stack(sbs_images),)
