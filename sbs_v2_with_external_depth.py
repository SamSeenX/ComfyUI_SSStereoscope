import torch
from PIL import Image
import numpy as np
import tqdm
import cv2
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
                "gap_fill_mode": (["Inpaint (Telea)", "Stretch", "None"], {"default": "Inpaint (Telea)"}),
                "stereo_layout": (["Side by Side", "Top Bottom"], {"default": "Side by Side"}),
                "highsodium_optimization": ("BOOLEAN", {"default": True, "label_on": "Fast (HighSodium)", "label_off": "Legacy"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("stereoscopic_image", "gap_mask")
    FUNCTION = "process"
    CATEGORY = "👀 SamSeen"
    DESCRIPTION = "V2.2: Create stereoscopic images (Side-by-Side or Top-Bottom). Features 'Gap Filling' modes including Fast Inpainting (Telea)."

    def process(self, base_image, depth_map, depth_scale, blur_radius, invert_depth=False, mode="Cross-eyed", highsodium_optimization=True, gap_fill_mode="Inpaint (Telea)", stereo_layout="Side by Side"):
        """
        Create a stereoscopic image from a standard image and an external depth map.
        """
        
        # Get batch size
        B = base_image.shape[0]
        sbs_images = []
        gap_masks = []

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
            
            # Helper to generate views based on layout
            if stereo_layout == "Side by Side":
                final_w, final_h = width * 2, height
                sbs_image = np.zeros((final_h, final_w, 3), dtype=np.uint8)
                gap_mask = np.ones((final_h, final_w), dtype=np.float32)
                
                # Slices
                left_view = sbs_image[:, :width, :]
                right_view = sbs_image[:, width:, :]
                
                left_mask_view = gap_mask[:, :width]
                right_mask_view = gap_mask[:, width:]
                
            else: # Top Bottom
                final_w, final_h = width, height * 2
                sbs_image = np.zeros((final_h, final_w, 3), dtype=np.uint8)
                gap_mask = np.ones((final_h, final_w), dtype=np.float32)
                
                # Slices
                left_view = sbs_image[:height, :, :]
                right_view = sbs_image[height:, :, :]
                
                left_mask_view = gap_mask[:height, :]
                right_mask_view = gap_mask[height:, :]

            # Assign Views based on Mode
            # Parallel: Left Eye = Original (Static), Right Eye = Shifted (Active)
            # Cross-eyed: Right Eye = Original (Static), Left Eye = Shifted (Active)
            if mode == "Parallel":
                static_view = left_view
                active_view = right_view
                static_mask = left_mask_view
                active_mask = right_mask_view
            else: # Cross-eyed
                static_view = right_view
                active_view = left_view
                static_mask = right_mask_view
                active_mask = left_mask_view
            
            # Fill Static Eye
            static_view[:] = np.array(current_image_pil)
            static_mask[:] = 0.0 # Valid content
            
            # Resolution-Relative Depth Scaling
            max_shift_pixels = width * (depth_scale / 500.0)
            depth_scaling_factor = max_shift_pixels / 255.0

            # Fill Range
            fill_range = 10 if gap_fill_mode == "Stretch" else 1

            if highsodium_optimization:
                # =============================================================
                # HighSodium's Optimized Algorithm (Vectorized)
                # =============================================================
                img_array = np.array(current_image_pil)
                depth_array = np.array(depth_map_img)
                
                # Calculate pixel shifts matrix
                pixel_shifts = (depth_array * depth_scaling_factor).astype(np.int32)
                pixel_shifts = np.clip(pixel_shifts, 0, width - 1)
                
                pbar = ProgressBar(20) # Throttled to 20 updates per image
                update_step = max(1, width // 20)
                
                # Process columns Right-to-Left (x range: width-1 -> 0)
                for x in range(width - 1, -1, -1):
                    # Throttled update
                    if (width - 1 - x) % update_step == 0:
                        pbar.update(1)
                    
                    # Source pixels for this column
                    source_pixels = img_array[:, x, :]
                    
                    # Shift amount for this column
                    shifts = pixel_shifts[:, x]
                    
                    # Target X positions
                    target_x = x + shifts
                    
                    # Gap fill / Splatting logic
                    for fill_offset in range(fill_range):
                        fill_x = target_x + fill_offset
                        
                        # Vectorized mask for valid positions
                        valid_mask = (fill_x >= 0) & (fill_x < width)
                        
                        if np.any(valid_mask):
                            valid_rows = np.where(valid_mask)[0]
                            valid_target_x = fill_x[valid_mask]
                            
                            # Apply to the Active View directly (relative coords)
                            active_view[valid_rows, valid_target_x, :] = source_pixels[valid_rows, :]
                            
                            # Mark valid on mask
                            active_mask[valid_rows, valid_target_x] = 0.0
            
            else:
                # Legacy Algorithm (Pixel-by-Pixel)
                 pbar = ProgressBar(20) # Throttled to 20 updates
                 update_step = max(1, height // 20)
                 
                 for y in tqdm.tqdm(range(height)):
                    if y % update_step == 0:
                        pbar.update(1)
                    for x in range(width):
                        try:
                            d_val = depth_map_img.getpixel((x,y))
                            if isinstance(d_val, tuple): d_val = d_val[0]
                            
                            pixel_shift = int(d_val * depth_scaling_factor)
                            new_x = x + pixel_shift
                            
                            if new_x >= width: new_x = width - 1
                            if new_x < 0: new_x = 0
                            
                            for i in range(fill_range): # Use fill_range
                                if new_x + i >= width or new_x < 0: break
                                # Write to Active View
                                active_view[y, new_x + i] = current_image_pil.getpixel((x, y))
                                active_mask[y, new_x + i] = 0.0
                        except Exception:
                            pass

            # =============================================================
            # Post-Processing: Inpainting
            # =============================================================
            if gap_fill_mode == "Inpaint (Telea)":
                # active_view and active_mask are slices of the main arrays.
                # However, cv2.inpaint needs contiguous arrays for best safety/performance or creates copies.
                # Let's create a copy to operate on.
                eye_img_copy = active_view.copy()
                mask_uint8 = (active_mask * 255).astype(np.uint8)
                
                # Inpaint
                inpainted_eye = cv2.inpaint(eye_img_copy, mask_uint8, 3, cv2.INPAINT_TELEA)
                
                # Write back to the slice
                active_view[:] = inpainted_eye
            
            # Convert back to tensor
            sbs_images.append(torch.from_numpy(sbs_image).float() / 255.0)
            gap_masks.append(torch.from_numpy(gap_mask).float()) 

        if not sbs_images:
             return (torch.zeros((B, 512, 1024, 3)), torch.zeros((B, 512, 1024)))

        return (torch.stack(sbs_images), torch.stack(gap_masks))

