import cv2
import numpy as np
from r_facelib.parsing.mediapipe_mesh import MediaPipeFaceMesh


# Global MediaPipe parser for face mask generation
_mediapipe_parser = None

def get_mediapipe_parser():
    """Get or create global MediaPipe parser instance"""
    global _mediapipe_parser
    if _mediapipe_parser is None:
        _mediapipe_parser = MediaPipeFaceMesh(
            static_image_mode=True,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
    return _mediapipe_parser


def create_mediapipe_mask(image, fallback_mask=None):
    """
    Create precise face mask using MediaPipe 468-point landmarks
    
    Args:
        image: Face image (BGR format)
        fallback_mask: Original mask to use if MediaPipe fails
        
    Returns:
        Precise face mask or fallback mask
    """
    try:
        parser = get_mediapipe_parser()
        mp_mask = parser.create_face_mask(image, dilate_pixels=5)
        
        if mp_mask is not None:
            # Smooth the mask edges
            mp_mask = cv2.GaussianBlur(mp_mask, (7, 7), 2)
            return mp_mask
        else:
            # Fallback to original mask if MediaPipe fails
            return fallback_mask if fallback_mask is not None else np.ones_like(image[:,:,0]) * 255
            
    except Exception as e:
        print(f"MediaPipe mask creation failed: {e}")
        return fallback_mask if fallback_mask is not None else np.ones_like(image[:,:,0]) * 255


def adaptive_laplacian_pyramid_blend(source_roi, target_roi, mask, max_levels=6):
    """
    Enhanced Laplacian pyramid blending with adaptive levels and edge-aware processing.
    
    Args:
        source_roi: Source face region (BGR)
        target_roi: Target face region (BGR) 
        mask: Binary mask for blending region
        max_levels: Maximum number of pyramid levels
    
    Returns:
        Blended result
    """
    # Ensure all inputs are same size
    h, w = target_roi.shape[:2]
    source_roi = cv2.resize(source_roi, (w, h))
    mask = cv2.resize(mask, (w, h))
    
    # Convert mask to 3-channel float
    if len(mask.shape) == 2:
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask = mask.astype(np.float32) / 255.0
    
    # Adaptive level calculation based on image size
    levels = min(max_levels, int(np.log2(min(h, w)) - 2))
    levels = max(2, levels)  # Minimum 2 levels
    
    # Build Gaussian pyramids with edge preservation
    source_pyr = [source_roi.astype(np.float32)]
    target_pyr = [target_roi.astype(np.float32)]
    mask_pyr = [mask]
    
    for i in range(levels):
        # Use INTER_AREA for downsampling to preserve edges
        source_pyr.append(cv2.pyrDown(source_pyr[i], dstsize=(source_pyr[i].shape[1]//2, source_pyr[i].shape[0]//2)))
        target_pyr.append(cv2.pyrDown(target_pyr[i], dstsize=(target_pyr[i].shape[1]//2, target_pyr[i].shape[0]//2)))
        mask_pyr.append(cv2.pyrDown(mask_pyr[i], dstsize=(mask_pyr[i].shape[1]//2, mask_pyr[i].shape[0]//2)))
    
    # Build Laplacian pyramids with enhanced detail preservation
    source_lap = [source_pyr[levels].astype(np.float32)]
    target_lap = [target_pyr[levels].astype(np.float32)]
    
    for i in range(levels, 0, -1):
        size = (source_pyr[i-1].shape[1], source_pyr[i-1].shape[0])
        source_expanded = cv2.pyrUp(source_pyr[i], dstsize=size).astype(np.float32)
        target_expanded = cv2.pyrUp(target_pyr[i], dstsize=size).astype(np.float32)
        
        # Enhanced detail preservation with edge-aware subtraction
        source_lap.append((source_pyr[i-1] - source_expanded).astype(np.float32))
        target_lap.append((target_pyr[i-1] - target_expanded).astype(np.float32))
    
    # Multi-scale blending with edge-aware masks
    blended_lap = []
    for i in range(levels + 1):
        level_mask = mask_pyr[levels - i] if i < levels else mask_pyr[0]
        
        # Edge-aware blending using gradient magnitude
        if i > 0:  # Skip coarsest level
            # Calculate gradient magnitude for edge detection
            # Ensure proper data types for Laplacian operation
            source_lap_float = source_lap[i].astype(np.float64)
            target_lap_float = target_lap[i].astype(np.float64)
            
            source_grad = cv2.Laplacian(source_lap_float, cv2.CV_64F)
            target_grad = cv2.Laplacian(target_lap_float, cv2.CV_64F)
            
            # Create edge-aware blending weights
            edge_weight = np.clip(np.abs(source_grad) / (np.abs(source_grad) + np.abs(target_grad) + 1e-8), 0, 1)
            edge_weight = np.mean(edge_weight, axis=2, keepdims=True)
            
            # Combine edge-aware and regular blending
            blended = source_lap[i] * level_mask * edge_weight + target_lap[i] * (1 - level_mask * edge_weight)
        else:
            blended = source_lap[i] * level_mask + target_lap[i] * (1 - level_mask)
        
        blended_lap.append(blended)
    
    # Reconstruct with enhanced detail preservation
    result = blended_lap[0].astype(np.float32)
    for i in range(1, levels + 1):
        size = (blended_lap[i].shape[1], blended_lap[i].shape[0])
        result = cv2.pyrUp(result, dstsize=size).astype(np.float32)
        result = cv2.add(result, blended_lap[i].astype(np.float32))
    
    return np.clip(result, 0, 255).astype(np.uint8)


def enhanced_poisson_blend(source_roi, target_roi, mask):
    """
    Enhanced Poisson blending with improved center calculation and multi-scale processing.
    
    Args:
        source_roi: Source face region (BGR)
        target_roi: Target face region (BGR)
        mask: Binary mask for blending region
        
    Returns:
        Blended result
    """
    try:
        # Ensure mask is properly formatted
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        
        # Ensure mask is binary
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # Enhanced center calculation using mask centroid
        mask_indices = np.where(mask > 0)
        if len(mask_indices[0]) > 0:
            # Use weighted centroid for better center calculation
            y_coords = mask_indices[0]
            x_coords = mask_indices[1]
            weights = mask[mask_indices]
            
            center_y = int(np.average(y_coords, weights=weights))
            center_x = int(np.average(x_coords, weights=weights))
            center = (center_x, center_y)
        else:
            # Fallback to geometric center
            center = (mask.shape[1] // 2, mask.shape[0] // 2)
        
        # Ensure center is within bounds
        center = (max(0, min(center[0], mask.shape[1] - 1)), 
                 max(0, min(center[1], mask.shape[0] - 1)))
        
        # Multi-scale Poisson blending for better detail preservation
        if min(mask.shape[:2]) > 256:
            # For large regions, use multi-scale approach
            scale_factor = 0.5
            small_source = cv2.resize(source_roi, None, fx=scale_factor, fy=scale_factor)
            small_target = cv2.resize(target_roi, None, fx=scale_factor, fy=scale_factor)
            small_mask = cv2.resize(mask, None, fx=scale_factor, fy=scale_factor)
            small_center = (int(center[0] * scale_factor), int(center[1] * scale_factor))
            
            # Ensure small center is within bounds
            small_center = (max(0, min(small_center[0], small_mask.shape[1] - 1)), 
                           max(0, min(small_center[1], small_mask.shape[0] - 1)))
            
            # Blend at smaller scale
            small_result = cv2.seamlessClone(small_source, small_target, small_mask, small_center, cv2.NORMAL_CLONE)
            
            # Upscale and refine
            result = cv2.resize(small_result, (target_roi.shape[1], target_roi.shape[0]))
            
            # Apply final refinement at full resolution
            result = cv2.seamlessClone(source_roi, result, mask, center, cv2.MIXED_CLONE)
        else:
            # Direct blending for smaller regions
            result = cv2.seamlessClone(source_roi, target_roi, mask, center, cv2.NORMAL_CLONE)
            
        return result
        
    except Exception as e:
        print(f"Enhanced Poisson blending failed: {e}")
        try:
            # Try fallback Poisson blending
            return poisson_blend_fallback(source_roi, target_roi, mask)
        except Exception as e2:
            print(f"Poisson fallback also failed: {e2}")
            # Final fallback to simple alpha blending
            mask_norm = mask.astype(np.float32) / 255.0
            if len(mask_norm.shape) == 2:
                mask_norm = cv2.cvtColor(mask_norm, cv2.COLOR_GRAY2BGR)
            return (source_roi * mask_norm + target_roi * (1 - mask_norm)).astype(np.uint8)


def poisson_blend_fallback(source_roi, target_roi, mask):
    """Enhanced fallback Poisson blending with better center calculation and error handling"""
    try:
        # Ensure mask is properly formatted
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        
        # Ensure mask is binary
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # Enhanced center calculation using mask centroid
        mask_indices = np.where(mask > 0)
        if len(mask_indices[0]) > 0:
            # Use weighted centroid for better center calculation
            y_coords = mask_indices[0]
            x_coords = mask_indices[1]
            weights = mask[mask_indices]
            
            center_y = int(np.average(y_coords, weights=weights))
            center_x = int(np.average(x_coords, weights=weights))
            center = (center_x, center_y)
        else:
            # Fallback to geometric center
            center = (mask.shape[1] // 2, mask.shape[0] // 2)
        
        # Ensure center is within bounds
        center = (max(0, min(center[0], mask.shape[1] - 1)), 
                 max(0, min(center[1], mask.shape[0] - 1)))
            
        return cv2.seamlessClone(source_roi, target_roi, mask, center, cv2.NORMAL_CLONE)
    except Exception as e:
        print(f"Poisson blend fallback failed: {e}")
        # Simple alpha blend fallback
        mask_norm = mask.astype(np.float32) / 255.0
        if len(mask_norm.shape) == 2:
            mask_norm = cv2.cvtColor(mask_norm, cv2.COLOR_GRAY2BGR)
        return (source_roi * mask_norm + target_roi * (1 - mask_norm)).astype(np.uint8)


def style_aware_blend(source_roi, target_roi, mask):
    """
    Style-aware blending that preserves target image characteristics while transferring source details.
    
    Args:
        source_roi: Source face region (BGR)
        target_roi: Target face region (BGR)
        mask: Binary mask for blending region
        
    Returns:
        Style-aware blended result
    """
    try:
        # Convert to LAB color space for better color transfer
        source_lab = cv2.cvtColor(source_roi, cv2.COLOR_BGR2LAB)
        target_lab = cv2.cvtColor(target_roi, cv2.COLOR_BGR2LAB)
        
        # Calculate color statistics for style transfer
        mask_float = mask.astype(np.float32) / 255.0
        if len(mask_float.shape) == 2:
            mask_float = cv2.cvtColor(mask_float, cv2.COLOR_GRAY2BGR)
        
        # Extract masked regions
        source_masked = source_lab * mask_float
        target_masked = target_lab * mask_float
        
        # Calculate mean and std for color transfer
        source_mean = np.mean(source_masked, axis=(0, 1))
        source_std = np.std(source_masked, axis=(0, 1))
        target_mean = np.mean(target_masked, axis=(0, 1))
        target_std = np.std(target_masked, axis=(0, 1))
        
        # Avoid division by zero and extreme values
        source_std = np.where(source_std == 0, 1, source_std)
        target_std = np.where(target_std == 0, 1, target_std)
        
        # Limit std values to prevent extreme color shifts
        source_std = np.clip(source_std, 1, 50)
        target_std = np.clip(target_std, 1, 50)
        
        # Apply color transfer with improved algorithm
        source_normalized = (source_lab - source_mean) / source_std
        source_transferred = source_normalized * target_std + target_mean
        
        # Clip values to valid LAB range
        source_transferred = np.clip(source_transferred, 0, 255)
        
        # Convert back to BGR
        source_style_transferred = cv2.cvtColor(source_transferred.astype(np.uint8), cv2.COLOR_LAB2BGR)
        
        # Blend with original target using the mask
        result = source_style_transferred * mask_float + target_roi * (1 - mask_float)
        
        return result.astype(np.uint8)
        
    except Exception as e:
        print(f"Style-aware blending failed: {e}")
        # Fallback to simple alpha blending
        mask_norm = mask.astype(np.float32) / 255.0
        if len(mask_norm.shape) == 2:
            mask_norm = cv2.cvtColor(mask_norm, cv2.COLOR_GRAY2BGR)
        return (source_roi * mask_norm + target_roi * (1 - mask_norm)).astype(np.uint8)


def edge_aware_blend(source_roi, target_roi, mask):
    """
    Edge-aware blending that preserves important edges and details.
    
    Args:
        source_roi: Source face region (BGR)
        target_roi: Target face region (BGR)
        mask: Binary mask for blending region
        
    Returns:
        Edge-aware blended result
    """
    try:
        # Convert to grayscale for edge detection
        source_gray = cv2.cvtColor(source_roi, cv2.COLOR_BGR2GRAY)
        target_gray = cv2.cvtColor(target_roi, cv2.COLOR_BGR2GRAY)
        
        # Enhanced edge detection with adaptive thresholds
        source_mean = np.mean(source_gray)
        target_mean = np.mean(target_gray)
        
        # Adaptive Canny thresholds based on image brightness
        low_threshold = max(30, min(80, int(min(source_mean, target_mean) * 0.3)))
        high_threshold = max(100, min(200, int(max(source_mean, target_mean) * 0.7)))
        
        # Detect edges using Canny with adaptive thresholds
        source_edges = cv2.Canny(source_gray, low_threshold, high_threshold)
        target_edges = cv2.Canny(target_gray, low_threshold, high_threshold)
        
        # Create edge-aware mask with improved processing
        edge_mask = cv2.bitwise_or(source_edges, target_edges)
        
        # Dilate edges to create wider edge regions
        kernel = np.ones((3, 3), np.uint8)
        edge_mask = cv2.dilate(edge_mask, kernel, iterations=2)
        
        # Apply Gaussian blur for smoother edge transitions
        edge_mask = cv2.GaussianBlur(edge_mask.astype(np.float32), (5, 5), 0)
        
        # Normalize edge mask
        if edge_mask.max() > 0:
            edge_mask = edge_mask / edge_mask.max()
        
        # Convert to 3-channel
        edge_mask = cv2.cvtColor(edge_mask.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        edge_mask = edge_mask.astype(np.float32) / 255.0
        
        # Combine original mask with edge mask
        combined_mask = mask.astype(np.float32) / 255.0
        if len(combined_mask.shape) == 2:
            combined_mask = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR)
        
        # Edge-aware blending with adaptive weights
        edge_weight = 0.3  # Weight for edge preservation
        final_mask = combined_mask * (1 - edge_weight) + edge_mask * edge_weight
        
        # Apply blending with edge preservation
        result = source_roi * final_mask + target_roi * (1 - final_mask)
        
        return result.astype(np.uint8)
        
    except Exception as e:
        print(f"Edge-aware blending failed: {e}")
        # Fallback to simple alpha blending
        mask_norm = mask.astype(np.float32) / 255.0
        if len(mask_norm.shape) == 2:
            mask_norm = cv2.cvtColor(mask_norm, cv2.COLOR_GRAY2BGR)
        return (source_roi * mask_norm + target_roi * (1 - mask_norm)).astype(np.uint8)


def multi_scale_blend(source_roi, target_roi, mask):
    """
    Multi-scale blending for optimal detail preservation at different scales.
    
    Args:
        source_roi: Source face region (BGR)
        target_roi: Target face region (BGR)
        mask: Binary mask for blending region
        
    Returns:
        Multi-scale blended result
    """
    try:
        # Define scales for multi-scale processing with adaptive selection
        h, w = source_roi.shape[:2]
        
        # Adaptive scale selection based on image size
        if min(h, w) < 64:
            scales = [1.0]  # Single scale for very small images
        elif min(h, w) < 128:
            scales = [1.0, 0.5]  # Two scales for small images
        else:
            scales = [1.0, 0.5, 0.25]  # Three scales for larger images
        
        results = []
        
        for scale in scales:
            if scale == 1.0:
                current_source = source_roi
                current_target = target_roi
                current_mask = mask
            else:
                # Resize for current scale with improved interpolation
                new_h, new_w = int(h * scale), int(w * scale)
                
                current_source = cv2.resize(source_roi, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
                current_target = cv2.resize(target_roi, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
                current_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            
            # Apply Laplacian pyramid blending at current scale
            try:
                current_result = adaptive_laplacian_pyramid_blend(current_source, current_target, current_mask)
            except Exception as e:
                print(f"Laplacian pyramid blending failed at scale {scale}: {e}")
                # Fallback to simple alpha blending for this scale
                mask_norm = current_mask.astype(np.float32) / 255.0
                if len(mask_norm.shape) == 2:
                    mask_norm = cv2.cvtColor(mask_norm, cv2.COLOR_GRAY2BGR)
                current_result = (current_source * mask_norm + current_target * (1 - mask_norm)).astype(np.uint8)
            
            if scale != 1.0:
                # Resize back to original size with improved interpolation
                current_result = cv2.resize(current_result, (w, h), interpolation=cv2.INTER_LANCZOS4)
            
            results.append(current_result)
        
        # Combine results from different scales with adaptive weights
        if len(scales) == 1:
            final_result = results[0]
        else:
            # Adaptive weights based on image characteristics
            if len(scales) == 2:
                weights = [0.7, 0.3]  # Higher weight for full resolution
            else:
                weights = [0.5, 0.3, 0.2]  # Weights for different scales
            
            final_result = np.zeros_like(source_roi, dtype=np.float32)
            
            for result, weight in zip(results, weights):
                final_result += result.astype(np.float32) * weight
        
        return np.clip(final_result, 0, 255).astype(np.uint8)
        
    except Exception as e:
        print(f"Multi-scale blending failed: {e}")
        # Fallback to simple alpha blending
        mask_norm = mask.astype(np.float32) / 255.0
        if len(mask_norm.shape) == 2:
            mask_norm = cv2.cvtColor(mask_norm, cv2.COLOR_GRAY2BGR)
        return (source_roi * mask_norm + target_roi * (1 - mask_norm)).astype(np.uint8)


def enhanced_swap_blend(source_face, target_face, source_roi, target_roi, transform_matrix):
    """
    Enhanced face swapping with multiple blending options
    """
    # Create mask from face landmarks or use ellipse approximation
    mask = np.zeros(target_roi.shape[:2], dtype=np.uint8)
    
    # Use face boundary or create elliptical mask
    center = (mask.shape[1] // 2, mask.shape[0] // 2)
    axes = (int(mask.shape[1] * 0.4), int(mask.shape[0] * 0.45))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    
    # Apply Gaussian blur to soften mask edges
    mask = cv2.GaussianBlur(mask, (21, 21), 0)
    
    try:
        # Try enhanced Laplacian pyramid blend first
        result = adaptive_laplacian_pyramid_blend(source_roi, target_roi, mask, max_levels=6)
    except:
        try:
            # Fallback to enhanced Poisson blend
            result = enhanced_poisson_blend(source_roi, target_roi, mask)
        except:
            # Final fallback to simple alpha blending
            mask_norm = mask.astype(np.float32) / 255.0
            if len(mask_norm.shape) == 2:
                mask_norm = cv2.cvtColor(mask_norm, cv2.COLOR_GRAY2BGR)
            result = (source_roi * mask_norm + target_roi * (1 - mask_norm)).astype(np.uint8)
    
    return result


def in_swap(img, bgr_fake, M, blend_method="original"):
    """
    Enhanced face swapping with multiple blend methods
    
    Args:
        img: Target image
        bgr_fake: Source face
        M: Transform matrix
        blend_method: "original", "pyramid", "poisson", "style", "edge", "multiscale", or "adaptive"
    """
    if blend_method == "pyramid":
        return enhanced_pyramid_swap(img, bgr_fake, M)
    elif blend_method == "poisson": 
        return enhanced_poisson_swap(img, bgr_fake, M)
    elif blend_method == "style":
        return enhanced_style_swap(img, bgr_fake, M)
    elif blend_method == "edge":
        return enhanced_edge_swap(img, bgr_fake, M)
    elif blend_method == "multiscale":
        return enhanced_multiscale_swap(img, bgr_fake, M)
    elif blend_method == "adaptive":
        return enhanced_adaptive_swap(img, bgr_fake, M)
    else:
        return original_in_swap(img, bgr_fake, M)


def original_in_swap(img, bgr_fake, M):
    """Enhanced original INSwapper blending method with improved mask handling and sophisticated blending"""
    target_img = img
    IM = cv2.invertAffineTransform(M)
    img_white = np.full((bgr_fake.shape[0], bgr_fake.shape[1]), 255, dtype=np.float32)

    bgr_fake = cv2.warpAffine(bgr_fake, IM, (target_img.shape[1], target_img.shape[0]), borderValue=0.0, flags=cv2.INTER_CUBIC)
    img_white = cv2.warpAffine(img_white, IM, (target_img.shape[1], target_img.shape[0]), borderValue=0.0)
    
    # Create MediaPipe-enhanced mask for original blending
    mp_mask = create_mediapipe_mask(bgr_fake, fallback_mask=img_white.astype(np.uint8))
    
    # Warp the MediaPipe mask to match target image
    mp_mask_warped = cv2.warpAffine(mp_mask, IM, (target_img.shape[1], target_img.shape[0]), borderValue=0.0)
    
    # Enhanced mask processing
    img_mask = np.zeros_like(mp_mask_warped)
    img_mask[mp_mask_warped > 20] = 255  # Create clean binary mask
    
    # Get mask bounds
    mask_h_inds, mask_w_inds = np.where(img_mask == 255)
    
    if len(mask_h_inds) == 0:
        # No valid mask found, return original image
        return target_img
    
    mask_h = np.max(mask_h_inds) - np.min(mask_h_inds)
    mask_w = np.max(mask_w_inds) - np.min(mask_w_inds)
    mask_size = int(np.sqrt(mask_h * mask_w))
    
    # Adaptive kernel size based on mask size
    k = max(mask_size // 10, 10)
    
    # Enhanced erosion for better edge handling
    kernel = np.ones((k, k), np.uint8)
    img_mask = cv2.erode(img_mask, kernel, iterations=1)
    
    # Improved blur for smoother edges
    k = max(mask_size // 20, 5)
    kernel_size = (k, k)
    blur_size = tuple(2 * i + 1 for i in kernel_size)
    img_mask = cv2.GaussianBlur(img_mask, blur_size, 0)
    
    # Additional smoothing for better blending
    k = 5
    kernel_size = (k, k)
    blur_size = tuple(2 * i + 1 for i in kernel_size)
    img_mask = cv2.GaussianBlur(img_mask, blur_size, 0)
    
    # Convert to float and reshape for blending
    img_mask = img_mask.astype(np.float32) / 255
    img_mask = np.reshape(img_mask, [img_mask.shape[0], img_mask.shape[1], 1])
    
    # Enhanced blending with edge preservation and color correction
    try:
        # Try sophisticated blending first
        # Apply color correction to match target lighting
        target_mean = np.mean(target_img.astype(np.float32), axis=(0, 1))
        source_mean = np.mean(bgr_fake.astype(np.float32), axis=(0, 1))
        
        # Color correction factor
        color_correction = target_mean / (source_mean + 1e-8)
        color_correction = np.clip(color_correction, 0.5, 2.0)  # Limit correction range
        
        # Apply color correction
        corrected_bgr_fake = np.clip(bgr_fake.astype(np.float32) * color_correction, 0, 255)
        
        # Enhanced blending with edge preservation
        fake_merged = img_mask * corrected_bgr_fake + (1 - img_mask) * target_img.astype(np.float32)
        
        # Apply additional smoothing at mask edges
        edge_kernel = np.ones((3, 3), np.float32) / 9
        edge_mask = cv2.filter2D(img_mask, -1, edge_kernel)
        
        # Final blending with edge smoothing
        final_result = edge_mask * fake_merged + (1 - edge_mask) * target_img.astype(np.float32)
        
        return np.clip(final_result, 0, 255).astype(np.uint8)
        
    except Exception as e:
        print(f"Enhanced original blending failed: {e}")
        # Fallback to simple alpha blending
        fake_merged = img_mask * bgr_fake + (1 - img_mask) * target_img.astype(np.float32)
        return fake_merged.astype(np.uint8)


def enhanced_pyramid_swap(img, bgr_fake, M):
    """Enhanced swap using adaptive Laplacian pyramid with improved mask handling"""
    target_img = img
    IM = cv2.invertAffineTransform(M)
    
    # Warp source face
    bgr_fake_warped = cv2.warpAffine(bgr_fake, IM, (target_img.shape[1], target_img.shape[0]), 
                                     borderValue=0.0, flags=cv2.INTER_CUBIC)
    
    # Create mask
    img_white = np.full((bgr_fake.shape[0], bgr_fake.shape[1]), 255, dtype=np.float32)
    mask = cv2.warpAffine(img_white, IM, (target_img.shape[1], target_img.shape[0]), borderValue=0.0)
    
    # Enhanced mask processing
    mask_binary = np.zeros_like(mask)
    mask_binary[mask > 20] = 255  # Create clean binary mask
    mask = mask_binary  # Use binary mask
    
    # Get face region with bounds checking
    mask_indices = np.where(mask == 255)
    if len(mask_indices[0]) == 0:
        return target_img
        
    y_min, y_max = np.min(mask_indices[0]), np.max(mask_indices[0])
    x_min, x_max = np.min(mask_indices[1]), np.max(mask_indices[1])
    
    # Ensure bounds are within image
    y_min = max(0, y_min)
    y_max = min(target_img.shape[0], y_max)
    x_min = max(0, x_min)
    x_max = min(target_img.shape[1], x_max)
    
    if y_max <= y_min or x_max <= x_min:
        return target_img
    
    # Extract ROIs
    target_roi = target_img[y_min:y_max, x_min:x_max]
    source_roi = bgr_fake_warped[y_min:y_max, x_min:x_max]
    mask_roi = mask[y_min:y_max, x_min:x_max]
    
    # Create MediaPipe-enhanced mask for precise face contours
    mp_mask_roi = create_mediapipe_mask(source_roi, fallback_mask=mask_roi.astype(np.uint8))
    
    # Ensure mask size matches ROI
    if mp_mask_roi.shape[:2] != mask_roi.shape[:2]:
        mp_mask_roi = cv2.resize(mp_mask_roi, (mask_roi.shape[1], mask_roi.shape[0]))
    
    # Apply adaptive pyramid blend with MediaPipe-enhanced mask
    blended_roi = adaptive_laplacian_pyramid_blend(source_roi, target_roi, mp_mask_roi, max_levels=6)
    
    # Place back in full image
    result = target_img.copy()
    result[y_min:y_max, x_min:x_max] = blended_roi
    return result


def enhanced_poisson_swap(img, bgr_fake, M):
    """Enhanced swap using improved Poisson blending with better error handling"""
    target_img = img
    IM = cv2.invertAffineTransform(M)
    
    bgr_fake_warped = cv2.warpAffine(bgr_fake, IM, (target_img.shape[1], target_img.shape[0]), 
                                     borderValue=0.0, flags=cv2.INTER_CUBIC)
    
    img_white = np.full((bgr_fake.shape[0], bgr_fake.shape[1]), 255, dtype=np.float32)
    mask = cv2.warpAffine(img_white, IM, (target_img.shape[1], target_img.shape[0]), borderValue=0.0)
    
    # Create MediaPipe-enhanced mask for Poisson blending
    mp_mask = create_mediapipe_mask(bgr_fake_warped, fallback_mask=mask.astype(np.uint8))
    
    # Enhanced mask processing
    mask_binary = np.zeros_like(mp_mask)
    mask_binary[mp_mask > 20] = 255  # Create clean binary mask
    
    # Get face region with bounds checking
    mask_indices = np.where(mask_binary == 255)
    if len(mask_indices[0]) == 0:
        return target_img
        
    y_min, y_max = np.min(mask_indices[0]), np.max(mask_indices[0])
    x_min, x_max = np.min(mask_indices[1]), np.max(mask_indices[1])
    
    # Ensure bounds are within image
    y_min = max(0, y_min)
    y_max = min(target_img.shape[0], y_max)
    x_min = max(0, x_min)
    x_max = min(target_img.shape[1], x_max)
    
    if y_max <= y_min or x_max <= x_min:
        return target_img
    
    # Extract ROIs
    target_roi = target_img[y_min:y_max, x_min:x_max]
    source_roi = bgr_fake_warped[y_min:y_max, x_min:x_max]
    mask_roi = mask_binary[y_min:y_max, x_min:x_max]
    
    return enhanced_poisson_blend(source_roi, target_roi, mask_roi.astype(np.uint8))


def enhanced_style_swap(img, bgr_fake, M):
    """Enhanced swap using style-aware blending"""
    target_img = img
    IM = cv2.invertAffineTransform(M)
    
    bgr_fake_warped = cv2.warpAffine(bgr_fake, IM, (target_img.shape[1], target_img.shape[0]), 
                                     borderValue=0.0, flags=cv2.INTER_CUBIC)
    
    img_white = np.full((bgr_fake.shape[0], bgr_fake.shape[1]), 255, dtype=np.float32)
    mask = cv2.warpAffine(img_white, IM, (target_img.shape[1], target_img.shape[0]), borderValue=0.0)
    
    # Create MediaPipe-enhanced mask
    mp_mask = create_mediapipe_mask(bgr_fake_warped, fallback_mask=mask.astype(np.uint8))
    
    # Fix: Proper binary mask
    mask_binary = np.zeros_like(mp_mask)
    mask_binary[mp_mask > 20] = 255
    
    # Get face region
    mask_indices = np.where(mask_binary == 255)
    if len(mask_indices[0]) == 0:
        return target_img
        
    y_min, y_max = np.min(mask_indices[0]), np.max(mask_indices[0])
    x_min, x_max = np.min(mask_indices[1]), np.max(mask_indices[1])
    
    # Extract ROIs
    target_roi = target_img[y_min:y_max, x_min:x_max]
    source_roi = bgr_fake_warped[y_min:y_max, x_min:x_max]
    mask_roi = mask_binary[y_min:y_max, x_min:x_max]
    
    # Apply style-aware blending
    blended_roi = style_aware_blend(source_roi, target_roi, mask_roi)
    
    # Place back in full image
    result = target_img.copy()
    result[y_min:y_max, x_min:x_max] = blended_roi
    return result


def enhanced_edge_swap(img, bgr_fake, M):
    """Enhanced swap using edge-aware blending"""
    target_img = img
    IM = cv2.invertAffineTransform(M)
    
    bgr_fake_warped = cv2.warpAffine(bgr_fake, IM, (target_img.shape[1], target_img.shape[0]), 
                                     borderValue=0.0, flags=cv2.INTER_CUBIC)
    
    img_white = np.full((bgr_fake.shape[0], bgr_fake.shape[1]), 255, dtype=np.float32)
    mask = cv2.warpAffine(img_white, IM, (target_img.shape[1], target_img.shape[0]), borderValue=0.0)
    
    # Create MediaPipe-enhanced mask
    mp_mask = create_mediapipe_mask(bgr_fake_warped, fallback_mask=mask.astype(np.uint8))
    
    # Fix: Proper binary mask
    mask_binary = np.zeros_like(mp_mask)
    mask_binary[mp_mask > 20] = 255
    
    # Get face region
    mask_indices = np.where(mask_binary == 255)
    if len(mask_indices[0]) == 0:
        return target_img
        
    y_min, y_max = np.min(mask_indices[0]), np.max(mask_indices[0])
    x_min, x_max = np.min(mask_indices[1]), np.max(mask_indices[1])
    
    # Extract ROIs
    target_roi = target_img[y_min:y_max, x_min:x_max]
    source_roi = bgr_fake_warped[y_min:y_max, x_min:x_max]
    mask_roi = mask_binary[y_min:y_max, x_min:x_max]
    
    # Apply edge-aware blending
    blended_roi = edge_aware_blend(source_roi, target_roi, mask_roi)
    
    # Place back in full image
    result = target_img.copy()
    result[y_min:y_max, x_min:x_max] = blended_roi
    return result


def enhanced_multiscale_swap(img, bgr_fake, M):
    """Enhanced swap using multi-scale blending"""
    target_img = img
    IM = cv2.invertAffineTransform(M)
    
    bgr_fake_warped = cv2.warpAffine(bgr_fake, IM, (target_img.shape[1], target_img.shape[0]), 
                                     borderValue=0.0, flags=cv2.INTER_CUBIC)
    
    img_white = np.full((bgr_fake.shape[0], bgr_fake.shape[1]), 255, dtype=np.float32)
    mask = cv2.warpAffine(img_white, IM, (target_img.shape[1], target_img.shape[0]), borderValue=0.0)
    
    # Create MediaPipe-enhanced mask
    mp_mask = create_mediapipe_mask(bgr_fake_warped, fallback_mask=mask.astype(np.uint8))
    
    # Fix: Proper binary mask
    mask_binary = np.zeros_like(mp_mask)
    mask_binary[mp_mask > 20] = 255
    
    # Get face region
    mask_indices = np.where(mask_binary == 255)
    if len(mask_indices[0]) == 0:
        return target_img
        
    y_min, y_max = np.min(mask_indices[0]), np.max(mask_indices[0])
    x_min, x_max = np.min(mask_indices[1]), np.max(mask_indices[1])
    
    # Extract ROIs
    target_roi = target_img[y_min:y_max, x_min:x_max]
    source_roi = bgr_fake_warped[y_min:y_max, x_min:x_max]
    mask_roi = mask_binary[y_min:y_max, x_min:x_max]
    
    # Apply multi-scale blending
    blended_roi = multi_scale_blend(source_roi, target_roi, mask_roi)
    
    # Place back in full image
    result = target_img.copy()
    result[y_min:y_max, x_min:x_max] = blended_roi
    return result


def enhanced_adaptive_swap(img, bgr_fake, M):
    """Enhanced swap using adaptive blending method selection"""
    target_img = img
    IM = cv2.invertAffineTransform(M)
    
    bgr_fake_warped = cv2.warpAffine(bgr_fake, IM, (target_img.shape[1], target_img.shape[0]), 
                                     borderValue=0.0, flags=cv2.INTER_CUBIC)
    
    img_white = np.full((bgr_fake.shape[0], bgr_fake.shape[1]), 255, dtype=np.float32)
    mask = cv2.warpAffine(img_white, IM, (target_img.shape[1], target_img.shape[0]), borderValue=0.0)
    
    # Create MediaPipe-enhanced mask
    mp_mask = create_mediapipe_mask(bgr_fake_warped, fallback_mask=mask.astype(np.uint8))
    
    # Fix: Proper binary mask
    mask_binary = np.zeros_like(mp_mask)
    mask_binary[mp_mask > 20] = 255
    
    # Get face region
    mask_indices = np.where(mask_binary == 255)
    if len(mask_indices[0]) == 0:
        return target_img
        
    y_min, y_max = np.min(mask_indices[0]), np.max(mask_indices[0])
    x_min, x_max = np.min(mask_indices[1]), np.max(mask_indices[1])
    
    # Extract ROIs
    target_roi = target_img[y_min:y_max, x_min:x_max]
    source_roi = bgr_fake_warped[y_min:y_max, x_min:x_max]
    mask_roi = mask_binary[y_min:y_max, x_min:x_max]
    
    # Adaptive method selection based on image characteristics
    h, w = target_roi.shape[:2]
    
    # For small faces, use simple blending
    if h < 64 or w < 64:
        blended_roi = enhanced_poisson_blend(source_roi, target_roi, mask_roi)
    # For medium faces, use pyramid blending
    elif h < 128 or w < 128:
        blended_roi = adaptive_laplacian_pyramid_blend(source_roi, target_roi, mask_roi, max_levels=4)
    # For large faces, use multi-scale blending
    else:
        blended_roi = multi_scale_blend(source_roi, target_roi, mask_roi)
    
    # Place back in full image
    result = target_img.copy()
    result[y_min:y_max, x_min:x_max] = blended_roi
    return result


def smooth_mask_edges(mask, blur_radius=15):
    """Soften hard mask edges to eliminate square artifacts"""
    smooth_mask = cv2.GaussianBlur(mask, (blur_radius*2+1, blur_radius*2+1), blur_radius/3)
    return smooth_mask

def refine_mask_shape(mask):
    """Make square masks more face-like using morphological operations"""
    # Convert to uint8 if needed
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    
    # Remove square corners with elliptical kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Smooth edges
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

def create_distance_mask(mask):
    """Create soft falloff based on distance from mask center"""
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
        
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    if dist_transform.max() > 0:
        # Normalize and apply smooth falloff
        dist_mask = dist_transform / dist_transform.max()
        return np.power(dist_mask, 0.7).astype(np.float32)  # Adjust falloff curve
    return mask.astype(np.float32) / 255.0

def mask_the_mask(original_mask, method="smooth"):
    """Master function to refine masks and eliminate artifacts"""
    if method == "smooth":
        return smooth_mask_edges(original_mask)
    elif method == "morphology":
        return refine_mask_shape(original_mask)
    elif method == "distance":
        return create_distance_mask(original_mask)
    elif method == "combined":
        # Ultimate mask refinement
        mask = refine_mask_shape(original_mask)
        mask = smooth_mask_edges(mask, blur_radius=10)
        mask = create_distance_mask(mask)
        return mask
    else:
        return original_mask
