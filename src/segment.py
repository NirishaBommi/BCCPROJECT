import cv2
import numpy as np

def segment_lesion(image_path_or_arr):
    """
    Reads an image, segments the primary skin lesion contour,
    and returns a binary mask and a blended color overlay.
    """
    if isinstance(image_path_or_arr, str):
        img = cv2.imread(image_path_or_arr)
    else:
        img = image_path_or_arr.copy()
        
    if img is None:
        raise ValueError("Invalid image input for segmentation.")
        
    h, w, c = img.shape
    
    # 1. Convert to grayscale and blur
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    
    # 2. Otsu thresholding
    # Since lesions are typically darker than surrounding skin, we invert the thresholding
    # or detect contrast. In dermoscopy, we can do adaptive thresholding.
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Clean up threshold mask with morphological opening and closing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # 3. Find contours and keep only the largest one (the lesion)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    mask = np.zeros_like(thresh)
    if contours:
        # Find largest contour by area
        largest_contour = max(contours, key=cv2.contourArea)
        # Only keep if area is significant (otherwise fall back to threshold mask)
        if cv2.contourArea(largest_contour) > (h * w * 0.005):
            cv2.drawContours(mask, [largest_contour], -1, 255, -1)
        else:
            mask = thresh
    else:
        mask = thresh
        
    # Apply a final smoothing to the mask edges
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # 4. Generate the overlay image (visualizing boundaries with a bright clinical cyan tint)
    overlay = img.copy()
    
    # Create colored mask (clinical cyan/teal: B=254, G=242, R=0 in BGR)
    colored_mask = np.zeros_like(img)
    colored_mask[mask == 255] = [254, 242, 0] 
    
    # Blend mask overlay on top of original image
    cv2.addWeighted(colored_mask, 0.4, overlay, 0.6, 0, overlay)
    
    # Draw contour boundary lines (bright teal line)
    contours_clean, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours_clean, -1, (254, 242, 0), 2)
    
    return mask, overlay
