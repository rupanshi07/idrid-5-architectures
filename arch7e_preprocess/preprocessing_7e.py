"""
Configurable preprocessing pipeline implementing the faculty's ablation
plan: P0 (baseline) through P5. Each function operates on a PIL image;
ROI cropping returns a bounding box that MUST also be applied to the
segmentation mask, to keep image/mask perfectly aligned.
"""
import numpy as np
import cv2
from PIL import Image


def get_roi_bbox(image_pil, threshold=7):
    """Detects the retinal field by thresholding a grayscale version of
    the image, then returns the bounding box of the largest connected
    region -- removes black borders around the circular fundus scan."""
    gray = np.array(image_pil.convert("L"))
    mask = gray > threshold
    if mask.sum() == 0:
        return 0, 0, image_pil.width, image_pil.height
    ys, xs = np.where(mask)
    top, bottom = int(ys.min()), int(ys.max()) + 1
    left, right = int(xs.min()), int(xs.max()) + 1
    return left, top, right, bottom


def apply_illumination_correction(image_pil, sigma=15, alpha=1.5, beta=-0.5):
    """Estimates smooth background illumination via large-sigma Gaussian
    blur, then subtracts it out -- corrects non-uniform fundus lighting
    (bright center / dark periphery or vice versa)."""
    img = np.array(image_pil).astype(np.float32)
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)
    corrected = cv2.addWeighted(img, alpha, blur, beta, 0)
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)
    return Image.fromarray(corrected)


def apply_clahe_lab(image_pil, clip_limit=2.0, tile_grid_size=(8, 8)):
    """CLAHE on the LAB L-channel -- your original method."""
    img_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l2 = clahe.apply(l)
    lab2 = cv2.merge((l2, a, b))
    img_bgr2 = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    return Image.fromarray(cv2.cvtColor(img_bgr2, cv2.COLOR_BGR2RGB))


def apply_clahe_green(image_pil, clip_limit=2.0, tile_grid_size=(8, 8)):
    """CLAHE on the green channel specifically (often has strongest
    retinal contrast), replicated across all 3 channels so the encoder
    still receives a standard 3-channel image."""
    img_rgb = np.array(image_pil)
    green = img_rgb[:, :, 1]
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    green_enhanced = clahe.apply(green)
    stacked = np.stack([green_enhanced, green_enhanced, green_enhanced], axis=-1)
    return Image.fromarray(stacked)


def apply_unsharp_mask(image_pil, sigma=2.0, alpha=0.3):
    """Mild sharpening: original + alpha * (original - blurred).
    Emphasizes lesion boundaries and fine structures."""
    img = np.array(image_pil).astype(np.float32)
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)
    sharpened = img + alpha * (img - blur)
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    return Image.fromarray(sharpened)


def apply_preprocessing_pipeline(image_pil, config):
    """Applies the configured pipeline, IN ORDER: illumination correction
    -> CLAHE (lab or green) -> sharpening. ROI cropping is handled
    separately in the dataset (needs to also crop the mask)."""
    image = image_pil
    if config.get("illumination_correction", False):
        image = apply_illumination_correction(image)
    clahe_mode = config.get("clahe_mode", "lab")
    if clahe_mode == "lab":
        image = apply_clahe_lab(image, config.get("clahe_clip", 2.0), config.get("clahe_tile", (8, 8)))
    elif clahe_mode == "green":
        image = apply_clahe_green(image, config.get("clahe_clip", 2.0), config.get("clahe_tile", (8, 8)))
    if config.get("sharpen", False):
        image = apply_unsharp_mask(image, config.get("sharpen_sigma", 2.0), config.get("sharpen_alpha", 0.3))
    return image


# ============================================================
# PRESETS -- per faculty's ordered ablation plan (P0 through P5)
# ============================================================
PRESETS = {
    "P0": {  # baseline -- current 7G behavior, unchanged
        "roi_crop": False, "illumination_correction": False,
        "clahe_mode": "lab", "clahe_clip": 2.0, "clahe_tile": (8, 8), "sharpen": False,
    },
    "P1": {  # + ROI crop
        "roi_crop": True, "illumination_correction": False,
        "clahe_mode": "lab", "clahe_clip": 2.0, "clahe_tile": (8, 8), "sharpen": False,
    },
    "P2a": {  # ROI + CLAHE clip=1.5
        "roi_crop": True, "illumination_correction": False,
        "clahe_mode": "lab", "clahe_clip": 1.5, "clahe_tile": (8, 8), "sharpen": False,
    },
    "P2b": {  # ROI + CLAHE clip=2.5
        "roi_crop": True, "illumination_correction": False,
        "clahe_mode": "lab", "clahe_clip": 2.5, "clahe_tile": (8, 8), "sharpen": False,
    },
    "P2c": {  # ROI + CLAHE clip=3.0
        "roi_crop": True, "illumination_correction": False,
        "clahe_mode": "lab", "clahe_clip": 3.0, "clahe_tile": (8, 8), "sharpen": False,
    },
    "P3": {  # ROI + illumination correction + CLAHE -- faculty's main candidate
        "roi_crop": True, "illumination_correction": True,
        "clahe_mode": "lab", "clahe_clip": 2.0, "clahe_tile": (8, 8), "sharpen": False,
    },
    "P4": {  # ROI + green-channel CLAHE
        "roi_crop": True, "illumination_correction": False,
        "clahe_mode": "green", "clahe_clip": 2.0, "clahe_tile": (8, 8), "sharpen": False,
    },
    "P5": {  # ROI + CLAHE + mild sharpening
        "roi_crop": True, "illumination_correction": False,
        "clahe_mode": "lab", "clahe_clip": 2.0, "clahe_tile": (8, 8),
        "sharpen": True, "sharpen_sigma": 2.0, "sharpen_alpha": 0.3,
    },
}
