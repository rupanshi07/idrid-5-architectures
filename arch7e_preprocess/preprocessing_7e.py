"""
Configurable preprocessing pipeline: P0 through P7. Each function
operates on a PIL image; ROI cropping returns a bounding box that MUST
also be applied to the segmentation mask, to keep image/mask aligned.
"""
import numpy as np
import cv2
from PIL import Image


def get_roi_bbox(image_pil, threshold=7):
    gray = np.array(image_pil.convert("L"))
    mask = gray > threshold
    if mask.sum() == 0:
        return 0, 0, image_pil.width, image_pil.height
    ys, xs = np.where(mask)
    top, bottom = int(ys.min()), int(ys.max()) + 1
    left, right = int(xs.min()), int(xs.max()) + 1
    return left, top, right, bottom


def apply_illumination_correction(image_pil, sigma=15, alpha=1.5, beta=-0.5):
    img = np.array(image_pil).astype(np.float32)
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)
    corrected = cv2.addWeighted(img, alpha, blur, beta, 0)
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)
    return Image.fromarray(corrected)


def apply_clahe_lab(image_pil, clip_limit=2.0, tile_grid_size=(8, 8)):
    img_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l2 = clahe.apply(l)
    lab2 = cv2.merge((l2, a, b))
    img_bgr2 = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    return Image.fromarray(cv2.cvtColor(img_bgr2, cv2.COLOR_BGR2RGB))


def apply_clahe_green_destructive(image_pil, clip_limit=2.0, tile_grid_size=(8, 8)):
    """The P4 approach -- replicates enhanced green across all 3
    channels, discarding R/B entirely. Kept for reference/comparison."""
    img_rgb = np.array(image_pil)
    green = img_rgb[:, :, 1]
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    green_enhanced = clahe.apply(green)
    stacked = np.stack([green_enhanced, green_enhanced, green_enhanced], axis=-1)
    return Image.fromarray(stacked)


def apply_green_enhancement_preserve_rgb(image_pil, clip_limit=2.0, tile_grid_size=(8, 8)):
    """NON-DESTRUCTIVE green enhancement, per faculty's point #4: keeps
    R and B channels UNCHANGED, only replaces G with its CLAHE-enhanced
    version. Preserves the RGB color distribution Swin's ImageNet
    pretraining expects, while still emphasizing the channel with
    strongest retinal structural contrast."""
    img_rgb = np.array(image_pil)
    r, g, b = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    g_enhanced = clahe.apply(g)
    combined = np.stack([r, g_enhanced, b], axis=-1)
    return Image.fromarray(combined)


def apply_unsharp_mask(image_pil, sigma=2.0, alpha=0.3):
    img = np.array(image_pil).astype(np.float32)
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)
    sharpened = np.clip(img + alpha * (img - blur), 0, 255).astype(np.uint8)
    return Image.fromarray(sharpened)


def apply_preprocessing_pipeline(image_pil, config):
    """Order: illumination correction -> green enhancement (preserve-RGB
    or destructive) -> LAB CLAHE -> sharpening. ROI is handled in dataset.py
    since it must also crop the mask."""
    image = image_pil
    if config.get("illumination_correction", False):
        image = apply_illumination_correction(image)

    green_mode = config.get("green_mode", None)
    if green_mode == "preserve_rgb":
        image = apply_green_enhancement_preserve_rgb(image, config.get("clahe_clip", 2.0), config.get("clahe_tile", (8, 8)))
    elif green_mode == "destructive":
        image = apply_clahe_green_destructive(image, config.get("clahe_clip", 2.0), config.get("clahe_tile", (8, 8)))

    if config.get("apply_lab_clahe", True):
        image = apply_clahe_lab(image, config.get("clahe_clip", 2.0), config.get("clahe_tile", (8, 8)))

    if config.get("sharpen", False):
        image = apply_unsharp_mask(image, config.get("sharpen_sigma", 2.0), config.get("sharpen_alpha", 0.3))
    return image


# ============================================================
# PRESETS
# ============================================================
PRESETS = {
    "P0": {"roi_crop": False, "illumination_correction": False, "green_mode": None,
           "apply_lab_clahe": True, "clahe_clip": 2.0, "clahe_tile": (8, 8), "sharpen": False},
    "P1": {"roi_crop": True, "illumination_correction": False, "green_mode": None,
           "apply_lab_clahe": True, "clahe_clip": 2.0, "clahe_tile": (8, 8), "sharpen": False},
    "P3": {"roi_crop": True, "illumination_correction": True, "green_mode": None,
           "apply_lab_clahe": True, "clahe_clip": 2.0, "clahe_tile": (8, 8), "sharpen": False},
    "P4": {"roi_crop": True, "illumination_correction": False, "green_mode": "destructive",
           "apply_lab_clahe": False, "clahe_clip": 2.0, "clahe_tile": (8, 8), "sharpen": False},
    "P5": {"roi_crop": True, "illumination_correction": False, "green_mode": None,
           "apply_lab_clahe": True, "clahe_clip": 2.0, "clahe_tile": (8, 8),
           "sharpen": True, "sharpen_sigma": 2.0, "sharpen_alpha": 0.3},

    # NEW -- per faculty's latest feedback
    "P6": {  # illumination correction + NON-destructive green enhancement, no ROI yet
        "roi_crop": False, "illumination_correction": True, "green_mode": "preserve_rgb",
        "apply_lab_clahe": True, "clahe_clip": 2.0, "clahe_tile": (8, 8), "sharpen": False,
    },
    "P7": {  # P6 + ROI crop -- the full combined candidate pipeline
        "roi_crop": True, "illumination_correction": True, "green_mode": "preserve_rgb",
        "apply_lab_clahe": True, "clahe_clip": 2.0, "clahe_tile": (8, 8), "sharpen": False,
    },
}
