import os
import glob
import csv
import random
import numpy as np
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

LESIONS = ["EX", "SE"]

LESION_FOLDER_MAP = {
    "EX": "Hard_Exudates",
    "SE": "Soft_Exudates",
}


def apply_clahe(image_pil, clip_limit=2.0, tile_grid_size=(8, 8)):
    """CLAHE (Contrast Limited Adaptive Histogram Equalization).
    Applied on the L (lightness) channel of LAB colour space, which
    boosts local contrast without distorting colour -- standard
    preprocessing for retinal fundus images, since lesions like
    exudates are often low-contrast against surrounding tissue.
    This is a deterministic PREPROCESSING step, applied to every
    image (train, val, test) -- not a random augmentation."""
    img_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l2 = clahe.apply(l)
    lab2 = cv2.merge((l2, a, b))
    img_bgr2 = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    img_rgb2 = cv2.cvtColor(img_bgr2, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb2)


class IDRiDJointDataset(Dataset):
    def __init__(self, seg_img_dir=None, seg_mask_dir=None,
                 cls_img_dir=None, cls_csv=None,
                 image_size=512, augment=False, zoom_crop_prob=0.0,
                 use_clahe=True):
        self.image_size = image_size
        self.augment = augment
        self.zoom_crop_prob = zoom_crop_prob
        self.use_clahe = use_clahe
        self.samples = []

        self.resize_norm = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.color_jitter = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02)

        if seg_img_dir and seg_mask_dir:
            self._add_segmentation_samples(seg_img_dir, seg_mask_dir)
        if cls_img_dir and cls_csv:
            self._add_classification_samples(cls_img_dir, cls_csv)
        if not self.samples:
            raise ValueError("No samples found. Check your directory paths.")

    def _add_segmentation_samples(self, img_dir, mask_dir):
        img_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")) +
                            glob.glob(os.path.join(img_dir, "*.png")))
        for img_path in img_paths:
            name = os.path.splitext(os.path.basename(img_path))[0]
            mask_paths = {}
            for les in LESIONS:
                folder = os.path.join(mask_dir, LESION_FOLDER_MAP[les])
                matches = glob.glob(os.path.join(folder, f"{name}_{les}.*"))
                mask_paths[les] = matches[0] if matches else None
            self.samples.append({
                "img_path": img_path, "has_seg": True, "mask_paths": mask_paths,
                "has_cls": False, "dr_grade": -1,
            })

    def _add_classification_samples(self, img_dir, csv_path):
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row.get("Image name") or row.get("Image Name") or list(row.values())[0]
                name = name.strip()
                grade_key = next((k for k in row.keys() if "retinopathy" in k.lower() and "grade" in k.lower()), None)
                if grade_key is None:
                    continue
                grade_str = row[grade_key].strip()
                if grade_str == "":
                    continue
                dr_grade = int(float(grade_str))
                candidates = glob.glob(os.path.join(img_dir, f"{name}.*"))
                if not candidates:
                    continue
                self.samples.append({
                    "img_path": candidates[0], "has_seg": False, "mask_paths": None,
                    "has_cls": True, "dr_grade": dr_grade,
                })

    def __len__(self):
        return len(self.samples)

    def class_counts(self, num_classes=5):
        counts = [0] * num_classes
        for s in self.samples:
            if s["has_cls"]:
                counts[s["dr_grade"]] += 1
        return counts

    def split_train_val(self, val_fraction=0.15, seed=42):
        seg_idx = [i for i, s in enumerate(self.samples) if s["has_seg"]]
        cls_idx = [i for i, s in enumerate(self.samples) if s["has_cls"]]
        rnd = random.Random(seed)
        rnd.shuffle(seg_idx)
        rnd.shuffle(cls_idx)

        def split(idx_list):
            n_val = max(1, int(len(idx_list) * val_fraction))
            return idx_list[n_val:], idx_list[:n_val]

        seg_train, seg_val = split(seg_idx)
        cls_train, cls_val = split(cls_idx)
        return (IDRiDSubset(self, seg_train + cls_train, augment=True, zoom_crop_prob=self.zoom_crop_prob),
                IDRiDSubset(self, seg_val + cls_val, augment=False, zoom_crop_prob=0.0))

    def _augment_pair(self, image, raw_masks):
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            raw_masks = {k: (v.transpose(Image.FLIP_LEFT_RIGHT) if v is not None else None)
                         for k, v in raw_masks.items()}
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            raw_masks = {k: (v.transpose(Image.FLIP_TOP_BOTTOM) if v is not None else None)
                         for k, v in raw_masks.items()}
        angle = random.uniform(-15, 15)
        image = image.rotate(angle, resample=Image.BILINEAR)
        raw_masks = {k: (v.rotate(angle, resample=Image.NEAREST) if v is not None else None)
                     for k, v in raw_masks.items()}
        return image, raw_masks

    def _get_sample(self, idx, augment, zoom_crop_prob=0.0):
        sample = self.samples[idx]
        image = Image.open(sample["img_path"]).convert("RGB")

        # CLAHE preprocessing -- applied to EVERY image (train, val, test),
        # deterministic, not randomized like augmentation.
        if self.use_clahe:
            image = apply_clahe(image)

        if sample["has_seg"]:
            raw_masks = {les: (Image.open(sample["mask_paths"][les]).convert("L")
                                if sample["mask_paths"][les] else None) for les in LESIONS}
        else:
            raw_masks = {les: None for les in LESIONS}

        if augment:
            image, raw_masks = self._augment_pair(image, raw_masks)
            image = self.color_jitter(image)

        image_tensor = self.resize_norm(image)

        mask = np.zeros((len(LESIONS), self.image_size, self.image_size), dtype=np.float32)
        if sample["has_seg"]:
            for i, les in enumerate(LESIONS):
                m = raw_masks[les]
                if m is not None:
                    m = m.resize((self.image_size, self.image_size), Image.NEAREST)
                    mask[i] = (np.array(m) > 0).astype(np.float32)
        mask = torch.from_numpy(mask)

        return {
            "image": image_tensor,
            "mask": mask,
            "seg_valid": torch.tensor(1.0 if sample["has_seg"] else 0.0),
            "dr_grade": torch.tensor(sample["dr_grade"], dtype=torch.long),
            "cls_valid": torch.tensor(1.0 if sample["has_cls"] else 0.0),
        }

    def __getitem__(self, idx):
        return self._get_sample(idx, augment=self.augment, zoom_crop_prob=self.zoom_crop_prob)


class IDRiDSubset(Dataset):
    def __init__(self, base_dataset, indices, augment, zoom_crop_prob=0.0):
        self.base = base_dataset
        self.indices = indices
        self.augment = augment
        self.zoom_crop_prob = zoom_crop_prob

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.base._get_sample(self.indices[idx], augment=self.augment, zoom_crop_prob=self.zoom_crop_prob)

    def sample_weights(self):
        has_seg_list = [self.base.samples[i]["has_seg"] for i in self.indices]
        n_seg = sum(has_seg_list)
        n_cls = len(has_seg_list) - n_seg
        weights = []
        for hs in has_seg_list:
            weights.append(0.5 / max(n_seg, 1) if hs else 0.5 / max(n_cls, 1))
        return weights

    def compute_pos_weight(self, num_lesions, image_size, max_weight=50.0):
        total_pos = torch.zeros(num_lesions)
        total_pixels = 0
        for i in self.indices:
            sample = self.base.samples[i]
            if not sample["has_seg"]:
                continue
            for c, les in enumerate(LESIONS):
                path = sample["mask_paths"][les]
                if path:
                    m = Image.open(path).convert("L").resize((image_size, image_size), Image.NEAREST)
                    arr = (np.array(m) > 0).astype(np.float32)
                    total_pos[c] += arr.sum()
            total_pixels += image_size * image_size
        total_neg = total_pixels - total_pos
        pos_weight = (total_neg / (total_pos + 1e-6)).clamp(max=max_weight)
        return pos_weight
