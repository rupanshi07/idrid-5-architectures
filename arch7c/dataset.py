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
LESION_FOLDER_MAP = {"EX": "Hard_Exudates", "SE": "Soft_Exudates"}


def apply_clahe(image_pil, clip_limit=2.0, tile_grid_size=(8, 8)):
    img_rgb = np.array(image_pil)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l2 = clahe.apply(l)
    lab2 = cv2.merge((l2, a, b))
    img_bgr2 = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    img_rgb2 = cv2.cvtColor(img_bgr2, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb2)


class IDRiDJointDataset(Dataset):
    def __init__(self, seg_img_dir=None, seg_mask_dir=None, cls_img_dir=None, cls_csv=None,
                 image_size=512, augment=False, zoom_crop_prob=0.25, use_clahe=True):
        self.image_size = image_size
        self.augment = augment
        self.zoom_crop_prob = zoom_crop_prob
        self.use_clahe = use_clahe
        self.samples = []

        self.resize_norm = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.color_jitter = T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02)

        if seg_img_dir and seg_mask_dir:
            self._add_segmentation_samples(seg_img_dir, seg_mask_dir)
        if cls_img_dir and cls_csv:
            self._add_classification_samples(cls_img_dir, cls_csv)
        if not self.samples:
            raise ValueError("No samples found. Check your dataset paths.")

    def _add_segmentation_samples(self, img_dir, mask_dir):
        img_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")) + glob.glob(os.path.join(img_dir, "*.png")))
        for img_path in img_paths:
            name = os.path.splitext(os.path.basename(img_path))[0]
            mask_paths = {}
            for lesion in LESIONS:
                folder = os.path.join(mask_dir, LESION_FOLDER_MAP[lesion])
                matches = glob.glob(os.path.join(folder, f"{name}_{lesion}.*"))
                mask_paths[lesion] = matches[0] if matches else None
            self.samples.append({"img_path": img_path, "has_seg": True, "mask_paths": mask_paths,
                                  "has_cls": False, "dr_grade": -1})

    def _add_classification_samples(self, img_dir, csv_path):
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row.get("Image name") or row.get("Image Name") or list(row.values())[0]
                name = name.strip()
                grade_key = None
                for key in row.keys():
                    if "retinopathy" in key.lower() and "grade" in key.lower():
                        grade_key = key
                        break
                if grade_key is None:
                    continue
                grade_str = row[grade_key].strip()
                if grade_str == "":
                    continue
                dr_grade = int(float(grade_str))
                candidates = glob.glob(os.path.join(img_dir, f"{name}.*"))
                if not candidates:
                    continue
                self.samples.append({"img_path": candidates[0], "has_seg": False, "mask_paths": None,
                                      "has_cls": True, "dr_grade": dr_grade})

    def __len__(self):
        return len(self.samples)

    def class_counts(self, num_classes=5):
        counts = [0] * num_classes
        for sample in self.samples:
            if sample["has_cls"]:
                grade = sample["dr_grade"]
                if 0 <= grade < num_classes:
                    counts[grade] += 1
        return counts

    def _augment_pair(self, image, raw_masks):
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            raw_masks = {k: (v.transpose(Image.FLIP_LEFT_RIGHT) if v is not None else None) for k, v in raw_masks.items()}
        if random.random() < 0.5:
            angle = random.uniform(-10, 10)
            image = image.rotate(angle, resample=Image.BILINEAR)
            raw_masks = {k: (v.rotate(angle, resample=Image.NEAREST) if v is not None else None) for k, v in raw_masks.items()}
        return image, raw_masks

    def _zoom_crop_pair(self, image, raw_masks):
        width, height = image.size
        scale = random.uniform(0.70, 0.90)
        crop_w, crop_h = int(width * scale), int(height * scale)
        if crop_w >= width or crop_h >= height:
            return image, raw_masks
        left = random.randint(0, width - crop_w)
        top = random.randint(0, height - crop_h)
        right, bottom = left + crop_w, top + crop_h
        image = image.crop((left, top, right, bottom))
        raw_masks = {k: (v.crop((left, top, right, bottom)) if v is not None else None) for k, v in raw_masks.items()}
        return image, raw_masks

    def _get_sample(self, idx, augment, zoom_crop_prob=0.0):
        sample = self.samples[idx]
        image = Image.open(sample["img_path"]).convert("RGB")
        if self.use_clahe:
            image = apply_clahe(image)

        if sample["has_seg"]:
            raw_masks = {lesion: (Image.open(sample["mask_paths"][lesion]).convert("L") if sample["mask_paths"][lesion] else None) for lesion in LESIONS}
        else:
            raw_masks = {lesion: None for lesion in LESIONS}

        if augment:
            image, raw_masks = self._augment_pair(image, raw_masks)
            if random.random() < zoom_crop_prob:
                image, raw_masks = self._zoom_crop_pair(image, raw_masks)
            image = self.color_jitter(image)

        image_tensor = self.resize_norm(image)

        mask = np.zeros((len(LESIONS), self.image_size, self.image_size), dtype=np.float32)
        if sample["has_seg"]:
            for i, lesion in enumerate(LESIONS):
                m = raw_masks[lesion]
                if m is not None:
                    m = m.resize((self.image_size, self.image_size), Image.NEAREST)
                    mask[i] = (np.array(m) > 0).astype(np.float32)
        mask = torch.from_numpy(mask)

        return {
            "image": image_tensor, "mask": mask,
            "seg_valid": torch.tensor(1.0 if sample["has_seg"] else 0.0, dtype=torch.float32),
            "dr_grade": torch.tensor(sample["dr_grade"], dtype=torch.long),
            "cls_valid": torch.tensor(1.0 if sample["has_cls"] else 0.0, dtype=torch.float32)
        }

    def __getitem__(self, idx):
        return self._get_sample(idx, augment=self.augment, zoom_crop_prob=self.zoom_crop_prob)

    def split_train_val(self, val_fraction=0.15, seed=42):
        seg_idx = [i for i, s in enumerate(self.samples) if s["has_seg"]]
        cls_by_grade = {grade: [] for grade in range(5)}
        for i, sample in enumerate(self.samples):
            if sample["has_cls"]:
                grade = sample["dr_grade"]
                if grade in cls_by_grade:
                    cls_by_grade[grade].append(i)

        rnd = random.Random(seed)
        rnd.shuffle(seg_idx)
        n_seg_val = max(1, int(len(seg_idx) * val_fraction))
        seg_val = seg_idx[:n_seg_val]
        seg_train = seg_idx[n_seg_val:]

        cls_train, cls_val = [], []
        for grade in range(5):
            indices = cls_by_grade[grade]
            rnd.shuffle(indices)
            if len(indices) == 0:
                continue
            n_val = max(1, int(len(indices) * val_fraction))
            cls_val.extend(indices[:n_val])
            cls_train.extend(indices[n_val:])

        rnd.shuffle(seg_train); rnd.shuffle(seg_val)
        rnd.shuffle(cls_train); rnd.shuffle(cls_val)
        train_indices = seg_train + cls_train
        val_indices = seg_val + cls_val
        rnd.shuffle(train_indices); rnd.shuffle(val_indices)

        return (IDRiDSubset(self, train_indices, augment=True, zoom_crop_prob=self.zoom_crop_prob),
                IDRiDSubset(self, val_indices, augment=False, zoom_crop_prob=0.0))


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
        seg_indices = [i for i in self.indices if self.base.samples[i]["has_seg"]]
        cls_indices = [i for i in self.indices if self.base.samples[i]["has_cls"]]
        n_seg, n_cls = len(seg_indices), len(cls_indices)
        weights = []
        for i in self.indices:
            weights.append(0.5 / max(n_seg, 1) if self.base.samples[i]["has_seg"] else 0.5 / max(n_cls, 1))
        return weights

    def compute_pos_weight(self, num_lesions, image_size, max_weight=50.0):
        total_pos = torch.zeros(num_lesions)
        total_pixels = 0
        for i in self.indices:
            sample = self.base.samples[i]
            if not sample["has_seg"]:
                continue
            for c, lesion in enumerate(LESIONS):
                path = sample["mask_paths"][lesion]
                if path:
                    m = Image.open(path).convert("L").resize((image_size, image_size), Image.NEAREST)
                    arr = (np.array(m) > 0).astype(np.float32)
                    total_pos[c] += arr.sum()
            total_pixels += image_size * image_size
        total_neg = total_pixels - total_pos
        return (total_neg / (total_pos + 1e-6)).clamp(max=max_weight)
