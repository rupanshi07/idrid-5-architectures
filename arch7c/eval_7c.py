import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score
import matplotlib.pyplot as plt

from dataset import IDRiDJointDataset
from model_7c import JointSegClsNet
from losses import ordinal_to_grade

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 512
BATCH_SIZE = 1
CHECKPOINT = "checkpoints_7c/best_model.pth"
OUTPUT_DIR = "checkpoints_7c"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# FIXED: these now point at the real, held-out TESTING set -- the
# original version reused the TRAINING image directory here, which
# would have measured segmentation on data the model already saw.
SEG_IMG_DIR = r"..\Dataset\A_Segmentation\Original_Images\b_Testing_Set"
SEG_MASK_DIR = r"..\Dataset\A_Segmentation\All_Segmentation_Groundtruths\b_Testing_Set"
CLS_IMG_DIR = r"..\Dataset\B_DiseaseGrading\Original_Images\b_Testing_Set"
CLS_CSV = r"..\Dataset\B_DiseaseGrading\Groundtruths\Testing_Labels.csv"

dataset = IDRiDJointDataset(
    seg_img_dir=SEG_IMG_DIR, seg_mask_dir=SEG_MASK_DIR,
    cls_img_dir=CLS_IMG_DIR, cls_csv=CLS_CSV,
    image_size=IMAGE_SIZE, augment=False, zoom_crop_prob=0.0, use_clahe=True
)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

print("Using device:", DEVICE)
print("Total test samples:", len(dataset))

model = JointSegClsNet(num_lesion_classes=2, num_dr_classes=4, pretrained=False).to(DEVICE)
checkpoint = torch.load(CHECKPOINT, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

all_seg_pred, all_seg_target = [], []
all_cls_pred, all_cls_target = [], []


def dice_per_channel(pred, target, eps=1e-6):
    pred = pred.astype(np.float32); target = target.astype(np.float32)
    intersection = (pred * target).sum()
    denominator = pred.sum() + target.sum()
    return (2.0 * intersection + eps) / (denominator + eps)


with torch.no_grad():
    for batch in loader:
        images = batch["image"].to(DEVICE, non_blocking=True)
        masks = batch["mask"].numpy()
        seg_valid = batch["seg_valid"].numpy()
        grades = batch["dr_grade"].numpy()
        cls_valid = batch["cls_valid"].numpy()

        seg_logits, cls_logits = model(images)
        seg_probability = torch.sigmoid(seg_logits)
        seg_prediction = (seg_probability > 0.5).float().cpu().numpy()

        for i in range(len(seg_valid)):
            if seg_valid[i] > 0:
                all_seg_pred.append(seg_prediction[i])
                all_seg_target.append(masks[i])

        cls_prediction = ordinal_to_grade(cls_logits).cpu().numpy()
        for i in range(len(cls_valid)):
            if cls_valid[i] > 0:
                all_cls_pred.append(int(cls_prediction[i]))
                all_cls_target.append(int(grades[i]))

if len(all_seg_pred) > 0:
    all_seg_pred = np.asarray(all_seg_pred)
    all_seg_target = np.asarray(all_seg_target)
    dice_scores = []
    for channel in range(2):
        scores = [dice_per_channel(all_seg_pred[i, channel], all_seg_target[i, channel]) for i in range(len(all_seg_pred))]
        dice_scores.append(np.mean(scores))
    print("\nPer-lesion Dice:")
    print(f"  EX: {dice_scores[0]:.4f}")
    print(f"  SE: {dice_scores[1]:.4f}")
    print(f"Mean Dice: {np.mean(dice_scores):.4f}")

if len(all_cls_target) > 0:
    y_true = np.asarray(all_cls_target)
    y_pred = np.asarray(all_cls_pred)
    accuracy = accuracy_score(y_true, y_pred)
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Quadratic Kappa: {qwk:.4f}")
    print(f"(n={len(y_true)})")
    print("\nConfusion matrix:")
    print(cm)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm)
    ax.set_xlabel("Predicted Grade"); ax.set_ylabel("True Grade")
    ax.set_title("7C DR Classification Confusion Matrix")
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels([0, 1, 2, 3, 4]); ax.set_yticklabels([0, 1, 2, 3, 4])
    for i in range(5):
        for j in range(5):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print("\nConfusion matrix saved to:", cm_path)
