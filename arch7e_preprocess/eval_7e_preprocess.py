import argparse
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
import matplotlib.pyplot as plt

from dataset import IDRiDJointDataset, LESIONS
from model_7g import JointSegClsNet


def dice_per_lesion(pred, target, eps=1e-6):
    pred_bin = (torch.sigmoid(pred) > 0.5).float()
    dims = (0, 2, 3)
    intersection = (pred_bin * target).sum(dims)
    union = pred_bin.sum(dims) + target.sum(dims)
    return intersection, union


def finalize_dice(intersection, union, eps=1e-6):
    return (2 * intersection + eps) / (union + eps)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--preset", required=True)
    p.add_argument("--seg_img_dir", required=True)
    p.add_argument("--seg_mask_dir", required=True)
    p.add_argument("--cls_img_dir", required=True)
    p.add_argument("--cls_csv", required=True)
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--checkpoint", required=True)
    return p.parse_args()


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{args.preset}] Using device: {device}")

    dataset = IDRiDJointDataset(
        seg_img_dir=args.seg_img_dir, seg_mask_dir=args.seg_mask_dir,
        cls_img_dir=args.cls_img_dir, cls_csv=args.cls_csv,
        image_size=args.image_size, augment=False, preprocess_preset=args.preset,
    )
    print(f"Total test samples: {len(dataset)}")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    model = JointSegClsNet(num_lesion_classes=len(LESIONS), num_dr_classes=5, pretrained=False).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    inter_total = torch.zeros(len(LESIONS))
    union_total = torch.zeros(len(LESIONS))
    all_preds, all_targets = [], []

    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device)
            mask = batch["mask"].to(device)
            seg_valid = batch["seg_valid"]
            dr_grade = batch["dr_grade"]
            cls_valid = batch["cls_valid"]
            seg_pred, cls_pred, _ = model(image)
            for i in range(image.size(0)):
                if seg_valid[i] == 1:
                    inter, uni = dice_per_lesion(seg_pred[i:i+1], mask[i:i+1])
                    inter_total += inter.cpu(); union_total += uni.cpu()
                if cls_valid[i] == 1:
                    all_preds.append(cls_pred[i].argmax().item())
                    all_targets.append(dr_grade[i].item())

    if union_total.sum() > 0:
        dice_per_les = finalize_dice(inter_total, union_total)
        print("\nPer-lesion Dice:")
        for les, d in zip(LESIONS, dice_per_les.tolist()):
            print(f"  {les}: {d:.4f}")
        print(f"Mean Dice: {dice_per_les.mean().item():.4f}")

    if all_preds:
        acc = accuracy_score(all_targets, all_preds)
        kappa = cohen_kappa_score(all_targets, all_preds, weights="quadratic")
        print(f"\nAccuracy: {acc:.4f}")
        print(f"Quadratic Kappa: {kappa:.4f}  (n={len(all_preds)})")


if __name__ == "__main__":
    main()
