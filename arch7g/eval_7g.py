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


def plot_confusion_matrix(cm, class_names, out_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names))); ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names); ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted grade"); ax.set_ylabel("True grade")
    ax.set_title("DR Grading -- Arch 7G (2D pos-embed, separate EX/SE, attn pooling)")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, ax=ax); fig.tight_layout()
    fig.savefig(out_path, dpi=200); plt.close(fig)
    print(f"Confusion matrix saved to {out_path}")


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seg_img_dir", required=True)
    p.add_argument("--seg_mask_dir", required=True)
    p.add_argument("--cls_img_dir", required=True)
    p.add_argument("--cls_csv", required=True)
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--checkpoint", default="checkpoints_7g/model_best.pth")
    p.add_argument("--cm_out", default="checkpoints_7g/confusion_matrix.png")
    return p.parse_args()


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[7G] Using device: {device}")

    dataset = IDRiDJointDataset(
        seg_img_dir=args.seg_img_dir, seg_mask_dir=args.seg_mask_dir,
        cls_img_dir=args.cls_img_dir, cls_csv=args.cls_csv,
        image_size=args.image_size, augment=False, use_clahe=True,
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
        cm = confusion_matrix(all_targets, all_preds, labels=list(range(5)))
        print("\nConfusion matrix:")
        print(cm)
        plot_confusion_matrix(cm, [str(i) for i in range(5)], args.cm_out)


if __name__ == "__main__":
    main()
