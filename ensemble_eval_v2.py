import sys
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
import matplotlib.pyplot as plt

sys.path.insert(0, "arch7g")

from dataset import IDRiDJointDataset, LESIONS
from model_5_gan_segmentation import Generator
from model_7g import JointSegClsNet as Model7G


def plot_confusion_matrix(cm, class_names, out_path, title):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names))); ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names); ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted grade"); ax.set_ylabel("True grade")
    ax.set_title(title)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, ax=ax); fig.tight_layout()
    fig.savefig(out_path, dpi=200); plt.close(fig)
    print(f"Confusion matrix saved to {out_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Ensemble: GAN + 7G] Using device: {device}")

    dataset = IDRiDJointDataset(
        seg_img_dir=r".\Dataset\A_Segmentation\Original_Images\b_Testing_Set",
        seg_mask_dir=r".\Dataset\A_Segmentation\All_Segmentation_Groundtruths\b_Testing_Set",
        cls_img_dir=r".\Dataset\B_DiseaseGrading\Original_Images\b_Testing_Set",
        cls_csv=r".\Dataset\B_DiseaseGrading\Groundtruths\Testing_Labels.csv",
        image_size=512, augment=False, use_clahe=True,
    )
    print(f"Total test samples: {len(dataset)}")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    model_gan = Generator(num_lesion_classes=len(LESIONS), pretrained=False).to(device)
    model_gan.load_state_dict(torch.load("Results_5_GAN/model_best.pth", map_location=device))
    model_gan.eval()

    model_7g = Model7G(num_lesion_classes=len(LESIONS), num_dr_classes=5, pretrained=False).to(device)
    model_7g.load_state_dict(torch.load("arch7g/Results_7G/model_best.pth", map_location=device))
    model_7g.eval()

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

            seg1, cls1 = model_gan(image)
            seg2, cls2, _ = model_7g(image)

            seg_prob = (torch.sigmoid(seg1) + torch.sigmoid(seg2)) / 2.0
            pred_bin = (seg_prob > 0.5).float()

            cls_prob = (torch.softmax(cls1, dim=1) + torch.softmax(cls2, dim=1)) / 2.0

            for i in range(image.size(0)):
                if seg_valid[i] == 1:
                    inter = (pred_bin[i] * mask[i]).sum(dim=(1, 2))
                    uni = pred_bin[i].sum(dim=(1, 2)) + mask[i].sum(dim=(1, 2))
                    inter_total += inter.cpu(); union_total += uni.cpu()
                if cls_valid[i] == 1:
                    grade_pred = cls_prob[i].argmax().item()
                    all_preds.append(grade_pred)
                    all_targets.append(dr_grade[i].item())

    if union_total.sum() > 0:
        dice_per_les = (2 * inter_total + 1e-6) / (union_total + 1e-6)
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
        plot_confusion_matrix(cm, [str(i) for i in range(5)], "ensemble_v2_confusion_matrix.png",
                               "DR Grading -- Ensemble (GAN + 7G)")


if __name__ == "__main__":
    main()
