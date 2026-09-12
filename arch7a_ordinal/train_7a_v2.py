import argparse
import os
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import cohen_kappa_score
from tqdm import tqdm

from dataset import IDRiDJointDataset, LESIONS
from model_7a_ordinal_ex_guided import JointSegClsNet
from losses import JointLoss, ordinal_to_grade


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
    p.add_argument("--seg_img_dir", required=True)
    p.add_argument("--seg_mask_dir", required=True)
    p.add_argument("--cls_img_dir", required=True)
    p.add_argument("--cls_csv", required=True)
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--val_fraction", type=float, default=0.15)
    p.add_argument("--seg_weight", type=float, default=2.5)
    p.add_argument("--cls_weight", type=float, default=1.5)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--out_dir", default="checkpoints_7a_ordinal_v2")
    return p.parse_args()


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    inter_total = torch.zeros(len(LESIONS))
    union_total = torch.zeros(len(LESIONS))
    all_preds, all_targets = [], []
    for batch in loader:
        image = batch["image"].to(device)
        mask = batch["mask"].to(device)
        seg_valid = batch["seg_valid"]
        dr_grade = batch["dr_grade"]
        cls_valid = batch["cls_valid"]
        with torch.autocast(device_type="cuda", enabled=(device.type == "cuda")):
            seg_pred, cls_pred = model(image)
        for i in range(image.size(0)):
            if seg_valid[i] == 1:
                inter, uni = dice_per_lesion(seg_pred[i:i+1].float(), mask[i:i+1])
                inter_total += inter.cpu(); union_total += uni.cpu()
            if cls_valid[i] == 1:
                grade_pred = ordinal_to_grade(cls_pred[i:i+1].float()).item()
                all_preds.append(grade_pred)
                all_targets.append(dr_grade[i].item())
    mean_dice = finalize_dice(inter_total, union_total).mean().item() if union_total.sum() > 0 else 0.0
    kappa = cohen_kappa_score(all_targets, all_preds, weights="quadratic") if len(all_preds) > 1 else 0.0
    return mean_dice, kappa


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Arch 7A Ordinal v2] Using device: {device}")

    full_dataset = IDRiDJointDataset(
        seg_img_dir=args.seg_img_dir, seg_mask_dir=args.seg_mask_dir,
        cls_img_dir=args.cls_img_dir, cls_csv=args.cls_csv,
        image_size=args.image_size, augment=False, zoom_crop_prob=0.35, use_clahe=True,
    )
    train_set, val_set = full_dataset.split_train_val(val_fraction=args.val_fraction)
    print(f"Train samples: {len(train_set)} | Val samples: {len(val_set)}")

    pos_weight = train_set.compute_pos_weight(len(LESIONS), args.image_size)
    seg_focal_alpha = (pos_weight / (1.0 + pos_weight)).clamp(0.5, 0.98).to(device)

    sample_weights = train_set.sample_weights()
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_set), replacement=True)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=sampler,
                               num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    model = JointSegClsNet(num_lesion_classes=len(LESIONS), num_dr_classes=4, pretrained=args.pretrained).to(device)
    criterion = JointLoss(seg_weight=args.seg_weight, cls_weight=args.cls_weight, seg_focal_alpha=seg_focal_alpha)

    # Differential learning rates -- same fix used in Architecture 7C.
    # A single flat LR (previously 5e-5 for everything) either destabilized
    # the pretrained Swin encoder or undertrained the classifier head --
    # this was the direct cause of the classification collapse.
    encoder_params, other_params, classifier_params = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("encoder."):
            encoder_params.append(param)
        elif name.startswith("cls_head") or name.startswith("cls_transformer") or "classifier" in name:
            classifier_params.append(param)
        else:
            other_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": 1e-5},
        {"params": other_params, "lr": 1e-4},
        {"params": classifier_params, "lr": 2e-4},
    ], weight_decay=args.weight_decay)

    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    os.makedirs(args.out_dir, exist_ok=True)
    best_score = -1.0

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"[7A-Ordinal-v2] Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            image = batch["image"].to(device); mask = batch["mask"].to(device)
            seg_valid = batch["seg_valid"].to(device); dr_grade = batch["dr_grade"].to(device)
            cls_valid = batch["cls_valid"].to(device)
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", enabled=(device.type == "cuda")):
                seg_pred, cls_pred = model(image)
                loss, seg_l, cls_l = criterion(seg_pred, mask, seg_valid, cls_pred, dr_grade, cls_valid)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer); scaler.update()
            running += loss.item()
            pbar.set_postfix(loss=loss.item(), seg=seg_l.item(), cls=cls_l.item())

        scheduler.step()
        val_dice, val_kappa = validate(model, val_loader, device)
        combined = 0.5 * val_dice + 0.5 * max(val_kappa, 0.0)
        print(f"Epoch {epoch+1}: val Dice={val_dice:.4f} | val Kappa={val_kappa:.4f} | combined={combined:.4f} | LR={scheduler.get_last_lr()}")

        torch.save(model.state_dict(), os.path.join(args.out_dir, "model_last.pth"))
        if combined > best_score:
            best_score = combined
            torch.save(model.state_dict(), os.path.join(args.out_dir, "model_best.pth"))
            print(f"  -> New best saved (combined={combined:.4f})")

    print(f"\n[Arch 7A Ordinal v2] Training complete. Best combined val score: {best_score:.4f}")


if __name__ == "__main__":
    main()
