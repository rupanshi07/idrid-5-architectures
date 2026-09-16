import argparse
import os
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import cohen_kappa_score, accuracy_score
from tqdm import tqdm

from dataset import IDRiDJointDataset, LESIONS
from model_7g import JointSegClsNet
from losses import dice_loss, sigmoid_focal_loss, multiclass_focal_loss


def dice_per_lesion(pred, target, eps=1e-6):
    pred_bin = (torch.sigmoid(pred) > 0.5).float()
    dims = (0, 2, 3)
    intersection = (pred_bin * target).sum(dims)
    union = pred_bin.sum(dims) + target.sum(dims)
    return intersection, union


def finalize_dice(intersection, union, eps=1e-6):
    return (2 * intersection + eps) / (union + eps)


def compute_loss(seg_out, aux_outs, mask, seg_valid, cls_out, dr_grade, cls_valid,
                  seg_focal_alpha, seg_weight, cls_weight, aux_weights=(0.4, 0.2)):
    if seg_valid.sum() > 0:
        valid_idx = seg_valid.bool()
        dsc = dice_loss(seg_out[valid_idx], mask[valid_idx])
        focal = sigmoid_focal_loss(seg_out, mask, seg_focal_alpha, gamma=2.0)
        focal = (focal * seg_valid).sum() / seg_valid.sum()
        seg_loss = dsc + focal
        for aux_out, w in zip(aux_outs, aux_weights):
            aux_dsc = dice_loss(aux_out[valid_idx], mask[valid_idx])
            aux_focal = sigmoid_focal_loss(aux_out, mask, seg_focal_alpha, gamma=2.0)
            aux_focal = (aux_focal * seg_valid).sum() / seg_valid.sum()
            seg_loss = seg_loss + w * (aux_dsc + aux_focal)
    else:
        seg_loss = torch.tensor(0.0, device=seg_out.device)

    if cls_valid.sum() > 0:
        valid_idx = cls_valid.bool()
        safe_target = dr_grade.clone()
        safe_target[safe_target < 0] = 0
        cls_focal = multiclass_focal_loss(cls_out, safe_target, gamma=2.0, weight=None)
        cls_loss = (cls_focal * cls_valid).sum() / cls_valid.sum()
    else:
        cls_loss = torch.tensor(0.0, device=cls_out.device)

    total = seg_weight * seg_loss + cls_weight * cls_loss
    return total, seg_loss.detach(), cls_loss.detach()


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seg_img_dir", required=True)
    p.add_argument("--seg_mask_dir", required=True)
    p.add_argument("--cls_img_dir", required=True)
    p.add_argument("--cls_csv", required=True)
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--warmup_epochs", type=int, default=3)
    p.add_argument("--freeze_epochs", type=int, default=5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--val_fraction", type=float, default=0.15)
    p.add_argument("--seg_weight", type=float, default=2.0)
    p.add_argument("--cls_weight", type=float, default=1.25)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--out_dir", default="checkpoints_7g")
    return p.parse_args()


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    inter_total = torch.zeros(len(LESIONS))
    union_total = torch.zeros(len(LESIONS))
    all_preds, all_targets = [], []
    for batch in loader:
        image = batch["image"].to(device); mask = batch["mask"].to(device)
        seg_valid = batch["seg_valid"]; dr_grade = batch["dr_grade"]; cls_valid = batch["cls_valid"]
        with torch.autocast(device_type="cuda", enabled=(device.type == "cuda")):
            seg_pred, cls_pred, _ = model(image)
        for i in range(image.size(0)):
            if seg_valid[i] == 1:
                inter, uni = dice_per_lesion(seg_pred[i:i+1].float(), mask[i:i+1])
                inter_total += inter.cpu(); union_total += uni.cpu()
            if cls_valid[i] == 1:
                all_preds.append(cls_pred[i].argmax().item())
                all_targets.append(dr_grade[i].item())
    mean_dice = finalize_dice(inter_total, union_total).mean().item() if union_total.sum() > 0 else 0.0
    kappa = cohen_kappa_score(all_targets, all_preds, weights="quadratic") if len(all_preds) > 1 else 0.0
    acc = accuracy_score(all_targets, all_preds) if len(all_preds) > 1 else 0.0
    return mean_dice, kappa, acc


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[7G] Using device: {device}")

    full_dataset = IDRiDJointDataset(
        seg_img_dir=args.seg_img_dir, seg_mask_dir=args.seg_mask_dir,
        cls_img_dir=args.cls_img_dir, cls_csv=args.cls_csv,
        image_size=args.image_size, augment=False, zoom_crop_prob=0.25, use_clahe=True,
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

    model = JointSegClsNet(num_lesion_classes=len(LESIONS), num_dr_classes=5, pretrained=True).to(device)
    model.set_encoder_trainable(False)
    print(f"Swin encoder FROZEN for the first {args.freeze_epochs} epochs")

    encoder_params, bottleneck_params, decoder_params, classifier_params = [], [], [], []
    for name, param in model.named_parameters():
        if name.startswith("encoder."):
            encoder_params.append(param)
        elif name.startswith("bottleneck_transformer.") or name.startswith("attn_pool."):
            bottleneck_params.append(param)
        elif name.startswith("cls_head.") or name.startswith("ex_extractor.") or name.startswith("se_extractor."):
            classifier_params.append(param)
        else:
            decoder_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": 1e-5},
        {"params": bottleneck_params, "lr": 5e-5},
        {"params": decoder_params, "lr": 1e-4},
        {"params": classifier_params, "lr": 2e-4},
    ], weight_decay=args.weight_decay)

    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159265))).item()
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))
    os.makedirs(args.out_dir, exist_ok=True)
    best_score = -1.0

    for epoch in range(args.epochs):
        if epoch == args.freeze_epochs:
            model.set_encoder_trainable(True)
            print(f"Epoch {epoch+1}: Swin encoder UNFROZEN")

        model.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"[7G] Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            image = batch["image"].to(device); mask = batch["mask"].to(device)
            seg_valid = batch["seg_valid"].to(device); dr_grade = batch["dr_grade"].to(device)
            cls_valid = batch["cls_valid"].to(device)
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", enabled=(device.type == "cuda")):
                seg_pred, cls_pred, aux_preds = model(image)
                loss, seg_l, cls_l = compute_loss(seg_pred, aux_preds, mask, seg_valid, cls_pred, dr_grade, cls_valid,
                                                   seg_focal_alpha, args.seg_weight, args.cls_weight)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer); scaler.update()
            running += loss.item()
            pbar.set_postfix(loss=loss.item(), seg=seg_l.item(), cls=cls_l.item())

        scheduler.step()
        val_dice, val_kappa, val_acc = validate(model, val_loader, device)
        # Same 0.7*Kappa + 0.3*Dice metric as 7E, to keep checkpoint selection comparable
        combined = 0.7 * max(val_kappa, 0.0) + 0.3 * val_dice
        print(f"Epoch {epoch+1}: val Dice={val_dice:.4f} | val Kappa={val_kappa:.4f} | val Acc={val_acc:.4f} | "
              f"combined(0.7K+0.3D)={combined:.4f}")

        torch.save(model.state_dict(), os.path.join(args.out_dir, "model_last.pth"))
        if combined > best_score:
            best_score = combined
            torch.save(model.state_dict(), os.path.join(args.out_dir, "model_best.pth"))
            print(f"  -> New best saved (combined={combined:.4f})")

    print(f"\n[7G] Training complete. Best combined score: {best_score:.4f}")


if __name__ == "__main__":
    main()
