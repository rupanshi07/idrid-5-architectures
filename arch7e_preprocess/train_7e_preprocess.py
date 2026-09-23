import argparse
import os
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import cohen_kappa_score, accuracy_score
from tqdm import tqdm

from dataset import IDRiDJointDataset, LESIONS
from model_7g import JointSegClsNet
from losses_7g_v3 import weighted_segmentation_loss, build_classification_criterion


def dice_per_lesion(pred, target, eps=1e-6):
    pred_bin = (torch.sigmoid(pred) > 0.5).float()
    dims = (0, 2, 3)
    intersection = (pred_bin * target).sum(dims)
    union = pred_bin.sum(dims) + target.sum(dims)
    return intersection, union


def finalize_dice(intersection, union, eps=1e-6):
    return (2 * intersection + eps) / (union + eps)


def compute_loss(seg_out, aux_outs, mask, seg_valid, cls_out, dr_grade, cls_valid,
                  seg_focal_alpha, lesion_weights, cls_criterion, seg_weight, cls_weight, aux_weights=(0.4, 0.2)):
    if seg_valid.sum() > 0:
        valid_idx = seg_valid.bool()
        seg_loss = weighted_segmentation_loss(seg_out[valid_idx], mask[valid_idx], seg_focal_alpha, lesion_weights)
        for aux_out, w in zip(aux_outs, aux_weights):
            seg_loss = seg_loss + w * weighted_segmentation_loss(aux_out[valid_idx], mask[valid_idx], seg_focal_alpha, lesion_weights)
    else:
        seg_loss = torch.tensor(0.0, device=seg_out.device)

    if cls_valid.sum() > 0:
        valid_idx = cls_valid.bool()
        safe_target = dr_grade.clone()
        safe_target[safe_target < 0] = 0
        cls_loss = cls_criterion(cls_out[valid_idx], safe_target[valid_idx])
    else:
        cls_loss = torch.tensor(0.0, device=cls_out.device)

    total = seg_weight * seg_loss + cls_weight * cls_loss
    return total, seg_loss.detach(), cls_loss.detach()


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--preset", required=True, choices=["P0", "P1", "P2a", "P2b", "P2c", "P3", "P4", "P5", "P6", "P7"])
    p.add_argument("--seg_img_dir", required=True)
    p.add_argument("--seg_mask_dir", required=True)
    p.add_argument("--cls_img_dir", required=True)
    p.add_argument("--cls_csv", required=True)
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum_steps", type=int, default=4)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--warmup_epochs", type=int, default=3)
    p.add_argument("--freeze_epochs", type=int, default=5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--val_fraction", type=float, default=0.15)
    p.add_argument("--seg_weight", type=float, default=2.0)
    p.add_argument("--cls_weight", type=float, default=1.25)
    p.add_argument("--ex_weight", type=float, default=1.0)
    p.add_argument("--se_weight", type=float, default=1.3)
    p.add_argument("--num_workers", type=int, default=4)
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
    dice_per_les = finalize_dice(inter_total, union_total) if union_total.sum() > 0 else torch.zeros(len(LESIONS))
    mean_dice = dice_per_les.mean().item()
    kappa = cohen_kappa_score(all_targets, all_preds, weights="quadratic") if len(all_preds) > 1 else 0.0
    acc = accuracy_score(all_targets, all_preds) if len(all_preds) > 1 else 0.0
    return mean_dice, kappa, acc, dice_per_les.tolist()


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = f"checkpoints_{args.preset}"
    os.makedirs(out_dir, exist_ok=True)
    print(f"[{args.preset}] Using device: {device}")

    full_dataset = IDRiDJointDataset(
        seg_img_dir=args.seg_img_dir, seg_mask_dir=args.seg_mask_dir,
        cls_img_dir=args.cls_img_dir, cls_csv=args.cls_csv,
        image_size=args.image_size, augment=False, zoom_crop_prob=0.25,
        preprocess_preset=args.preset,
    )
    train_set, val_set = full_dataset.split_train_val(val_fraction=args.val_fraction)
    print(f"Train samples: {len(train_set)} | Val samples: {len(val_set)}")

    pos_weight = train_set.compute_pos_weight(len(LESIONS), args.image_size)
    seg_focal_alpha = (pos_weight / (1.0 + pos_weight)).clamp(0.5, 0.98).to(device)
    lesion_weights = [args.ex_weight, args.se_weight]

    class_weights = torch.tensor([1.0, 1.4, 1.0, 1.1, 1.3], device=device)
    cls_criterion = build_classification_criterion(class_weights, label_smoothing=0.05)

    sample_weights = train_set.sample_weights()
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_set), replacement=True)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=sampler,
                               num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    model = JointSegClsNet(num_lesion_classes=len(LESIONS), num_dr_classes=5, pretrained=True).to(device)
    model.set_encoder_trainable(False)

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
    best_kappa, best_dice = -1.0, -1.0

    for epoch in range(args.epochs):
        if epoch == args.freeze_epochs:
            model.set_encoder_trainable(True)
            print(f"Epoch {epoch+1}: Swin encoder UNFROZEN")

        model.train()
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"[{args.preset}] Epoch {epoch+1}/{args.epochs}")
        for step, batch in enumerate(pbar):
            image = batch["image"].to(device); mask = batch["mask"].to(device)
            seg_valid = batch["seg_valid"].to(device); dr_grade = batch["dr_grade"].to(device)
            cls_valid = batch["cls_valid"].to(device)
            with torch.autocast(device_type="cuda", enabled=(device.type == "cuda")):
                seg_pred, cls_pred, aux_preds = model(image)
                loss, seg_l, cls_l = compute_loss(seg_pred, aux_preds, mask, seg_valid, cls_pred, dr_grade, cls_valid,
                                                   seg_focal_alpha, lesion_weights, cls_criterion, args.seg_weight, args.cls_weight)
                loss_scaled = loss / args.grad_accum_steps
            scaler.scale(loss_scaled).backward()
            if (step + 1) % args.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer); scaler.update()
                optimizer.zero_grad()
            pbar.set_postfix(loss=loss.item(), seg=seg_l.item(), cls=cls_l.item())

        scheduler.step()
        val_dice, val_kappa, val_acc, dice_per_les = validate(model, val_loader, device)
        print(f"Epoch {epoch+1}: val Dice={val_dice:.4f} (EX={dice_per_les[0]:.4f}, SE={dice_per_les[1]:.4f}) | "
              f"val Kappa={val_kappa:.4f} | val Acc={val_acc:.4f}")

        torch.save(model.state_dict(), os.path.join(out_dir, "model_last.pth"))
        if val_kappa > best_kappa:
            best_kappa = val_kappa
            torch.save(model.state_dict(), os.path.join(out_dir, "model_best_kappa.pth"))
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), os.path.join(out_dir, "model_best_dice.pth"))

    print(f"\n[{args.preset}] Complete. Best Kappa: {best_kappa:.4f} | Best Dice: {best_dice:.4f}")


if __name__ == "__main__":
    main()
