import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import cohen_kappa_score
from tqdm import tqdm

from dataset import IDRiDJointDataset, LESIONS
from model_5_gan_segmentation import Generator, Discriminator
from losses import JointLoss
from metrics import dice_per_lesion, finalize_dice


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seg_img_dir", required=True)
    p.add_argument("--seg_mask_dir", required=True)
    p.add_argument("--cls_img_dir", required=True)
    p.add_argument("--cls_csv", required=True)
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr_g", type=float, default=1e-4)
    p.add_argument("--lr_d", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--val_fraction", type=float, default=0.15)
    p.add_argument("--seg_weight", type=float, default=2.5)
    p.add_argument("--cls_weight", type=float, default=1.3)
    p.add_argument("--adv_weight", type=float, default=0.1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--out_dir", default="checkpoints_5_gan")
    return p.parse_args()


@torch.no_grad()
def validate(generator, loader, device):
    generator.eval()
    inter_total = torch.zeros(len(LESIONS))
    union_total = torch.zeros(len(LESIONS))
    all_preds, all_targets = [], []
    for batch in loader:
        image = batch["image"].to(device); mask = batch["mask"].to(device)
        seg_valid = batch["seg_valid"]; dr_grade = batch["dr_grade"]; cls_valid = batch["cls_valid"]
        seg_pred, cls_pred = generator(image)
        for i in range(image.size(0)):
            if seg_valid[i] == 1:
                inter, uni = dice_per_lesion(seg_pred[i:i+1], mask[i:i+1])
                inter_total += inter.cpu(); union_total += uni.cpu()
            if cls_valid[i] == 1:
                all_preds.append(cls_pred[i].argmax().item())
                all_targets.append(dr_grade[i].item())
    mean_dice = finalize_dice(inter_total, union_total).mean().item() if union_total.sum() > 0 else 0.0
    kappa = cohen_kappa_score(all_targets, all_preds, weights="quadratic") if len(all_preds) > 1 else 0.0
    return mean_dice, kappa


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Architecture 5: GAN] Using device: {device}")

    full_dataset = IDRiDJointDataset(
        seg_img_dir=args.seg_img_dir, seg_mask_dir=args.seg_mask_dir,
        cls_img_dir=args.cls_img_dir, cls_csv=args.cls_csv,
        image_size=args.image_size, augment=False, zoom_crop_prob=0.0, use_clahe=True,
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

    generator = Generator(num_lesion_classes=len(LESIONS), pretrained=True).to(device)
    discriminator = Discriminator(in_channels=3 + len(LESIONS)).to(device)

    task_criterion = JointLoss(seg_weight=args.seg_weight, cls_weight=args.cls_weight, seg_focal_alpha=seg_focal_alpha)
    adv_criterion = nn.BCEWithLogitsLoss()

    opt_g = torch.optim.Adam(generator.parameters(), lr=args.lr_g, weight_decay=args.weight_decay)
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr_d, weight_decay=args.weight_decay)

    os.makedirs(args.out_dir, exist_ok=True)
    best_score = -1.0

    for epoch in range(args.epochs):
        generator.train(); discriminator.train()
        running_g, running_d = 0.0, 0.0
        pbar = tqdm(train_loader, desc=f"[GAN] Epoch {epoch+1}/{args.epochs}")

        for batch in pbar:
            image = batch["image"].to(device); mask = batch["mask"].to(device)
            seg_valid = batch["seg_valid"].to(device); dr_grade = batch["dr_grade"].to(device)
            cls_valid = batch["cls_valid"].to(device)
            has_seg = seg_valid.sum() > 0

            seg_pred, cls_pred = generator(image)
            seg_prob = torch.sigmoid(seg_pred)

            # ---- Train Discriminator (only on segmentation-labeled samples) ----
            loss_d_val = 0.0
            if has_seg:
                opt_d.zero_grad()
                idx = seg_valid.bool()
                real_score = discriminator(image[idx], mask[idx])
                fake_score = discriminator(image[idx], seg_prob[idx].detach())
                loss_d = 0.5 * (adv_criterion(real_score, torch.ones_like(real_score)) +
                                 adv_criterion(fake_score, torch.zeros_like(fake_score)))
                loss_d.backward()
                opt_d.step()
                loss_d_val = loss_d.item()

            # ---- Train Generator (task loss + adversarial loss) ----
            opt_g.zero_grad()
            task_loss, seg_l, cls_l = task_criterion(seg_pred, mask, seg_valid, cls_pred, dr_grade, cls_valid)

            adv_loss = torch.tensor(0.0, device=device)
            if has_seg:
                idx = seg_valid.bool()
                fake_score2 = discriminator(image[idx], seg_prob[idx])
                adv_loss = adv_criterion(fake_score2, torch.ones_like(fake_score2))

            total_g_loss = task_loss + args.adv_weight * adv_loss
            total_g_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=2.0)
            opt_g.step()

            running_g += total_g_loss.item()
            running_d += loss_d_val
            pbar.set_postfix(G=total_g_loss.item(), D=loss_d_val, seg=seg_l.item(), cls=cls_l.item(), adv=adv_loss.item())

        val_dice, val_kappa = validate(generator, val_loader, device)
        combined = 0.5 * val_dice + 0.5 * max(val_kappa, 0.0)
        print(f"Epoch {epoch+1}: G loss={running_g/len(train_loader):.4f} | D loss={running_d/len(train_loader):.4f} | "
              f"val Dice={val_dice:.4f} | val Kappa={val_kappa:.4f} | combined={combined:.4f}")

        torch.save(generator.state_dict(), os.path.join(args.out_dir, "model_last.pth"))
        if combined > best_score:
            best_score = combined
            torch.save(generator.state_dict(), os.path.join(args.out_dir, "model_best.pth"))
            print(f"  -> New best saved (combined={combined:.4f})")

    print(f"\n[Architecture 5] Training complete. Best combined val score: {best_score:.4f}")


if __name__ == "__main__":
    main()
