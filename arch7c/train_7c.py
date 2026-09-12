import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from dataset import IDRiDJointDataset
from model_7c import JointSegClsNet
from losses import JointLoss

SEED = 42
IMAGE_SIZE = 512
BATCH_SIZE = 1
NUM_WORKERS = 4
EPOCHS = 40
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = "checkpoints_7c"

SEG_IMG_DIR = r"..\Augmented_Seg_Images"
SEG_MASK_DIR = r"..\Augmented_Seg_Masks"
CLS_IMG_DIR = r"..\Balanced_Training_Images"
CLS_CSV = r"..\Dataset\B_DiseaseGrading\Groundtruths\Training_Labels_Balanced.csv"


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def train_one_epoch(model, train_loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss = total_seg_loss = total_cls_loss = 0.0
    batches = 0
    for batch in train_loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        seg_valid = batch["seg_valid"].to(device, non_blocking=True)
        grades = batch["dr_grade"].to(device, non_blocking=True)
        cls_valid = batch["cls_valid"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", enabled=(device == "cuda")):
            seg_pred, cls_pred = model(images)
            loss, seg_loss, cls_loss = criterion(seg_pred, masks, seg_valid, cls_pred, grades, cls_valid)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item(); total_seg_loss += seg_loss.item(); total_cls_loss += cls_loss.item()
        batches += 1
    return total_loss / max(batches, 1), total_seg_loss / max(batches, 1), total_cls_loss / max(batches, 1)


@torch.no_grad()
def validate_loss(model, val_loader, criterion, device):
    model.eval()
    total_loss = total_seg_loss = total_cls_loss = 0.0
    batches = 0
    for batch in val_loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        seg_valid = batch["seg_valid"].to(device, non_blocking=True)
        grades = batch["dr_grade"].to(device, non_blocking=True)
        cls_valid = batch["cls_valid"].to(device, non_blocking=True)
        seg_pred, cls_pred = model(images)
        loss, seg_loss, cls_loss = criterion(seg_pred, masks, seg_valid, cls_pred, grades, cls_valid)
        total_loss += loss.item(); total_seg_loss += seg_loss.item(); total_cls_loss += cls_loss.item()
        batches += 1
    return total_loss / max(batches, 1), total_seg_loss / max(batches, 1), total_cls_loss / max(batches, 1)


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    set_seed(SEED)
    print("Using device:", DEVICE)

    full_dataset = IDRiDJointDataset(
        seg_img_dir=SEG_IMG_DIR, seg_mask_dir=SEG_MASK_DIR,
        cls_img_dir=CLS_IMG_DIR, cls_csv=CLS_CSV,
        image_size=IMAGE_SIZE, augment=False, zoom_crop_prob=0.25, use_clahe=True
    )
    train_dataset, val_dataset = full_dataset.split_train_val(val_fraction=0.15, seed=SEED)
    print("Train samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))
    print("Classification counts:", full_dataset.class_counts())

    sample_weights = train_dataset.sample_weights()
    sampler = WeightedRandomSampler(weights=torch.DoubleTensor(sample_weights), num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    model = JointSegClsNet(num_lesion_classes=2, num_dr_classes=4, pretrained=True).to(DEVICE)

    pos_weight = train_dataset.compute_pos_weight(num_lesions=2, image_size=IMAGE_SIZE)
    print("Segmentation pos_weight:", pos_weight)
    # FIXED: .to(DEVICE) added -- this tensor was staying on CPU and
    # mismatching with GPU tensors inside the loss computation.
    seg_focal_alpha = (pos_weight / (1.0 + pos_weight)).clamp(0.5, 0.98).to(DEVICE)
    print("Segmentation focal alpha:", seg_focal_alpha)

    criterion = JointLoss(seg_weight=2.5, cls_weight=1.5, dice_weight=1.0, focal_weight=1.0,
                           seg_focal_alpha=seg_focal_alpha, seg_focal_gamma=2.0, cls_focal_gamma=2.0, ordinal_pos_weight=None)

    encoder_params, decoder_params, classifier_params, other_params = [], [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("encoder."):
            encoder_params.append(param)
        elif name.startswith("cls_head."):
            classifier_params.append(param)
        elif name.startswith("up") or name.startswith("seg_head") or name.startswith("se_"):
            decoder_params.append(param)
        else:
            other_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": 1e-5},
        {"params": decoder_params, "lr": 1e-4},
        {"params": classifier_params, "lr": 2e-4},
        {"params": other_params, "lr": 1e-4},
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda"))

    best_val_loss = float("inf")
    best_checkpoint = os.path.join(CHECKPOINT_DIR, "best_model.pth")

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_seg, train_cls = train_one_epoch(model, train_loader, criterion, optimizer, scaler, DEVICE)
        val_loss, val_seg, val_cls = validate_loss(model, val_loader, criterion, DEVICE)
        scheduler.step()

        print(f"\nEpoch [{epoch}/{EPOCHS}]")
        print(f"Train Loss: {train_loss:.4f}  Seg: {train_seg:.4f}  Cls: {train_cls:.4f}")
        print(f"Val Loss: {val_loss:.4f}  Seg: {val_seg:.4f}  Cls: {val_cls:.4f}")
        print("LR:", scheduler.get_last_lr())

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "best_val_loss": best_val_loss}, best_checkpoint)
            print("Saved best checkpoint.")

    print("\nTraining complete.")
    print("Best checkpoint:", best_checkpoint)


if __name__ == "__main__":
    main()
