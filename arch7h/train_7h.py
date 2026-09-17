import os
import argparse
import random
import numpy as np

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from dataset import IDRiDJointDataset, LESIONS
from model_7h import JointSegClsNet
from losses_7h import segmentation_loss, hybrid_classification_loss

# ============================================================
# SEED
# ============================================================

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# ARGUMENTS
# ============================================================

def get_args():

    p = argparse.ArgumentParser()

    p.add_argument(
        "--seg_img_dir",
        required=True
    )

    p.add_argument(
        "--seg_mask_dir",
        required=True
    )

    p.add_argument(
        "--cls_img_dir",
        required=True
    )

    p.add_argument(
        "--cls_csv",
        required=True
    )

    p.add_argument(
        "--image_size",
        type=int,
        default=512
    )

    p.add_argument(
        "--batch_size",
        type=int,
        default=1
    )

    p.add_argument(
        "--epochs",
        type=int,
        default=60
    )

    p.add_argument(
        "--freeze_epochs",
        type=int,
        default=5
    )

    p.add_argument(
        "--warmup_epochs",
        type=int,
        default=5
    )

    p.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4
    )

    p.add_argument(
        "--val_fraction",
        type=float,
        default=0.15
    )

    p.add_argument(
        "--seg_weight",
        type=float,
        default=2.0
    )

    p.add_argument(
        "--cls_weight",
        type=float,
        default=1.5
    )

    p.add_argument(
        "--num_workers",
        type=int,
        default=4
    )

    p.add_argument(
        "--out_dir",
        default="checkpoints_7h"
    )

    p.add_argument(
        "--seed",
        type=int,
        default=42
    )

    return p.parse_args()


# ============================================================
# DICE
# ============================================================

def dice_statistics(
    pred,
    target
):

    pred = (
        torch.sigmoid(pred)
        > 0.5
    ).float()

    intersection = (
        pred * target
    ).sum(
        dim=(0, 2, 3)
    )

    denominator = (
        pred.sum(
            dim=(0, 2, 3)
        )
        +
        target.sum(
            dim=(0, 2, 3)
        )
    )

    return (
        intersection.cpu(),
        denominator.cpu()
    )


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def validate(
    model,
    loader,
    device
):

    from sklearn.metrics import (
        accuracy_score,
        cohen_kappa_score
    )

    model.eval()

    inter_total = torch.zeros(
        len(LESIONS)
    )

    denom_total = torch.zeros(
        len(LESIONS)
    )

    predictions = []
    targets = []

    for batch in loader:

        image = batch["image"].to(
            device,
            non_blocking=True
        )

        mask = batch["mask"].to(
            device,
            non_blocking=True
        )

        seg_valid = batch[
            "seg_valid"
        ].to(device)

        dr_grade = batch[
            "dr_grade"
        ].to(device)

        cls_valid = batch[
            "cls_valid"
        ].to(device)

        with torch.autocast(
            device_type="cuda",
            enabled=device.type == "cuda"
        ):

            seg_out, cls_out, _ = model(
                image
            )

        # Segmentation
        for i in range(
            image.size(0)
        ):

            if seg_valid[i].item() == 1:

                inter, denom = dice_statistics(
                    seg_out[i:i + 1].float(),
                    mask[i:i + 1]
                )

                inter_total += inter
                denom_total += denom

            # Classification
            if cls_valid[i].item() == 1:

                predictions.append(
                    cls_out[i].argmax().item()
                )

                targets.append(
                    dr_grade[i].item()
                )

    dice = (
        2.0 * inter_total
        /
        (denom_total + 1e-6)
    )

    mean_dice = dice.mean().item()

    if len(targets) > 1:

        accuracy = accuracy_score(
            targets,
            predictions
        )

        kappa = cohen_kappa_score(
            targets,
            predictions,
            weights="quadratic"
        )

    else:

        accuracy = 0.0
        kappa = 0.0

    return (
        mean_dice,
        accuracy,
        kappa,
        dice
    )


# ============================================================
# MAIN
# ============================================================

def main():

    args = get_args()

    set_seed(
        args.seed
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"[7H] Using device: {device}"
    )

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    full_dataset = IDRiDJointDataset(

        seg_img_dir=args.seg_img_dir,

        seg_mask_dir=args.seg_mask_dir,

        cls_img_dir=args.cls_img_dir,

        cls_csv=args.cls_csv,

        image_size=args.image_size,

        augment=False,

        zoom_crop_prob=0.20,

        use_clahe=True
    )

    train_set, val_set = (
        full_dataset.split_train_val(
            val_fraction=args.val_fraction,
            seed=args.seed
        )
    )

    print(
        f"Train samples: {len(train_set)}"
    )

    print(
        f"Validation samples: {len(val_set)}"
    )

    # --------------------------------------------------------
    # CLASS WEIGHTS
    # --------------------------------------------------------

    counts = full_dataset.class_counts()

    print(
        f"Classification counts: {counts}"
    )

    # Square-root inverse frequency.
    # Less aggressive than direct inverse frequency.
    weights = []

    for count in counts:

        if count > 0:

            weights.append(
                1.0 / np.sqrt(count)
            )

        else:

            weights.append(0.0)

    weights = torch.tensor(
        weights,
        dtype=torch.float32
    )

    weights = (
        weights
        /
        weights.mean()
    )

    weights = weights.to(
        device
    )

    print(
        f"Class weights: {weights.tolist()}"
    )

    # --------------------------------------------------------
    # SAMPLER
    # --------------------------------------------------------

    sample_weights = (
        train_set.sample_weights()
    )

    sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=len(train_set),
        replacement=True
    )

    train_loader = DataLoader(

        train_set,

        batch_size=args.batch_size,

        sampler=sampler,

        num_workers=args.num_workers,

        pin_memory=(
            device.type == "cuda"
        )
    )

    val_loader = DataLoader(

        val_set,

        batch_size=args.batch_size,

        shuffle=False,

        num_workers=args.num_workers,

        pin_memory=(
            device.type == "cuda"
        )
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = JointSegClsNet(
        num_lesion_classes=2,
        num_dr_classes=5,
        pretrained=True
    ).to(device)

    # --------------------------------------------------------
    # FREEZE ENCODER
    # --------------------------------------------------------

    for param in model.encoder.parameters():

        param.requires_grad = False

    print(
        f"Swin encoder frozen for "
        f"{args.freeze_epochs} epochs"
    )

    # --------------------------------------------------------
    # PARAMETER GROUPS
    # --------------------------------------------------------

    encoder_params = []

    bottleneck_params = []

    decoder_params = []

    classifier_params = []

    for name, param in model.named_parameters():

        if not param.requires_grad:

            encoder_params.append(
                param
            )

        elif name.startswith(
            "encoder."
        ):

            encoder_params.append(
                param
            )

        elif name.startswith(
            "bottleneck_transformer."
        ):

            bottleneck_params.append(
                param
            )

        elif name.startswith(
            "cls_head."
        ):

            classifier_params.append(
                param
            )

        else:

            decoder_params.append(
                param
            )

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(

        [

            {
                "params": encoder_params,
                "lr": 1e-5
            },

            {
                "params": bottleneck_params,
                "lr": 5e-5
            },

            {
                "params": decoder_params,
                "lr": 1e-4
            },

            {
                "params": classifier_params,
                "lr": 2e-4
            }

        ],

        weight_decay=args.weight_decay
    )

    # --------------------------------------------------------
    # COSINE SCHEDULER
    # --------------------------------------------------------

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(

        optimizer,

        T_max=args.epochs,

        eta_min=1e-6
    )

    # --------------------------------------------------------
    # AMP
    # --------------------------------------------------------

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda"
    )

    os.makedirs(
        args.out_dir,
        exist_ok=True
    )

    best_score = -1.0

    # ========================================================
    # TRAINING
    # ========================================================

    for epoch in range(
        args.epochs
    ):

        # Unfreeze encoder
        if epoch == args.freeze_epochs:

            for param in model.encoder.parameters():

                param.requires_grad = True

            print(
                f"Epoch {epoch + 1}: "
                f"Swin encoder UNFROZEN"
            )

        model.train()

        running_loss = 0.0

        pbar = tqdm(
            train_loader,
            desc=(
                f"[7H] "
                f"Epoch {epoch + 1}/{args.epochs}"
            )
        )

        for batch in pbar:

            image = batch[
                "image"
            ].to(
                device,
                non_blocking=True
            )

            mask = batch[
                "mask"
            ].to(
                device,
                non_blocking=True
            )

            seg_valid = batch[
                "seg_valid"
            ].to(device)

            dr_grade = batch[
                "dr_grade"
            ].to(device)

            cls_valid = batch[
                "cls_valid"
            ].to(device)

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.autocast(
                device_type="cuda",
                enabled=device.type == "cuda"
            ):

                seg_out, cls_out, _ = model(
                    image
                )

                # --------------------------------------------
                # SEGMENTATION
                # --------------------------------------------

                if seg_valid.sum() > 0:

                    valid_seg = (
                        seg_valid.bool()
                    )

                    seg_loss = segmentation_loss(

                        seg_out[
                            valid_seg
                        ],

                        mask[
                            valid_seg
                        ]
                    )

                else:

                    seg_loss = torch.tensor(
                        0.0,
                        device=device
                    )

                # --------------------------------------------
                # CLASSIFICATION
                # --------------------------------------------

                if cls_valid.sum() > 0:

                    valid_cls = (
                        cls_valid.bool()
                    )

                    safe_target = (
                        dr_grade[
                            valid_cls
                        ].clone()
                    )

                    safe_target[
                        safe_target < 0
                    ] = 0

                    cls_loss = (
                        hybrid_classification_loss(

                            cls_out[
                                valid_cls
                            ],

                            safe_target,

                            class_weights=weights
                        )
                    )

                else:

                    cls_loss = torch.tensor(
                        0.0,
                        device=device
                    )

                # --------------------------------------------
                # TOTAL
                # --------------------------------------------

                total_loss = (
                    args.seg_weight * seg_loss
                    +
                    args.cls_weight * cls_loss
                )

            scaler.scale(
                total_loss
            ).backward()

            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            scaler.step(
                optimizer
            )

            scaler.update()

            running_loss += (
                total_loss.item()
            )

            pbar.set_postfix(
                loss=f"{total_loss.item():.4f}",
                seg=f"{seg_loss.item():.4f}",
                cls=f"{cls_loss.item():.4f}"
            )

        scheduler.step()

        # ====================================================
        # VALIDATION
        # ====================================================

        val_dice, val_acc, val_kappa, lesion_dice = validate(
            model,
            val_loader,
            device
        )

        # Main objective:
        # Kappa + segmentation quality
        combined = (
            0.55 * max(
                val_kappa,
                0.0
            )
            +
            0.30 * val_dice
            +
            0.15 * val_acc
        )

        print(
            f"\nEpoch {epoch + 1}"
        )

        print(
            f"Val Dice: {val_dice:.4f}"
        )

        print(
            f"  EX Dice: "
            f"{lesion_dice[0].item():.4f}"
        )

        print(
            f"  SE Dice: "
            f"{lesion_dice[1].item():.4f}"
        )

        print(
            f"Val Accuracy: "
            f"{val_acc:.4f}"
        )

        print(
            f"Val Kappa: "
            f"{val_kappa:.4f}"
        )

        print(
            f"Combined: "
            f"{combined:.4f}"
        )

        # ----------------------------------------------------
        # SAVE LAST
        # ----------------------------------------------------

        torch.save(
            model.state_dict(),
            os.path.join(
                args.out_dir,
                "model_last.pth"
            )
        )

        # ----------------------------------------------------
        # SAVE BEST
        # ----------------------------------------------------

        if combined > best_score:

            best_score = combined

            torch.save(
                model.state_dict(),
                os.path.join(
                    args.out_dir,
                    "model_best.pth"
                )
            )

            print(
                ">>> NEW BEST MODEL SAVED"
            )

    print(
        f"\n[7H] Training complete."
    )

    print(
        f"Best combined score: "
        f"{best_score:.4f}"
    )


if __name__ == "__main__":

    main()
