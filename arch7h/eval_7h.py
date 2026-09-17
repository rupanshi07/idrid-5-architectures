import argparse
import os

import torch
import numpy as np

from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix
)

import matplotlib.pyplot as plt

from dataset import (
    IDRiDJointDataset,
    LESIONS
)

from model_7h import JointSegClsNet


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
        "--checkpoint",
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
        "--num_workers",
        type=int,
        default=4
    )

    p.add_argument(
        "--cm_out",
        default="confusion_matrix.png"
    )

    return p.parse_args()


# ============================================================
# DICE
# ============================================================

def dice_per_lesion(
    pred,
    target,
    threshold=0.5,
    eps=1e-6
):

    prob = torch.sigmoid(
        pred
    )

    binary = (
        prob > threshold
    ).float()

    intersection = (
        binary * target
    ).sum(
        dim=(0, 2, 3)
    )

    denominator = (
        binary.sum(
            dim=(0, 2, 3)
        )
        +
        target.sum(
            dim=(0, 2, 3)
        )
    )

    dice = (
        2.0 * intersection
        +
        eps
    ) / (
        denominator
        +
        eps
    )

    return dice


# ============================================================
# CONFUSION MATRIX
# ============================================================

def plot_confusion_matrix(
    cm,
    output_path
):

    plt.figure(
        figsize=(7, 6)
    )

    plt.imshow(cm)

    plt.title(
        "7H Confusion Matrix"
    )

    plt.colorbar()

    plt.xlabel(
        "Predicted Grade"
    )

    plt.ylabel(
        "True Grade"
    )

    ticks = np.arange(5)

    plt.xticks(
        ticks,
        [str(i) for i in range(5)]
    )

    plt.yticks(
        ticks,
        [str(i) for i in range(5)]
    )

    for i in range(5):

        for j in range(5):

            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center"
            )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200
    )

    plt.close()


# ============================================================
# MAIN
# ============================================================

@torch.no_grad()
def main():

    args = get_args()

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

    dataset = IDRiDJointDataset(

        seg_img_dir=args.seg_img_dir,

        seg_mask_dir=args.seg_mask_dir,

        cls_img_dir=args.cls_img_dir,

        cls_csv=args.cls_csv,

        image_size=args.image_size,

        augment=False,

        use_clahe=True
    )

    loader = DataLoader(

        dataset,

        batch_size=args.batch_size,

        shuffle=False,

        num_workers=args.num_workers,

        pin_memory=(
            device.type == "cuda"
        )
    )

    print(
        f"Total test samples: "
        f"{len(dataset)}"
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = JointSegClsNet(
        num_lesion_classes=2,
        num_dr_classes=5,
        pretrained=False
    ).to(device)

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device
    )

    model.load_state_dict(
        checkpoint
    )

    model.eval()

    # --------------------------------------------------------
    # STORAGE
    # --------------------------------------------------------

    dice_sum = torch.zeros(
        len(LESIONS)
    )

    dice_count = 0

    predictions = []

    targets = []

    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------

    for batch in loader:

        image = batch[
            "image"
        ].to(device)

        mask = batch[
            "mask"
        ].to(device)

        seg_valid = batch[
            "seg_valid"
        ]

        dr_grade = batch[
            "dr_grade"
        ]

        cls_valid = batch[
            "cls_valid"
        ]

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

                dice = dice_per_lesion(

                    seg_out[
                        i:i + 1
                    ].float(),

                    mask[
                        i:i + 1
                    ]
                )

                dice_sum += dice.cpu()

                dice_count += 1

            # Classification
            if cls_valid[i].item() == 1:

                predictions.append(
                    cls_out[
                        i
                    ].argmax().item()
                )

                targets.append(
                    dr_grade[
                        i
                    ].item()
                )

    # --------------------------------------------------------
    # DICE RESULTS
    # --------------------------------------------------------

    dice = (
        dice_sum
        /
        max(dice_count, 1)
    )

    mean_dice = dice.mean().item()

    print(
        "\nPer-lesion Dice:"
    )

    for i, lesion in enumerate(
        LESIONS
    ):

        print(
            f"  {lesion}: "
            f"{dice[i].item():.4f}"
        )

    print(
        f"Mean Dice: "
        f"{mean_dice:.4f}"
    )

    # --------------------------------------------------------
    # CLASSIFICATION
    # --------------------------------------------------------

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

        cm = confusion_matrix(
            targets,
            predictions,
            labels=[
                0,
                1,
                2,
                3,
                4
            ]
        )

    else:

        accuracy = 0.0

        kappa = 0.0

        cm = np.zeros(
            (5, 5),
            dtype=int
        )

    print(
        f"Accuracy: "
        f"{accuracy:.4f}"
    )

    print(
        f"Quadratic Kappa: "
        f"{kappa:.4f} "
        f"(n={len(targets)})"
    )

    print(
        "\nConfusion matrix:"
    )

    print(cm)

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    output_dir = os.path.dirname(
        args.cm_out
    )

    if output_dir:

        os.makedirs(
            output_dir,
            exist_ok=True
        )

    plot_confusion_matrix(
        cm,
        args.cm_out
    )

    print(
        f"Confusion matrix saved to "
        f"{args.cm_out}"
    )


if __name__ == "__main__":

    main()
