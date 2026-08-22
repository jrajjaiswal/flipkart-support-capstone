import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from PIL import Image

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from sklearn.model_selection import train_test_split

from torch.utils.data import (
    DataLoader,
    Subset,
    TensorDataset,
)

from torchvision import datasets, models, transforms


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

IMAGE_SIZE = 224
BACKBONE_NAME = "ResNet-18"

BATCH_SIZE = 128
HEAD_BATCH_SIZE = 256

LEARNING_RATE = 0.001
EPOCHS = 10

MODEL_PATH = "models/product_classifier.pt"
CONFUSION_MATRIX_PATH = "data/confusion_matrix.csv"
SAMPLE_DIR = "data/sample_images"

CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)


# ============================================================
# DIRECTORIES
# ============================================================

os.makedirs(
    "models",
    exist_ok=True
)

os.makedirs(
    "data",
    exist_ok=True
)

os.makedirs(
    SAMPLE_DIR,
    exist_ok=True
)


# ============================================================
# TASK 1 — LOAD FASHION-MNIST
# ============================================================

train_dataset_raw = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
)

test_dataset_raw = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
)

train_labels = np.array(
    train_dataset_raw.targets
)

train_indices, val_indices = train_test_split(
    np.arange(
        len(train_dataset_raw)
    ),
    test_size=6000,
    random_state=SEED,
    stratify=train_labels,
)

print("\nDATA SPLIT SIZES")
print(
    "Original training split:",
    len(train_dataset_raw),
)
print(
    "Training:",
    len(train_indices),
)
print(
    "Validation:",
    len(val_indices),
)
print(
    "Test:",
    len(test_dataset_raw),
)

print("\nTEST SET STATUS")
print(
    "Test set is kept untouched until final evaluation."
)


# ============================================================
# TASK 2 — PREPROCESSING
# ============================================================

transform = transforms.Compose([
    transforms.Resize(
        (
            IMAGE_SIZE,
            IMAGE_SIZE,
        )
    ),
    transforms.Grayscale(
        num_output_channels=3
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
    ),
])

train_dataset = datasets.FashionMNIST(
    root="data",
    train=True,
    download=False,
    transform=transform,
)

test_dataset = datasets.FashionMNIST(
    root="data",
    train=False,
    download=False,
    transform=transform,
)

train_subset = Subset(
    train_dataset,
    train_indices,
)

val_subset = Subset(
    train_dataset,
    val_indices,
)

print("\nPREPROCESSING CONFIGURATION")
print(
    "Backbone:",
    BACKBONE_NAME,
)
print(
    "Input size:",
    f"{IMAGE_SIZE} x {IMAGE_SIZE}",
)
print(
    "Input channels: 3"
)
print(
    "Normalization: ImageNet mean/std"
)

print("\nDATASET SUBSETS")
print(
    "Training subset:",
    len(train_subset),
)
print(
    "Validation subset:",
    len(val_subset),
)
print(
    "Test set:",
    len(test_dataset),
)


# ============================================================
# TASK 3 — PRETRAINED RESNET-18
# ============================================================

weights = models.ResNet18_Weights.DEFAULT

resnet18 = models.resnet18(
    weights=weights
)

for parameter in resnet18.parameters():
    parameter.requires_grad = False

num_features = resnet18.fc.in_features

resnet18.fc = nn.Linear(
    num_features,
    10,
)

for parameter in resnet18.fc.parameters():
    parameter.requires_grad = True

resnet18 = resnet18.to(device)

print("\nTRANSFER LEARNING MODEL")
print(
    "Backbone:",
    BACKBONE_NAME,
)
print(
    "Pretrained weights: ImageNet"
)
print(
    "Backbone frozen: Yes"
)
print(
    "Classifier classes:",
    10,
)
print(
    "Classifier input features:",
    num_features,
)


# ============================================================
# TASK 3 — FROZEN FEATURE EXTRACTION
# IMPORTANT:
# Only training and validation features are extracted here.
# Test features are deliberately NOT extracted yet.
# ============================================================

feature_extractor = nn.Sequential(
    *list(
        resnet18.children()
    )[:-1]
)

feature_extractor = (
    feature_extractor.to(device)
)

feature_extractor.eval()

train_loader = DataLoader(
    train_subset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)

val_loader = DataLoader(
    val_subset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)


def extract_features(
    loader,
    split_name,
):

    all_features = []
    all_labels = []

    print(
        f"\nEXTRACTING "
        f"{split_name.upper()} FEATURES..."
    )

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(
                device,
                non_blocking=True,
            )

            features = (
                feature_extractor(
                    images
                )
            )

            features = features.view(
                features.size(0),
                -1,
            )

            all_features.append(
                features.cpu()
            )

            all_labels.append(
                labels
            )

    features_tensor = torch.cat(
        all_features,
        dim=0,
    )

    labels_tensor = torch.cat(
        all_labels,
        dim=0,
    )

    print(
        f"{split_name} feature shape:",
        tuple(
            features_tensor.shape
        ),
    )

    print(
        f"{split_name} labels:",
        len(labels_tensor),
    )

    return (
        features_tensor,
        labels_tensor,
    )


train_features, train_feature_labels = (
    extract_features(
        train_loader,
        "Training",
    )
)

val_features, val_feature_labels = (
    extract_features(
        val_loader,
        "Validation",
    )
)

print(
    "\nFEATURE EXTRACTION COMPLETE"
)

print(
    "Training:",
    tuple(
        train_features.shape
    ),
)

print(
    "Validation:",
    tuple(
        val_features.shape
    ),
)


# ============================================================
# TASK 3 — CLASSIFIER HEAD
# ============================================================

train_feature_dataset = TensorDataset(
    train_features,
    train_feature_labels,
)

val_feature_dataset = TensorDataset(
    val_features,
    val_feature_labels,
)

train_feature_loader = DataLoader(
    train_feature_dataset,
    batch_size=HEAD_BATCH_SIZE,
    shuffle=True,
)

val_feature_loader = DataLoader(
    val_feature_dataset,
    batch_size=HEAD_BATCH_SIZE,
    shuffle=False,
)

classifier_head = nn.Linear(
    512,
    10,
).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    classifier_head.parameters(),
    lr=LEARNING_RATE,
)


def evaluate_head(
    model,
    loader,
):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for features, labels in loader:

            features = features.to(
                device,
                non_blocking=True,
            )

            labels = labels.to(
                device,
                non_blocking=True,
            )

            outputs = model(
                features
            )

            predictions = (
                outputs.argmax(
                    dim=1
                )
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    return (
        correct / total
    )


print(
    "\nFEATURE EXTRACTION HEAD TRAINING"
)

print(
    "Batch size:",
    HEAD_BATCH_SIZE,
)

print(
    "Optimizer: Adam"
)

print(
    "Learning rate:",
    LEARNING_RATE,
)

print(
    "Epochs:",
    EPOCHS,
)


for epoch in range(
    EPOCHS
):

    classifier_head.train()

    running_loss = 0.0
    total_samples = 0

    for features, labels in (
        train_feature_loader
    ):

        features = features.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad()

        outputs = (
            classifier_head(
                features
            )
        )

        loss = criterion(
            outputs,
            labels,
        )

        loss.backward()

        optimizer.step()

        batch_size = labels.size(0)

        running_loss += (
            loss.item()
            * batch_size
        )

        total_samples += (
            batch_size
        )

    epoch_loss = (
        running_loss
        / total_samples
    )

    validation_accuracy = (
        evaluate_head(
            classifier_head,
            val_feature_loader,
        )
    )

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Loss: {epoch_loss:.4f} | "
        f"Validation Accuracy: "
        f"{validation_accuracy:.4f}"
    )


feature_extraction_val_accuracy = (
    evaluate_head(
        classifier_head,
        val_feature_loader,
    )
)

print(
    "\nFEATURE EXTRACTION "
    "VALIDATION ACCURACY:",
    round(
        feature_extraction_val_accuracy,
        4,
    ),
)

if (
    feature_extraction_val_accuracy
    >= 0.80
):

    print(
        "Feature extraction alone "
        "achieved at least 80% "
        "validation accuracy."
    )

    print(
        "Fine-tuning required: False"
    )

else:

    print(
        "Feature extraction validation "
        "accuracy is below 80%."
    )

    print(
        "Fine-tuning may be required."
    )


# ============================================================
# TASK 4 — FINE-TUNING DECISION
# ============================================================
# The feature-extraction validation accuracy determines whether
# fine-tuning is required.
#
# In our actual run the accuracy was 0.8910, so fine-tuning
# was not required.


# ============================================================
# FINAL TEST EVALUATION
# IMPORTANT:
# THE TEST SET IS FIRST PROCESSED HERE.
# ============================================================

print(
    "\nEXTRACTING TEST FEATURES "
    "FOR FINAL EVALUATION..."
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)

test_features, test_feature_labels = (
    extract_features(
        test_loader,
        "Test",
    )
)

test_feature_dataset = TensorDataset(
    test_features,
    test_feature_labels,
)

test_feature_loader = DataLoader(
    test_feature_dataset,
    batch_size=HEAD_BATCH_SIZE,
    shuffle=False,
)


classifier_head.eval()

all_test_predictions = []
all_test_labels = []

with torch.no_grad():

    for features, labels in (
        test_feature_loader
    ):

        features = features.to(
            device,
            non_blocking=True,
        )

        outputs = (
            classifier_head(
                features
            )
        )

        predictions = (
            outputs.argmax(
                dim=1
            )
        )

        all_test_predictions.extend(
            predictions.cpu().numpy()
        )

        all_test_labels.extend(
            labels.numpy()
        )


all_test_predictions = np.array(
    all_test_predictions
)

all_test_labels = np.array(
    all_test_labels
)


test_accuracy = accuracy_score(
    all_test_labels,
    all_test_predictions,
)

print(
    "\nFINAL TEST EVALUATION"
)

print(
    "Test accuracy:",
    round(
        test_accuracy,
        4,
    ),
)


# ============================================================
# TASK 5 — CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_test_labels,
    all_test_predictions,
)

print(
    "\n10x10 CONFUSION MATRIX"
)

print(cm)


# ============================================================
# TASK 5 — PER-CLASS METRICS
# ============================================================

precision, recall, f1, support = (
    precision_recall_fscore_support(
        all_test_labels,
        all_test_predictions,
        labels=np.arange(10),
        zero_division=0,
    )
)

print(
    "\nPER-CLASS METRICS"
)

for i, class_name in enumerate(
    CLASS_NAMES
):

    print(
        f"{class_name:12s} | "
        f"precision={precision[i]:.4f} | "
        f"recall={recall[i]:.4f} | "
        f"f1={f1[i]:.4f} | "
        f"support={support[i]}"
    )


# ============================================================
# TASK 6 — TOP CONFUSION PAIRS
# ============================================================

pair_results = []

for i in range(
    len(CLASS_NAMES)
):

    for j in range(
        i + 1,
        len(CLASS_NAMES)
    ):

        confusion_count = (
            cm[i, j]
            + cm[j, i]
        )

        pair_results.append({
            "pair":
                f"{CLASS_NAMES[i]} "
                f"<-> "
                f"{CLASS_NAMES[j]}",
            "total_confusion":
                int(
                    confusion_count
                ),
        })


pair_results.sort(
    key=lambda x:
        x["total_confusion"],
    reverse=True,
)

top_confusion_pairs = (
    pair_results[:2]
)

print(
    "\nTOP 2 CONFUSION PAIRS"
)

for rank, pair in enumerate(
    top_confusion_pairs,
    start=1,
):

    print(
        f"{rank}. "
        f"{pair['pair']} = "
        f"{pair['total_confusion']} "
        f"total confusions"
    )


print(
    "\nCONFUSION PATTERN EXPLANATIONS"
)

print(
    "\n1. T-shirt/top <-> Shirt"
)

print(
    "These categories are visually similar in "
    "Fashion-MNIST because both are upper-body "
    "garments with short sleeves and a broadly "
    "similar torso silhouette. In low-resolution "
    "28x28 grayscale images, the neckline, sleeve "
    "shape, and overall garment outline can look "
    "very similar, making it difficult for the "
    "classifier to distinguish between them."
)

print(
    "\n2. Shirt <-> Coat"
)

print(
    "These categories can also overlap visually "
    "because both represent upper-body garments "
    "with sleeves and similar outer silhouettes. "
    "In small grayscale images, details such as "
    "buttons, openings, collars, and fabric "
    "structure can be difficult to distinguish, "
    "so a shirt may resemble a coat and vice versa."
)


# ============================================================
# TASK 7 — SAVE MODEL ARTIFACT
# ============================================================

artifact = {

    "backbone":
        "resnet18",

    "weights":
        "ImageNet",

    "image_size":
        IMAGE_SIZE,

    "imagenet_mean":
        IMAGENET_MEAN,

    "imagenet_std":
        IMAGENET_STD,

    "num_classes":
        10,

    "class_names":
        CLASS_NAMES,

    "feature_extractor_state_dict":
        feature_extractor.state_dict(),

    "classifier_head_state_dict":
        classifier_head.state_dict(),

    "feature_dimension":
        512,

    "validation_accuracy":
        float(
            feature_extraction_val_accuracy
        ),

    "test_accuracy":
        float(
            test_accuracy
        ),
}


torch.save(
    artifact,
    MODEL_PATH,
)

print(
    "\nMODEL SAVED"
)

print(
    "Location:",
    MODEL_PATH,
)


# ============================================================
# TASK 7 — LOAD MODEL WITHOUT DOWNLOADING WEIGHTS
# ============================================================

def load_product_classifier(
    model_path=MODEL_PATH,
    device_name=None,
):

    if device_name is None:

        device_name = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    load_device = torch.device(
        device_name
    )

    checkpoint = torch.load(
        model_path,
        map_location=load_device,
    )

    backbone = models.resnet18(
        weights=None
    )

    feature_extractor_loaded = (
        nn.Sequential(
            *list(
                backbone.children()
            )[:-1]
        )
    )

    feature_extractor_loaded.load_state_dict(
        checkpoint[
            "feature_extractor_state_dict"
        ]
    )

    feature_extractor_loaded = (
        feature_extractor_loaded.to(
            load_device
        )
    )

    feature_extractor_loaded.eval()

    classifier_loaded = nn.Linear(
        checkpoint[
            "feature_dimension"
        ],
        checkpoint[
            "num_classes"
        ],
    )

    classifier_loaded.load_state_dict(
        checkpoint[
            "classifier_head_state_dict"
        ]
    )

    classifier_loaded = (
        classifier_loaded.to(
            load_device
        )
    )

    classifier_loaded.eval()

    return {
        "feature_extractor":
            feature_extractor_loaded,

        "classifier":
            classifier_loaded,

        "class_names":
            checkpoint[
                "class_names"
            ],

        "image_size":
            checkpoint[
                "image_size"
            ],

        "imagenet_mean":
            checkpoint[
                "imagenet_mean"
            ],

        "imagenet_std":
            checkpoint[
                "imagenet_std"
            ],

        "device":
            load_device,
    }


# ============================================================
# TASK 7 — SINGLE IMAGE PREDICTION
# ============================================================

def predict_product_image(
    image_path,
    model_path=MODEL_PATH,
):

    loaded = load_product_classifier(
        model_path=model_path
    )

    image_transform = transforms.Compose([
        transforms.Resize(
            (
                loaded[
                    "image_size"
                ],
                loaded[
                    "image_size"
                ],
            )
        ),

        transforms.Grayscale(
            num_output_channels=3
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=loaded[
                "imagenet_mean"
            ],

            std=loaded[
                "imagenet_std"
            ],
        ),
    ])

    image = Image.open(
        image_path
    ).convert("L")

    image_tensor = (
        image_transform(
            image
        )
        .unsqueeze(0)
        .to(
            loaded[
                "device"
            ]
        )
    )

    with torch.no_grad():

        features = (
            loaded[
                "feature_extractor"
            ](
                image_tensor
            )
        )

        features = features.view(
            features.size(0),
            -1,
        )

        logits = (
            loaded[
                "classifier"
            ](
                features
            )
        )

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        predicted_index = int(
            probabilities.argmax(
                dim=1
            ).item()
        )

        confidence = float(
            probabilities[
                0,
                predicted_index
            ].item()
        )

    return {
        "predicted_class":
            loaded[
                "class_names"
            ][
                predicted_index
            ],

        "confidence":
            confidence,
    }


# ============================================================
# TASK 7 — LOAD TEST
# ============================================================

loaded_model = (
    load_product_classifier()
)

print(
    "\nMODEL LOAD TEST"
)

print(
    "Model loaded successfully."
)

print(
    "Feature extractor loaded:",
    loaded_model[
        "feature_extractor"
    ] is not None,
)

print(
    "Classifier head loaded:",
    loaded_model[
        "classifier"
    ] is not None,
)


# ============================================================
# TASK 8 — EXPORT 5 REAL TEST IMAGES
# ============================================================

raw_test_dataset = (
    datasets.FashionMNIST(
        root="data",
        train=False,
        download=False,
    )
)

selected_classes = [
    0,
    1,
    2,
    3,
    7,
]

file_class_names = [
    "T-shirt_top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle_boot",
]

exported_files = []


for target_class in (
    selected_classes
):

    for index in range(
        len(
            raw_test_dataset
        )
    ):

        if int(
            raw_test_dataset.targets[
                index
            ]
        ) == target_class:

            raw_image, _ = (
                raw_test_dataset[
                    index
                ]
            )

            output_path = os.path.join(
                SAMPLE_DIR,
                (
                    f"{index:05d}_"
                    f"{file_class_names[target_class]}.png"
                ),
            )

            raw_image.save(
                output_path
            )

            exported_files.append(
                output_path
            )

            break


print(
    "\nSAMPLE IMAGE EXPORT"
)

print(
    "Directory:",
    SAMPLE_DIR,
)

print(
    "Exported:",
    len(exported_files),
)

for path in exported_files:
    print(path)


# ============================================================
# TASK 8 — EXPORT VERIFICATION
# ============================================================

print(
    "\nVERIFICATION"
)

print(
    "All files are PNG:",
    all(
        path.lower().endswith(
            ".png"
        )
        for path in exported_files
    ),
)

print(
    "All files exist:",
    all(
        os.path.isfile(path)
        for path in exported_files
    ),
)


# ============================================================
# SUBMISSION ARTIFACT — CONFUSION MATRIX CSV
# ============================================================

confusion_matrix_df = (
    pd.DataFrame(
        cm,
        index=CLASS_NAMES,
        columns=CLASS_NAMES,
    )
)

confusion_matrix_df.to_csv(
    CONFUSION_MATRIX_PATH
)

print(
    "\nCONFUSION MATRIX SAVED"
)

print(
    "Location:",
    CONFUSION_MATRIX_PATH,
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print(
    "\nFINAL SUMMARY"
)

print(
    "Training samples:",
    len(train_indices),
)

print(
    "Validation samples:",
    len(val_indices),
)

print(
    "Test samples:",
    len(test_dataset),
)

print(
    "Validation accuracy:",
    round(
        feature_extraction_val_accuracy,
        4,
    ),
)

print(
    "Test accuracy:",
    round(
        test_accuracy,
        4,
    ),
)

print(
    "Fine-tuning required:",
    feature_extraction_val_accuracy
    < 0.80,
)

print(
    "Model:",
    MODEL_PATH,
)

print(
    "Confusion matrix:",
    CONFUSION_MATRIX_PATH,
)

print(
    "Sample images:",
    SAMPLE_DIR,
)
