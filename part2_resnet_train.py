import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import wandb
from tqdm import tqdm
import os

import eval_cifar100
import eval_ood

# ------------------------- CONFIG ---------------------------------
CONFIG = {
    "model": "ResNet18_Part2",
    "batch_size": 128,
    "learning_rate": 0.1,
    "epochs": 30,
    "num_workers": 4,  # set to 0 if issues persist on Windows
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "data_dir": "./data",
    "ood_dir": "./data/ood-test",
    "wandb_project": "sp25-ds542-challenge",
    "seed": 42
}

# ------------------------- TRANSFORMS ------------------------------
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408),
                         (0.2675, 0.2565, 0.2761)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408),
                         (0.2675, 0.2565, 0.2761)),
])

# ------------------------- TRAIN & VAL FUNCTIONS -------------------
def train(epoch, model, trainloader, optimizer, criterion, CONFIG):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    loop = tqdm(trainloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]")

    for inputs, labels in loop:
        inputs, labels = inputs.to(CONFIG["device"]), labels.to(CONFIG["device"])
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        loop.set_postfix(loss=running_loss / (total / CONFIG["batch_size"]),
                         acc=100. * correct / total)

    return running_loss / len(trainloader), 100. * correct / total

def validate(model, valloader, criterion, CONFIG):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        loop = tqdm(valloader, desc="[Validate]")
        for inputs, labels in loop:
            inputs, labels = inputs.to(CONFIG["device"]), labels.to(CONFIG["device"])
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            loop.set_postfix(loss=running_loss / (total / CONFIG["batch_size"]),
                             acc=100. * correct / total)

    return running_loss / len(valloader), 100. * correct / total

# ------------------------- MAIN -------------------------------------
if __name__ == "__main__":
    # Set random seed
    torch.manual_seed(CONFIG["seed"])

    # ------------------------- DATA LOADING ----------------------------
    trainset = torchvision.datasets.CIFAR100(root=CONFIG["data_dir"], train=True,
                                             download=True, transform=transform_train)
    train_size = int(0.8 * len(trainset))
    val_size = len(trainset) - train_size
    trainset, valset = torch.utils.data.random_split(trainset, [train_size, val_size])

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=CONFIG["batch_size"],
                                              shuffle=True, num_workers=CONFIG["num_workers"])
    valloader = torch.utils.data.DataLoader(valset, batch_size=CONFIG["batch_size"],
                                            shuffle=False, num_workers=CONFIG["num_workers"])

    testset = torchvision.datasets.CIFAR100(root=CONFIG["data_dir"], train=False,
                                            download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=CONFIG["batch_size"],
                                             shuffle=False, num_workers=CONFIG["num_workers"])

    # ------------------------- MODEL SETUP -----------------------------
    model = models.resnet18(num_classes=100).to(CONFIG["device"])

    # ------------------------- TRAINING SETUP --------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=CONFIG["learning_rate"],
                          momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])

    wandb.init(project=CONFIG["wandb_project"], name=CONFIG["model"], config=CONFIG)
    wandb.watch(model)

    # ------------------------- TRAINING LOOP ---------------------------
    best_val_acc = 0.0
    for epoch in range(CONFIG["epochs"]):
        train_loss, train_acc = train(epoch, model, trainloader, optimizer, criterion, CONFIG)
        val_loss, val_acc = validate(model, valloader, criterion, CONFIG)
        scheduler.step()

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"]
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")

    wandb.finish()

    # ------------------------- EVALUATION & SUBMISSION -----------------
    predictions, clean_accuracy = eval_cifar100.evaluate_cifar100_test(model, testloader, CONFIG["device"])
    print(f"Clean CIFAR-100 Test Accuracy: {clean_accuracy:.2f}%")

    all_predictions = eval_ood.evaluate_ood_test(model, CONFIG)
    submission_df_ood = eval_ood.create_ood_df(all_predictions)
    submission_df_ood.to_csv("submission_part_2.csv", index=False)
    print("submission_part_2.csv created.")
