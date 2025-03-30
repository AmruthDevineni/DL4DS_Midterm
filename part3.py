import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import wide_resnet50_2
from tqdm import tqdm
import wandb
import os
import numpy as np

import eval_cifar100
import eval_ood

# ---------------------- CONFIG ----------------------
CONFIG = {
    "model": "WideResNet50_2_Top5",
    "batch_size": 128,
    "learning_rate": 0.01,
    "epochs": 50,
    "num_workers": 4,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "data_dir": "./data",
    "ood_dir": "./data/ood-test",
    "wandb_project": "sp25-ds542-challenge",
    "seed": 42,
    "label_smoothing": 0.1,
    "ema_decay": 0.999
}

# ---------------------- DATA AUGMENTATION ----------------------
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
    transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408),
                         (0.2675, 0.2565, 0.2761))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408),
                         (0.2675, 0.2565, 0.2761))
])

# ---------------------- EMA CLASS ----------------------
class EMA:
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {name: param.clone().detach() for name, param in model.named_parameters() if param.requires_grad}

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = self.decay * self.shadow[name].data + (1.0 - self.decay) * param.data

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

# ---------------------- TRAINING ----------------------
def train(epoch, model, loader, optimizer, criterion, ema):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    loop = tqdm(loader, desc=f"Epoch {epoch+1} [Train]")
    for inputs, labels in loop:
        inputs, labels = inputs.to(CONFIG["device"]), labels.to(CONFIG["device"])
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        ema.update()

        total_loss += loss.item()
        _, preds = outputs.max(1)
        total += labels.size(0)
        correct += preds.eq(labels).sum().item()

        loop.set_postfix(loss=total_loss/(total/CONFIG["batch_size"]), acc=100.*correct/total)

    return total_loss / len(loader), 100. * correct / total

# ---------------------- VALIDATION ----------------------
def validate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        loop = tqdm(loader, desc="[Validate]")
        for inputs, labels in loop:
            inputs, labels = inputs.to(CONFIG["device"]), labels.to(CONFIG["device"])
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

            loop.set_postfix(loss=total_loss/(total/CONFIG["batch_size"]), acc=100.*correct/total)

    return total_loss / len(loader), 100. * correct / total

# ---------------------- MAIN ----------------------
if __name__ == "__main__":
    torch.manual_seed(CONFIG["seed"])

    trainset = torchvision.datasets.CIFAR100(root=CONFIG["data_dir"], train=True, download=True, transform=transform_train)
    train_size = int(0.8 * len(trainset))
    val_size = len(trainset) - train_size
    trainset, valset = torch.utils.data.random_split(trainset, [train_size, val_size])

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"])
    valloader = torch.utils.data.DataLoader(valset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])
    testloader = torch.utils.data.DataLoader(torchvision.datasets.CIFAR100(root=CONFIG["data_dir"], train=False, download=True, transform=transform_test),
                                             batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])

    model = wide_resnet50_2(weights='DEFAULT')
    model.fc = nn.Linear(model.fc.in_features, 100)
    model = model.to(CONFIG["device"])

    ema = EMA(model, CONFIG["ema_decay"])
    criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG["label_smoothing"])
    optimizer = optim.SGD(model.parameters(), lr=CONFIG["learning_rate"], momentum=0.9, weight_decay=5e-4)

    # Scheduler with warmup
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=5)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"] - 5)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[5])

    wandb.init(project=CONFIG["wandb_project"], name=CONFIG["model"], config=CONFIG)
    wandb.watch(model)

    best_val_acc = 0.0
    for epoch in range(CONFIG["epochs"]):
        train_loss, train_acc = train(epoch, model, trainloader, optimizer, criterion, ema)
        val_loss, val_acc = validate(model, valloader, criterion)
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
            ema.apply_shadow()
            torch.save(model.state_dict(), "best_model.pth")

    wandb.finish()

    # Final Evaluation
    predictions, clean_accuracy = eval_cifar100.evaluate_cifar100_test(model, testloader, CONFIG["device"])
    print(f"Clean CIFAR-100 Test Accuracy: {clean_accuracy:.2f}%")

    all_predictions = eval_ood.evaluate_ood_test(model, CONFIG)
    submission_df = eval_ood.create_ood_df(all_predictions)
    submission_df.to_csv("submission_part_3.csv", index=False)
    print("submission_part_3.csv created.")
