import os
import sys
import json
import logging
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim

from model import vgg

def parse_args():
    parser = argparse.ArgumentParser(description="Train a VGG network on a flower dataset")
    parser.add_argument('--data_dir', default=os.path.join(os.getcwd(), "../../data_set/flower_data"), type=str, help='Directory for dataset')
    parser.add_argument('--epochs', default=30, type=int, help='Number of epochs to train')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size for training')
    parser.add_argument('--learning_rate', default=0.0001, type=float, help='Learning rate')
    parser.add_argument('--model_name', default='vgg16', type=str, help='Model to use')
    parser.add_argument('--num_classes', default=5, type=int, help='Number of classes')
    return parser.parse_args()

def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_dataloaders(data_dir, batch_size):
    data_transforms = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    }

    train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, "train"), transform=data_transforms["train"])
    validate_dataset = datasets.ImageFolder(root=os.path.join(data_dir, "val"), transform=data_transforms["val"])

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=nw)
    validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=batch_size, shuffle=False, num_workers=nw)

    return train_loader, validate_loader, len(train_dataset), len(validate_dataset)

def train_one_epoch(model, device, data_loader, optimizer, loss_function, epoch, epochs):
    model.train()
    running_loss = 0.0
    train_bar = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(train_bar):
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss)
    return running_loss / len(data_loader)

def validate(model, device, data_loader, total):
    model.eval()
    acc = 0.0
    with torch.no_grad():
        val_bar = tqdm(data_loader, file=sys.stdout)
        for val_data in val_bar:
            images, labels = val_data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predict_y = torch.max(outputs, dim=1)[1]
            acc += torch.eq(predict_y, labels).sum().item()
    return acc / total

def main():
    args = parse_args()
    device = get_device()
    logging.info(f"Using {device} device.")
    train_loader, validate_loader, train_num, val_num = get_dataloaders(args.data_dir, args.batch_size)
    logging.info(f"Using {train_num} images for training, {val_num} images for validation.")

    model = vgg(model_name=args.model_name, num_classes=args.num_classes, init_weights=True).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    best_acc = 0.0
    save_path = f'./{args.model_name}Net.pth'
    for epoch in range(args.epochs):
        logging.info(f"Epoch {epoch+1}/{args.epochs}")
        train_loss = train_one_epoch(model, device, train_loader, optimizer, loss_function, epoch, args.epochs)
        val_accuracy = validate(model, device, validate_loader, val_num)
        logging.info(f'[epoch {epoch+1}] train_loss: {train_loss:.3f} val_accuracy: {val_accuracy:.3f}')

        if val_accuracy > best_acc:
            best_acc = val_accuracy
            torch.save(model.state_dict(), save_path)
            logging.info(f"Saved best model weights to {save_path}")

    logging.info('Finished Training')

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()
