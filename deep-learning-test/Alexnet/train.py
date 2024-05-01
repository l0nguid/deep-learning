import os
import json

import torch
import torch.nn as nn
from torchvision import transforms, datasets
from tqdm import tqdm
import torch.optim as optim

from model import AlexNet

def setup_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device.")
    return device

def setup_data_transforms():
    return {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ]),
        "val": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
    }

def main():
    device = setup_device()
    data_transforms = setup_data_transforms()
    data_root = os.path.abspath(os.path.join(os.getcwd(), "../data_set/flower_data"))
    assert os.path.exists(data_root), f"Data path {data_root} does not exist."

    train_dataset = datasets.ImageFolder(root=os.path.join(data_root, "train"),
                                         transform=data_transforms["train"])
    train_num = len(train_dataset)

    flower_list = train_dataset.class_to_idx
    cla_dict = {val: key for key, val in flower_list.items()}
    with open('class_indices.json', 'w') as json_file:
        json.dump(cla_dict, json_file, indent=4)

    batch_size = 32
    nw = min(os.cpu_count(), batch_size if batch_size > 1 else 0, 8)
    print(f'Using {nw} dataloader workers.')

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(data_root, "val"),
                                            transform=data_transforms["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=4, shuffle=False,
                                                  num_workers=nw)

    print(f"using {train_num} images for training, {val_num} images for validation.")

    net = AlexNet(num_classes=5, init_weights=True).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0002)

    epochs = 30
    save_path = './AlexNet.pth'
    best_acc = 0.0
    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_bar.set_description(f"train epoch[{epoch + 1}/{epochs}] loss:{running_loss/(step + 1):.3f}")

        net.eval()
        acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        val_accuracy = acc / val_num
        print(f'[epoch {epoch + 1}] train_loss: {running_loss / len(train_loader):.3f}  val_accuracy: {val_accuracy:.3f}')

        if val_accuracy > best_acc:
            best_acc = val_accuracy
            torch.save(net.state_dict(), save_path)

    print('Finished Training')

if __name__ == '__main__':
    main()
