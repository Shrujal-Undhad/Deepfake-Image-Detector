import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from ultralytics import YOLO
from efficientnet_pytorch import EfficientNet
from tqdm import tqdm
from torchvision import transforms
import multiprocessing
import sys

def main():
    # Check device
    device = torch.device('cuda')
    print('Using device:', device)

    # Load YOLOv8
    yolo_model = YOLO('models/yolov8n.pt').to(device)

    # Face Detection using YOLO
    def detect_faces_yolo(img_path, model):
        img = cv2.imread(img_path)
        results = model(img)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        faces = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            face = img[y1:y2, x1:x2]
            if face.size > 0:
                faces.append(cv2.resize(face, (256, 256)))
        return faces

    # Load images from dataset folder
    def load_image_paths(folder, label, max_samples=2000):
        paths, labels = [], []
        count = 0
        for file in os.listdir(folder):
            if count >= max_samples:
                break
            path = os.path.join(folder, file)
            if os.path.isfile(path):
                paths.append(path)
                labels.append(label)
                count += 1
        return paths, labels

    # Define transforms including normalization and augmentation
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  std=[0.229, 0.224, 0.225]),
    ])

    # DeepfakeDataset class
    class DeepfakeDataset(Dataset):
        def __init__(self, paths, labels, model, transform=None, extract_features=False):
            self.paths = paths
            self.labels = labels
            self.model = model
            self.size = transform
            self.transform = transform
            self.extract_features = extract_features

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx):
            img = cv2.imread(self.paths[idx])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
            label = torch.tensor(self.labels[idx], dtype=torch.float32)

            if self.extract_features:
                # Extract face using YOLO and resize
                faces = detect_faces_yolo(self.paths[idx], self.model)
                if len(faces) > 0:
                    img = faces[0]  # Use the first detected face

            if self.transform:
                img = self.transform(img)

            return img, label

    # Load dataset
    real_paths, real_labels = load_image_paths('data/train/real', 0, max_samples=2000)
    fake_paths, fake_labels = load_image_paths('data/train/fake', 1, max_samples=2000)

    all_paths = real_paths + fake_paths
    all_labels = real_labels + fake_labels

    X_train, X_val, y_train, y_val = train_test_split(all_paths, all_labels, test_size=0.2, random_state=42)

    print(f"Total images: {len(all_paths)}")
    print(f"Training images: {len(X_train)}")
    print(f"Validation images: {len(X_val)}")

    # Create dataset and dataloaders with transforms
    train_ds = DeepfakeDataset(X_train, y_train, yolo_model, transform=train_transform, extract_features=True)
    val_ds = DeepfakeDataset(X_val, y_val, yolo_model, transform=val_transform, extract_features=True)

    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=32, num_workers=0, pin_memory=True)

    # EfficientNet model
    model = EfficientNet.from_pretrained('efficientnet-b0')
    model._fc = nn.Sequential(nn.Linear(model._fc.in_features, 1), nn.Sigmoid())
    model = model.to(device)

    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    # Training EfficientNet with validation and metrics
    scaler = GradScaler()
    epochs = 10
    processed_images = 0

    def calculate_accuracy(preds, labels):
        preds = (preds > 0.5).float()
        correct = (preds == labels).sum().item()
        return correct / labels.size(0)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_acc = 0.0
        processed_images = 0
        total_batches = len(train_dl)
        print(f"\nEpoch {epoch + 1}/{epochs}")

        for batch_idx, (xb, yb) in enumerate(train_dl):
            xb, yb = xb.to(device), yb.to(torch.float32).unsqueeze(1).to(device)
            optimizer.zero_grad()
            with autocast(device_type='cuda'):
                preds = model(xb)
                loss = loss_fn(preds, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_size = xb.size(0)
            processed_images += batch_size
            total_loss += loss.item()
            total_acc += calculate_accuracy(preds, yb) * batch_size

            print(f"  Batch {batch_idx+1}/{total_batches} - Loss: {loss.item():.4f} - Acc: {calculate_accuracy(preds, yb):.4f} - Images: {processed_images}")

        avg_loss = total_loss / total_batches
        avg_acc = total_acc / len(train_ds)
        print(f"Epoch {epoch + 1} training complete. Avg Loss: {avg_loss:.4f}, Avg Acc: {avg_acc:.4f}")

        # Validation loop
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        val_batches = len(val_dl)
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(torch.float32).unsqueeze(1).to(device)
                preds = model(xb)
                loss = loss_fn(preds, yb)
                val_loss += loss.item()
                val_acc += calculate_accuracy(preds, yb) * xb.size(0)

        avg_val_loss = val_loss / val_batches
        avg_val_acc = val_acc / len(val_ds)
        print(f"Epoch {epoch + 1} validation complete. Avg Loss: {avg_val_loss:.4f}, Avg Acc: {avg_val_acc:.4f}")

        scheduler.step(avg_val_loss)

    torch.save(model.state_dict(), 'models/efficientnet_model.pt')
    print(f"\n✅ Training complete. Total images processed: {processed_images}")

    # Extract features for XGBoost
    def extract_deep_features(paths, model, device, yolo_model, size=(256, 256)):
        model.eval()
        features = []

        with torch.no_grad():
            for path in tqdm(paths, desc="Extracting deep features"):
                img = cv2.imread(path)
                results = yolo_model(img)
                boxes = results[0].boxes.xyxy.cpu().numpy()

                if len(boxes) > 0:
                    x1, y1, x2, y2 = map(int, boxes[0])
                    face = img[y1:y2, x1:x2]
                    if face.size > 0:
                        img = cv2.resize(face, size)
                    else:
                        img = cv2.resize(img, size)
                else:
                    img = cv2.resize(img, size)

                img = img[:, :, ::-1].copy()  # BGR to RGB
                img_tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0

                feat = model.extract_features(img_tensor)
                feat = torch.nn.functional.adaptive_avg_pool2d(feat, (1, 1))
                feat = feat.view(-1).cpu().numpy()
                features.append(feat)

        return np.array(features)

    # Extract features for training and testing
    X_train_feats = extract_deep_features(X_train, model, device, yolo_model)
    X_val_feats = extract_deep_features(X_val, model, device, yolo_model)

    # Train XGBoost
    xgb = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        tree_method='gpu_hist' if torch.cuda.is_available() else 'auto'
    )

    xgb.fit(X_train_feats, y_train,
            eval_set=[(X_val_feats, y_val)],
            early_stopping_rounds=10,
            verbose=True)

    preds = xgb.predict(X_val_feats)
    print("XGBoost Accuracy:", accuracy_score(y_val, preds))
    xgb.save_model('models/xgboost_model.json')

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
