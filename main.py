import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import onnx
import onnxruntime as ort
from PIL import Image
import numpy as np

# Step 1: Data Preparation
def prepare_data(data_dir):
    """
    Prepares the dataset for training and testing.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, test_loader

# Step 2: Model Training
def train_model(train_loader, num_epochs=8):
    """
    Trains a ResNet-18 model for meteorite classification.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.resnet18(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)  # Binary classification: meteorite vs non-meteorite
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

    return model

# Step 3: Export Model to ONNX
def export_to_onnx(model, output_path='meteorite_model.onnx'):
    """
    Exports the trained PyTorch model to ONNX format.
    """
    dummy_input = torch.randn(1, 3, 224, 224).to(next(model.parameters()).device)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['input'],
        output_names=['output'],
        opset_version=11
    )
    print(f"Model exported to {output_path}")

# Step 4: Inference with ONNX Runtime
def run_inference(onnx_model_path, image_path, labels):
    """
    Runs inference on a test image using the ONNX model.
    """
    session = ort.InferenceSession(onnx_model_path)
    input_name = session.get_inputs()[0].name

    # Preprocess the image
    img = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = preprocess(img).unsqueeze(0).numpy()

    # Run inference
    outputs = session.run(None, {input_name: img_tensor})
    predicted_idx = np.argmax(outputs[0])

    print(f"Predicted Class: {labels[predicted_idx]}")

# Main Workflow
if __name__ == "__main__":
    # Paths
    data_dir = "data/meteorite_nometeorite"
    onnx_model_path = "meteorite_model.onnx"
    test_image_path = "data/meteorite_nometeorite/test/meteorite/MeteoriteImage (1).jpg"
    labels = ['Non-Meteorite', 'Meteorite']

    # Step 1: Prepare Data
    train_loader, test_loader = prepare_data(data_dir)

    # Step 2: Train Model
    model = train_model(train_loader)

    # Step 3: Export Model to ONNX
    export_to_onnx(model, onnx_model_path)

    # Step 4: Run Inference
    run_inference(onnx_model_path, test_image_path, labels)