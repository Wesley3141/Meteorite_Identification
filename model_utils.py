import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

def create_model():
    """
    Creates a ResNet-18 model for binary classification.
    """
    model = models.resnet18(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)  # Binary classification: meteorite vs non-meteorite
    return model

def train_model(model, train_loader, num_epochs=8):
    """
    Trains the model.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

def export_to_onnx(model, output_path='meteorite_model.onnx'):
    """
    Exports the trained model to ONNX format.
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