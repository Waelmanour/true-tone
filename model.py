import os
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class AudioDataset(Dataset):
    def __init__(self, audio_paths, labels):
        self.audio_paths = audio_paths
        self.labels = labels

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        # Load and preprocess audio
        audio, sr = librosa.load(audio_path, duration=2)
        # Ensure audio length is consistent
        if len(audio) < sr * 2:
            audio = np.pad(audio, (0, sr * 2 - len(audio)))
        else:
            audio = audio[:sr * 2]
        # Extract mel spectrogram features with fixed parameters
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, n_fft=2048, hop_length=512)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        # Normalize with epsilon to avoid division by zero
        eps = 1e-6
        mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + eps)
        return torch.FloatTensor(mel_spec_db), self.labels[idx]

class AudioClassifier(nn.Module):
    def __init__(self):
        super(AudioClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))  # Adaptive pooling to fixed size
        self.fc1 = nn.Linear(64 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.adaptive_pool(x)
        x = x.view(-1, 64 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.clamp(self.fc2(x), min=0.0, max=1.0)  # Ensure outputs are between 0 and 1
        return x

def load_dataset():
    data_types = ['training', 'testing', 'validation']
    datasets = {dtype: {'paths': [], 'labels': []} for dtype in data_types}
    
    # Define all dataset folders
    dataset_folders = ['for-2seconds', 'for-norm', 'for-original', 'for-rerecorded']
    
    total_files = 0
    for dtype in data_types:
        for folder in dataset_folders:
            # Load real audio files
            real_dir = os.path.join('dataset', folder, dtype, 'real')
            if os.path.exists(real_dir):
                files = [f for f in os.listdir(real_dir) if f.endswith('.wav')]
                datasets[dtype]['paths'].extend([os.path.join(real_dir, f) for f in files])
                datasets[dtype]['labels'].extend([0] * len(files))  # 0 for real
                total_files += len(files)
                print(f'Loaded {len(files)} real files from {folder}/{dtype}')
            
            # Load fake audio files
            fake_dir = os.path.join('dataset', folder, dtype, 'fake')
            if os.path.exists(fake_dir):
                files = [f for f in os.listdir(fake_dir) if f.endswith('.wav')]
                datasets[dtype]['paths'].extend([os.path.join(fake_dir, f) for f in files])
                datasets[dtype]['labels'].extend([1] * len(files))  # 1 for fake
                total_files += len(files)
                print(f'Loaded {len(files)} fake files from {folder}/{dtype}')
    
    print(f'\nTotal files loaded: {total_files}')
    for dtype in data_types:
        print(f'{dtype} set size: {len(datasets[dtype]["paths"])}')
    
    return datasets

def train_model(model, train_loader, criterion, optimizer, device, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device).float()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            predicted = (outputs.squeeze() > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if (batch_idx + 1) % 10 == 0:
                print(f'Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss.item():.4f}, Accuracy: {100*correct/total:.2f}%')
        
        epoch_loss = running_loss/len(train_loader)
        epoch_acc = 100 * correct / total
        print(f'\nEpoch {epoch+1} Summary:')
        print(f'Average Loss: {epoch_loss:.4f}')
        print(f'Accuracy: {epoch_acc:.2f}%\n')
    
    return model

def save_model(model, path='model.pth'):
    torch.save(model.state_dict(), path)

def load_trained_model(path='model.pth'):
    model = AudioClassifier()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def predict(model, audio_path, device):
    model.eval()
    audio, sr = librosa.load(audio_path, duration=2)
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / mel_spec_db.std()
    
    with torch.no_grad():
        input_tensor = torch.FloatTensor(mel_spec_db).unsqueeze(0).to(device)
        output = model(input_tensor)
        prediction = output.item()
    
    return prediction

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load and prepare datasets
    datasets = load_dataset()
    
    # Create data loaders for each set
    train_dataset = AudioDataset(datasets['training']['paths'], datasets['training']['labels'])
    test_dataset = AudioDataset(datasets['testing']['paths'], datasets['testing']['labels'])
    val_dataset = AudioDataset(datasets['validation']['paths'], datasets['validation']['labels'])
    
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model and training components
    model = AudioClassifier().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters())
    
    # Train model
    model = train_model(model, train_loader, criterion, optimizer, device)
    
    # Evaluate model on test and validation sets
    model.eval()
    with torch.no_grad():
        for loader_name, loader in [('Test', test_loader), ('Validation', val_loader)]:
            correct = 0
            total = 0
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                predicted = (outputs.squeeze() > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            accuracy = 100 * correct / total
            print(f'{loader_name} Accuracy: {accuracy:.2f}%')
    
    # Save the trained model
    save_model(model)

if __name__ == '__main__':
    main()