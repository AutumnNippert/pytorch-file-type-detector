import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor

import os

from skimage import io
import numpy as np
import pandas as pd

file_types = [
    '.py', '.c', '.cpp', '.h', '.java', '.js', '.html', '.css', '.xml', '.json',
    '.txt', '.csv', '.md', '.pdf', '.doc', '.xls', '.ppt', '.zip', '.rar', '.tar',
    '.gz', '.7z', '.bmp', '.jpg', '.jpeg', '.png', '.gif', '.svg', '.mp3', '.wav',
    '.mp4', '.avi', '.mov', '.wmv', '.flv', '.mkv', '.aac', '.psd', '.ai', '.eps',
    '.svg', '.ogg', '.ico', '.exe', '.dll', '.obj', '.class', '.jar', '.apk',
    '.pdb', '.ini', '.cfg', '.bat', '.sh', '.ps1', '.sql', '.bak', '.log', '.bak',
    '.docx', '.xlsx', '.pptx', '.odt', '.ods', '.odp', '.dwg', '.dxf', '.max',
    '.tif', '.tiff', '.mov', '.m4a', '.srt', '.sub', '.ass', '.ttf', '.woff',
    '.woff2', '.eot', '.otf', '.swf', '.ttf', '.lnk', '.url', '.dat', '.db',
    '.sqlite', '.sqlite3', '.dbf', '.ps', '.rtf', '.tex', '.chm', '.csv', '.tsv',
    '.yaml', '.yml', '.bak', '.tmp', '.swp', '.ics', '.iso', '.img', '.rpm',
    '.deb', '.pkg'
]


class FileClassificationDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.label_map = {'.py': 0, '.c': 1, '.java': 2}

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, label = self.data.iloc[idx]
        image = io.imread(os.path.join(self.root_dir, img_path))

        image = np.array(image)
        image_tensor = torch.from_numpy(image)
        # The line below is used for color images, but these images are grayscale
        # image_tensor = image_tensor.permute(2, 0, 1) # change from (512, 512, 1) to (1, 512, 512) because it likes channel first

        label_id = self.label_map[label]
        label_id = torch.tensor([label_id])

        return image_tensor.float(), label_id.float()
    
train_data = FileClassificationDataset(csv_file='train_data_dataset.csv', root_dir='train_data', transform=ToTensor())
test_data = FileClassificationDataset(csv_file='test_data_dataset.csv', root_dir='test_data', transform=ToTensor())

# view data
print(f'Train data size: {len(train_data)}')
print(f'Test data size: {len(test_data)}')
for i in range(3):
    image, label = train_data[i]
    print(f'Image shape: {image.shape}')
    print(f'Label: {label}')

batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(256*256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, len(file_types)),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def train_loop():
    model = NeuralNetwork().to(device)
    print(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    def train(dataloader, model, loss_fn, optimizer):
        print('Starting training...')
        size = len(dataloader.dataset)
        model.train() # set model to training mode

        for batch, (X, y) in enumerate(dataloader):
            print(f'batch {batch}...')
            # print(X.shape, y.shape)

            X, y = X.to(device), y.to(device)

            # Compute prediction error
            pred = model(X)
            y = y.view(-1).long()  # Reshape the target tensor if necessary THIS WAS CHAT-GPT
            loss = loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            #if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def test(dataloader, model, loss_fn):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                y = y.view(-1).long()  # Reshape the target tensor if necessary THIS WAS CHAT-GPT
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    epochs = 100
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
    print("Done!")

    torch.save(model.state_dict(), "model.pth")

def eval():
    model = NeuralNetwork().to(device)
    model.load_state_dict(torch.load("model.pth"))
    model.eval()

    with torch.no_grad():
        accuracy = 0
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            y = y.view(-1).long()  # Reshape the target tensor if necessary THIS WAS CHAT-GPT
            # get predicted label
            pred_index = pred.argmax(1)
            accuracy += (pred_index == y).type(torch.float).sum().item()
        
        print(f'Total accuracy: {accuracy/len(test_dataloader.dataset)}')


def predict(filename):
    from PIL import Image
    model = NeuralNetwork().to(device)
    model.load_state_dict(torch.load("model.pth"))

    #process image
    image = io.imread(filename)
    image = Image.fromarray(image)
    image = image.resize((256, 256))
    image = np.array(image)
    image_tensor = torch.from_numpy(image)
    # image_tensor = image_tensor.permute(2, 0, 1) # change from (512, 512, 1) to (1, 512, 512) because it likes channel first
    image_tensor = image_tensor.unsqueeze(0) # add batch dimension
    image_tensor = image_tensor.float() # convert to float

    # get prediction
    model.eval()
    X, y = image_tensor, torch.tensor([0])
    with torch.no_grad():
        X, y = X.to(device), y.to(device)
        pred = model(X)
        
    # get index with highest probability
    pred_index = pred.argmax(1)
    # get file type from index
    pred_index = file_types[pred_index]

    # OUTPUT
    print(f'Input: {filename}')
    print(f'Predicted type: {pred_index}')

    # for each in pred, print the probability out of 100 for each file type
    probabilities = torch.nn.functional.softmax(pred, dim=1)
    # to dict with index
    probabilities = {i: probabilities[0][i].item() for i in range(len(file_types))}
    # sort by value
    probabilities = {k: v for k, v in sorted(probabilities.items(), key=lambda item: item[1], reverse=True)}

    print('Probabilities:')
    for k, v in probabilities.items():
        print(f'{file_types[k].ljust(7)}: {v*100:>6.2f}%')


if __name__ == '__main__':
    train_loop()
    predict('test_data/2015-03-23b.c.png')
    eval()