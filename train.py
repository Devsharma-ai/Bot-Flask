# import numpy as np
# import random
# import json
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from nltk_utils import bag_of_words, tokenize, stem
# from model import NeuralNet

# with open('intents.json', 'r') as f:
#     intents = json.load(f)

# all_words = []
# tags = []
# xy = []
# # loop through each sentence in our intents patterns
# for intent in intents['intents']:
#     tag = intent['tag']
#     # add to tag list
#     tags.append(tag)
#     for pattern in intent['patterns']:
#         # tokenize each word in the sentence
#         w = tokenize(pattern)
#         # add to our words list
#         all_words.extend(w)
#         # add to xy pair
#         xy.append((w, tag))

# # stem and lower each word
# ignore_words = ['?', '.', '!']
# all_words = [stem(w) for w in all_words if w not in ignore_words]
# # remove duplicates and sort
# all_words = sorted(set(all_words))
# tags = sorted(set(tags))

# print(len(xy), "patterns")
# print(len(tags), "tags:", tags)
# print(len(all_words), "unique stemmed words:", all_words)

# # create training data
# X_train = []
# y_train = []
# for (pattern_sentence, tag) in xy:
#     # X: bag of words for each pattern_sentence
#     bag = bag_of_words(pattern_sentence, all_words)
#     X_train.append(bag)
#     # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
#     label = tags.index(tag)
#     y_train.append(label)

# X_train = np.array(X_train)
# y_train = np.array(y_train)

# # Hyper-parameters
# num_epochs = 1000
# batch_size = 4
# learning_rate = 0.001
# input_size = len(X_train[0])
# hidden_size = 8
# output_size = len(tags)
# print(input_size, output_size)


# class ChatDataset(Dataset):

#     def __init__(self):
#         self.n_samples = len(X_train)
#         self.x_data = X_train
#         self.y_data = y_train

#     # support indexing such that dataset[i] can be used to get i-th sample
#     def __getitem__(self, index):
#         return self.x_data[index], self.y_data[index]

#     # we can call len(dataset) to return the size
#     def __len__(self):
#         return self.n_samples


# dataset = ChatDataset()
# train_loader = DataLoader(dataset=dataset,
#                           batch_size=batch_size,
#                           shuffle=True,
#                           num_workers=0)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model = NeuralNet(input_size, hidden_size, output_size).to(device)

# # Loss and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# # Train the model
# for epoch in range(num_epochs):
#     for (words, labels) in train_loader:
#         words = words.to(device)
#         labels = labels.to(dtype=torch.long).to(device)

#         # Forward pass
#         outputs = model(words)
#         # if y would be one-hot, we must apply
#         # labels = torch.max(labels, 1)[1]
#         loss = criterion(outputs, labels)

#         # Backward and optimize
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#     if (epoch+1) % 100 == 0:
#         print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


# print(f'final loss: {loss.item():.4f}')

# data = {
#     "model_state": model.state_dict(),
#     "input_size": input_size,
#     "hidden_size": hidden_size,
#     "output_size": output_size,
#     "all_words": all_words,
#     "tags": tags
# }

# FILE = "data.pth"
# torch.save(data, FILE)

# print(f'training complete. file saved to {FILE}')

import numpy as np
import random
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from nltk_utils import bag_of_words, tokenize, remove_stopwords, stem
from model import NeuralNet
from torch.nn.utils.rnn import pad_sequence

# Load intents from the JSON file
with open('fixed_file.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

# Loop through each sentence in our intents patterns
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        # Tokenize the sentence and remove stopwords
        tokens = tokenize(pattern)
        tokens = remove_stopwords(tokens)
        all_words.extend(tokens)
        xy.append((tokens, tag))

# Stem and lower each word
ignore_words = ['?', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]

# Remove duplicates and sort
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(f'{len(xy)} patterns')
print(f'{len(tags)} tags:', tags)
print(f'{len(all_words)} unique stemmed words:', all_words)

# Create training data
X_train = []
y_train = []

for (pattern_sentence, tag) in xy:
    # X: bag of words for each pattern_sentence
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)  # Convert list of arrays to a single numpy array
y_train = np.array(y_train)  # Convert list to numpy array

X_train = torch.tensor(X_train, dtype=torch.float32)  # Convert to tensor
y_train = torch.tensor(y_train, dtype=torch.long)  # Convert to tensor

# Hyper-parameters
num_epochs = 3000  # Increased epochs
batch_size = 4  # Increased batch size
learning_rate = 0.0001  # Decreased learning rate
input_size = len(X_train[0])
hidden_size = 128  # Increased hidden size
output_size = len(tags)

# Define Dataset and DataLoader


class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples


# Define LSTM Model with Bidirectional LSTM
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size

        # Bidirectional LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True,
                            num_layers=2, dropout=0.2, bidirectional=True)

        # Fully connected layer
        # Multiply by 2 for bidirectional
        self.fc = nn.Linear(hidden_size * 2, output_size)

        # Softmax for output classification
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Add a sequence length dimension (batch_size, seq_len, input_size)
        x = x.unsqueeze(1)
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]  # Last hidden state from LSTM
        out = self.fc(last_out)
        out = self.softmax(out)
        return out


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model initialization
model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=1e-5)  # L2 regularization

# DataLoader for batching
dataset = ChatDataset()
train_loader = DataLoader(
    dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# Train the model
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)

        # Forward pass
        outputs = model(words)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Early stopping condition (based on validation loss or other criteria) can be added here

print(f'Final loss: {loss.item():.4f}')

# Save the model state
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'Training complete. File saved to {FILE}')
