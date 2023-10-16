import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import random
import json

class PolicyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PolicyNetwork, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

def preprocess_data(level_data):
    # Flatten the level_data into a 1D array
    flattened_data = [val for row in level_data for val in row]

    # Convert the flattened_data to a tensor of appropriate shape and type
    input_tensor = torch.tensor(flattened_data, dtype=torch.float32)

    # Reshape the input tensor to have a batch dimension of 1
    input_tensor = input_tensor.unsqueeze(0)

    return input_tensor

def load_level_map(file_path):
    with open(file_path, 'r') as file:
        level_data = json.load(file)
    return level_data

i = 1 # First Level
level_data = load_level_map(f'level{i}.json')

TILE_SIZE = 32
MAP_WIDTH = 10
MAP_HEIGHT = 8


input_size = 80# <size of your input>
hidden_size = 10 # <number of units in the hidden layer>
output_size = 8 #<number of actions in the action space> 

policy_net = PolicyNetwork(input_size, hidden_size, output_size)

preprocessed_data = preprocess_data(level_data)
print(preprocessed_data.shape)  # Output: torch.Size([1, 80])


#preprocessed_data = level_data # preprocess_data(level_data)
#preprocessed_data = preprocess_data(level_data)


# Convert the preprocessed data to a tensor
#input_tensor = torch.tensor(preprocessed_data, dtype=torch.float32)
#input_tensor = sourceTensor.clone().detach().requires_grad_(True)

# Forward pass through the policy network
#action_probs = policy_net(input_tensor)