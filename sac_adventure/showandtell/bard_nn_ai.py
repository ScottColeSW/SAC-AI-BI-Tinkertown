import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import random
import json

class PolicyNetwork(nn.Module):
    def __init__(self, output_size, hidden_size, input_size):
        super(PolicyNetwork, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

def preprocess_data(level_data, width, height):
    # Flatten the level_data into a 1D array
    flattened_data = [val for row in level_data for val in row]

    # Calculate the input size dynamically
    output_size = len(level_data)

    # Convert the flattened_data to a tensor of appropriate shape and type
    input_tensor = torch.tensor(flattened_data, dtype=torch.float32).unsqueeze(0)

    return input_tensor

def load_level_map(file_path):
    with open(file_path, 'r') as file:
        level_data = json.load(file)
    return level_data

def main():
    i = 1 # First Level
    level_data = load_level_map(f'level{i}.json')

    width = 1
    height = 1

    # Calculate the hidden size and output size dynamically
    hidden_size = 4 #int(width * height)
    output_size = len(level_data)

    input_size = output_size

    print(f'hidden={hidden_size}, out={output_size}, input={input_size}') 

    policy_net = PolicyNetwork(output_size, hidden_size, input_size)

    input_tensor = preprocessed_data = preprocess_data(level_data, width=1, height=1)

    # Forward pass through the policy network
    action_probs = policy_net(input_tensor)

    # Get the action with the highest probability
    action = action_probs.argmax(dim=1)

    print(action)

if __name__ == '__main__':
    main()
