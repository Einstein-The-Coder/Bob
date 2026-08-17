import torch
import torch.nn as nn

class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        
        # 1. Convolutional Layers (The Visual Processor)
        # Input: 14 channels (12 layers for pieces, 2 for metadata)
        # Output: 128 channels of abstract board features
        self.conv1 = nn.Conv2d(14, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Flattens the 8x8 board grid into a flat vector for the output heads
        self.flatten = nn.Flatten()
        
        # 2. The Policy Head (Decides WHAT move to make)
        # Outputs a score for all 4,096 possible move combinations
        self.policy_head = nn.Linear(128 * 8 * 8, 4096) 
        
        # 3. The Value Head (Evaluates WHO is winning)
        # Outputs a single rating between -1 (Black winning) and 1 (White winning)
        self.value_head = nn.Sequential(
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh() # Tanh squeezes the output perfectly between -1 and 1
        )

    def forward(self, x):
        """Defines how data flows through the neural network."""
        # Process the spatial relationships of the pieces on the board
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.flatten(x)
        
        # Split the data into two separate answers
        policy = self.policy_head(x)
        value = self.value_head(x)
        
        return policy, value
