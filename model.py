import torch
import torch.nn as nn


class ChessNet(nn.Module):

    def __init__(self):

        super().__init__()


        # ====================================================
        # BOARD PROCESSOR
        # ====================================================

        self.conv1 = nn.Conv2d(
            14,
            64,
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            64,
            64,
            kernel_size=3,
            padding=1
        )

        self.conv3 = nn.Conv2d(
            64,
            128,
            kernel_size=3,
            padding=1
        )


        self.flatten = nn.Flatten()


        # ====================================================
        # POLICY HEAD
        # ====================================================

        self.policy_head = nn.Linear(
            128 * 8 * 8,
            4096
        )


        # ====================================================
        # VALUE HEAD
        # ====================================================

        self.value_head = nn.Sequential(

            nn.Linear(
                128 * 8 * 8,
                256
            ),

            nn.ReLU(),

            nn.Linear(
                256,
                1
            ),

            nn.Tanh()
        )


    # ========================================================
    # FORWARD
    # ========================================================

    def forward(self, x):

        x = torch.relu(
            self.conv1(x)
        )

        x = torch.relu(
            self.conv2(x)
        )

        x = torch.relu(
            self.conv3(x)
        )

        x = self.flatten(x)


        policy = self.policy_head(x)

        value = self.value_head(x)


        return policy, value
