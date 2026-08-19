import torch
import torch.nn as nn


# ============================================================
# RESIDUAL BLOCK
# ============================================================

class ResidualBlock(nn.Module):
    """
    A residual block helps the network learn deeper chess
    patterns while preserving information from earlier layers.
    """

    def __init__(self, channels: int):
        super().__init__()

        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            bias=False
        )

        self.bn2 = nn.BatchNorm2d(channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        # Residual connection
        x = x + identity

        x = self.relu(x)

        return x


# ============================================================
# CHESS NETWORK
# ============================================================

class ChessNet(nn.Module):
    """
    Chess neural network.

    Input:
        18 x 8 x 8

    Output:
        policy: 4096 move scores
        value:  -1 to +1

    The 18 input channels are:

        0   White pawns
        1   White knights
        2   White bishops
        3   White rooks
        4   White queens
        5   White king

        6   Black pawns
        7   Black knights
        8   Black bishops
        9   Black rooks
        10  Black queens
        11  Black king

        12  Side to move

        13  White kingside castling
        14  White queenside castling
        15  Black kingside castling
        16  Black queenside castling

        17  En-passant square
    """

    def __init__(
        self,
        input_channels=18,
        channels=128,
        num_residual_blocks=6,
        policy_size=4096
    ):

        super().__init__()


        # ====================================================
        # INPUT LAYER
        # ====================================================

        self.input_layer = nn.Sequential(

            nn.Conv2d(
                input_channels,
                channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(channels),

            nn.ReLU(inplace=True)
        )


        # ====================================================
        # RESIDUAL TOWER
        # ====================================================

        self.residual_tower = nn.Sequential(

            *[
                ResidualBlock(channels)
                for _ in range(num_residual_blocks)
            ]
        )


        # ====================================================
        # POLICY HEAD
        # ====================================================

        self.policy_conv = nn.Sequential(

            nn.Conv2d(
                channels,
                32,
                kernel_size=1,
                bias=False
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(inplace=True)
        )


        self.policy_head = nn.Linear(
            32 * 8 * 8,
            policy_size
        )


        # ====================================================
        # VALUE HEAD
        # ====================================================

        self.value_conv = nn.Sequential(

            nn.Conv2d(
                channels,
                32,
                kernel_size=1,
                bias=False
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(inplace=True)
        )


        self.value_fc1 = nn.Linear(
            32 * 8 * 8,
            256
        )

        self.value_fc2 = nn.Linear(
            256,
            1
        )

        self.value_relu = nn.ReLU(
            inplace=True
        )

        self.value_tanh = nn.Tanh()


    # ========================================================
    # FORWARD
    # ========================================================

    def forward(self, x):

        # ----------------------------------------------------
        # Validate input
        # ----------------------------------------------------

        if x.ndim != 4:

            raise ValueError(
                "Expected input shape "
                "(batch, 18, 8, 8), "
                f"got {tuple(x.shape)}"
            )


        if x.shape[1] != 18:

            raise ValueError(
                "ChessNet expects 18 input channels, "
                f"got {x.shape[1]}"
            )


        if x.shape[2] != 8 or x.shape[3] != 8:

            raise ValueError(
                "ChessNet expects an 8x8 board, "
                f"got {x.shape[2]}x{x.shape[3]}"
            )


        # ====================================================
        # INPUT
        # ====================================================

        x = self.input_layer(x)


        # ====================================================
        # RESIDUAL NETWORK
        # ====================================================

        x = self.residual_tower(x)


        # ====================================================
        # POLICY
        # ====================================================

        policy = self.policy_conv(x)

        policy = policy.flatten(
            start_dim=1
        )

        policy = self.policy_head(
            policy
        )


        # ====================================================
        # VALUE
        # ====================================================

        value = self.value_conv(x)

        value = value.flatten(
            start_dim=1
        )

        value = self.value_fc1(
            value
        )

        value = self.value_relu(
            value
        )

        value = self.value_fc2(
            value
        )

        value = self.value_tanh(
            value
        )


        return policy, value


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print()
    print("==============================")
    print("       TESTING CHESSNET")
    print("==============================")
    print()


    # Create model

    model = ChessNet()

    print(
        "Model created successfully."
    )


    # Fake chess board batch:
    #
    # batch = 2
    # channels = 18
    # board = 8x8

    test_input = torch.zeros(
        2,
        18,
        8,
        8
    )


    print(
        "Input shape:",
        test_input.shape
    )


    # Forward pass

    with torch.no_grad():

        policy, value = model(
            test_input
        )


    print(
        "Policy shape:",
        policy.shape
    )

    print(
        "Value shape:",
        value.shape
    )


    # Verify output sizes

    assert policy.shape == (
        2,
        4096
    )

    assert value.shape == (
        2,
        1
    )


    # Verify value range

    assert torch.all(
        value >= -1
    )

    assert torch.all(
        value <= 1
    )


    print()
    print("Policy output: 4096")
    print("Value output: 1")
    print("Value range: -1 to +1")
    print()
    print("==============================")
    print("        TEST PASSED")
    print("==============================")
    print()