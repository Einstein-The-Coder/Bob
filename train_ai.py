import os
import glob
import random

import torch
import torch.nn as nn
import torch.optim as optim
import chess
import chess.pgn

from torch.utils.data import Dataset, DataLoader

from utils import board_to_tensor, move_to_index
from model import ChessNet


# ============================================================
# SETTINGS
# ============================================================

TRAIN_FOLDER = "train"

MODEL_PATH = "chess_model_weights.pth"

BATCH_SIZE = 128

EPOCHS = 10

LEARNING_RATE = 0.001

WEIGHT_DECAY = 1e-4

VALUE_WEIGHT = 1.0

POLICY_WEIGHT = 1.0

MAX_GAMES = None

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# DATASET
# ============================================================

class ChessDataset(Dataset):

    def __init__(self, positions):

        self.positions = positions


    def __len__(self):

        return len(self.positions)


    def __getitem__(self, index):

        board_tensor, move_index, value_target = (
            self.positions[index]
        )

        return (
            board_tensor,
            torch.tensor(
                move_index,
                dtype=torch.long
            ),
            torch.tensor(
                value_target,
                dtype=torch.float32
            )
        )


# ============================================================
# VALUE TARGET
# ============================================================

def get_value_target(
    game_result,
    side_to_move
):
    """
    Converts the PGN result into a value target.

    The value is ALWAYS from the perspective of
    the player whose turn it is.

    +1 = side to move eventually wins
    -1 = side to move eventually loses
     0 = draw
    """

    if game_result == "1/2-1/2":

        return 0.0


    if game_result == "1-0":

        if side_to_move == chess.WHITE:

            return 1.0

        return -1.0


    if game_result == "0-1":

        if side_to_move == chess.BLACK:

            return 1.0

        return -1.0


    # Unknown result

    return 0.0


# ============================================================
# LOAD PGN DATA
# ============================================================

def load_training_data(folder_path):

    search_path = os.path.join(
        folder_path,
        "*.pgn"
    )

    pgn_files = glob.glob(
        search_path
    )


    if not pgn_files:

        raise FileNotFoundError(
            f"No PGN files found in '{folder_path}'."
        )


    print()
    print("==============================")
    print("       LOADING PGN DATA")
    print("==============================")
    print()

    print(
        "Found",
        len(pgn_files),
        "PGN file(s)."
    )

    print()


    positions = []

    games_loaded = 0

    moves_loaded = 0


    # --------------------------------------------------------
    # Read every PGN
    # --------------------------------------------------------

    for pgn_filename in pgn_files:

        print(
            "Reading:",
            os.path.basename(
                pgn_filename
            )
        )


        with open(
            pgn_filename,
            "r",
            encoding="utf-8",
            errors="ignore"
        ) as pgn_file:


            while True:

                if (
                    MAX_GAMES is not None
                    and
                    games_loaded >= MAX_GAMES
                ):

                    break


                game = chess.pgn.read_game(
                    pgn_file
                )


                if game is None:

                    break


                games_loaded += 1


                result = game.headers.get(
                    "Result",
                    "*"
                )


                board = game.board()


                # ------------------------------------------------
                # Mainline moves
                # ------------------------------------------------

                for move in game.mainline_moves():

                    # Save the position BEFORE the move.

                    board_tensor = (
                        board_to_tensor(
                            board
                        )
                    )


                    move_index = (
                        move_to_index(
                            move
                        )
                    )


                    value_target = (
                        get_value_target(
                            result,
                            board.turn
                        )
                    )


                    positions.append(
                        (
                            board_tensor,
                            move_index,
                            value_target
                        )
                    )


                    moves_loaded += 1


                    # Advance board

                    board.push(move)


        print(
            "  Positions so far:",
            len(positions)
        )


    print()
    print("==============================")
    print("       DATA LOADED")
    print("==============================")
    print()

    print(
        "Games:",
        games_loaded
    )

    print(
        "Positions:",
        moves_loaded
    )

    print()


    return positions


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    policy_loss_fn,
    value_loss_fn
):

    model.train()


    total_loss = 0.0

    total_policy_loss = 0.0

    total_value_loss = 0.0

    correct_moves = 0

    total_positions = 0


    for batch_index, batch in enumerate(loader):

        inputs, target_moves, target_values = batch


        inputs = inputs.to(
            DEVICE,
            non_blocking=True
        )


        target_moves = target_moves.to(
            DEVICE,
            non_blocking=True
        )


        target_values = target_values.to(
            DEVICE,
            non_blocking=True
        )


        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        policy, value = model(
            inputs
        )


        value = value.squeeze(
            1
        )


        # ----------------------------------------------------
        # Losses
        # ----------------------------------------------------

        policy_loss = policy_loss_fn(
            policy,
            target_moves
        )


        value_loss = value_loss_fn(
            value,
            target_values
        )


        loss = (
            POLICY_WEIGHT * policy_loss
            +
            VALUE_WEIGHT * value_loss
        )


        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        optimizer.zero_grad(
            set_to_none=True
        )


        loss.backward()


        # Prevent huge gradients

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )


        optimizer.step()


        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        batch_size = (
            inputs.size(0)
        )


        total_loss += (
            loss.item() * batch_size
        )


        total_policy_loss += (
            policy_loss.item()
            * batch_size
        )


        total_value_loss += (
            value_loss.item()
            * batch_size
        )


        # ----------------------------------------------------
        # Policy accuracy
        # ----------------------------------------------------

        predictions = (
            policy.argmax(
                dim=1
            )
        )


        correct_moves += (
            predictions ==
            target_moves
        ).sum().item()
        


        total_positions += (
            batch_size
        )


        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            batch_index + 1
        ) % 20 == 0:

            current_accuracy = (
                correct_moves /
                max(
                    1,
                    total_positions
                )
            ) * 100


            print(
                f"    Batch "
                f"{batch_index + 1:4d}"
                f" | Loss "
                f"{loss.item():.4f}"
                f" | Accuracy "
                f"{current_accuracy:.2f}%"
            )


    average_loss = (
        total_loss /
        max(
            1,
            total_positions
        )
    )


    average_policy_loss = (
        total_policy_loss /
        max(
            1,
            total_positions
        )
    )


    average_value_loss = (
        total_value_loss /
        max(
            1,
            total_positions
        )
    )


    accuracy = (
        correct_moves /
        max(
            1,
            total_positions
        )
    ) * 100


    return (
        average_loss,
        average_policy_loss,
        average_value_loss,
        accuracy
    )


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================

def train_on_folder(
    folder_path,
    epochs=EPOCHS
):

    print()
    print("==============================")
    print("        CHESS AI TRAINER")
    print("==============================")
    print()

    print(
        "Device:",
        DEVICE
    )

    print()


    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    positions = load_training_data(
        folder_path
    )


    if len(positions) == 0:

        print(
            "No training positions found."
        )

        return


    # --------------------------------------------------------
    # Shuffle data
    # --------------------------------------------------------

    random.shuffle(
        positions
    )


    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = ChessDataset(
        positions
    )


    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    loader = DataLoader(

        dataset,

        batch_size=BATCH_SIZE,

        shuffle=True,

        num_workers=0,

        pin_memory=(
            DEVICE.type == "cuda"
        )
    )


    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = ChessNet()


    model.to(
        DEVICE
    )


    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = optim.AdamW(

        model.parameters(),

        lr=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY
    )


    # --------------------------------------------------------
    # Learning rate scheduler
    # --------------------------------------------------------

    scheduler = optim.lr_scheduler.CosineAnnealingLR(

        optimizer,

        T_max=epochs
    )


    # --------------------------------------------------------
    # Loss functions
    # --------------------------------------------------------

    policy_loss_fn = nn.CrossEntropyLoss()

    value_loss_fn = nn.MSELoss()


    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print()
    print(
        "Training positions:",
        len(dataset)
    )

    print(
        "Batch size:",
        BATCH_SIZE
    )

    print(
        "Epochs:",
        epochs
    )

    print()


    best_loss = float("inf")


    for epoch in range(epochs):

        print()
        print("==============================")
        print(
            f"       EPOCH {epoch + 1}/{epochs}"
        )
        print("==============================")
        print()


        (
            loss,
            policy_loss,
            value_loss,
            accuracy
        ) = train_one_epoch(

            model,

            loader,

            optimizer,

            policy_loss_fn,

            value_loss_fn
        )


        scheduler.step()


        print()

        print(
            f"Epoch {epoch + 1} complete"
        )

        print(
            f"Total loss:   {loss:.4f}"
        )

        print(
            f"Policy loss:  {policy_loss:.4f}"
        )

        print(
            f"Value loss:   {value_loss:.4f}"
        )

        print(
            f"Policy acc:   {accuracy:.2f}%"
        )

        print(
            f"Learning rate:"
            f" {scheduler.get_last_lr()[0]:.7f}"
        )


        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if loss < best_loss:

            best_loss = loss


            torch.save(
                model.state_dict(),
                MODEL_PATH
            )


            print()

            print(
                "Saved new best model:"
            )

            print(
                MODEL_PATH
            )


    print()
    print("==============================")
    print("      TRAINING COMPLETE")
    print("==============================")
    print()

    print(
        "Best loss:",
        best_loss
    )

    print(
        "Model saved to:",
        MODEL_PATH
    )

    print()


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    train_on_folder(
        TRAIN_FOLDER,
        epochs=EPOCHS
    )