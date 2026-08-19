import os
import glob
import random

import torch
import torch.nn as nn
import torch.optim as optim
import chess
import chess.pgn
import zstandard as zstd

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

POLICY_WEIGHT = 1.0

VALUE_WEIGHT = 1.0

# Set to None to use every game.
MAX_GAMES = None

# Set to None to use every position.
MAX_POSITIONS = None

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

def get_value_target(result, side_to_move):

    """
    Value is from the perspective of the player
    whose turn it is.

    +1 = player to move eventually wins
     0 = draw
    -1 = player to move eventually loses
    """

    if result == "1/2-1/2":
        return 0.0

    if result == "1-0":

        if side_to_move == chess.WHITE:
            return 1.0

        return -1.0

    if result == "0-1":

        if side_to_move == chess.BLACK:
            return 1.0

        return -1.0

    # Unknown result
    return 0.0


# ============================================================
# PGN FILE READER
# ============================================================

def open_pgn_file(filename):

    """
    Opens either:

        .pgn
        .pgn.zst

    Returns a text stream suitable for chess.pgn.read_game().
    """

    if filename.lower().endswith(".zst"):

        zstd_file = open(
            filename,
            "rb"
        )

        decompressor = zstd.ZstdDecompressor()

        stream = decompressor.stream_reader(
            zstd_file
        )

        text_stream = (
            stream
            .read()
            .decode(
                "utf-8",
                errors="ignore"
            )
        )

        return text_stream, zstd_file, stream

    else:

        text_file = open(
            filename,
            "r",
            encoding="utf-8",
            errors="ignore"
        )

        return text_file, None, None


# ============================================================
# STREAM PGN.ZST
# ============================================================

def open_pgn_zst_stream(filename):

    """
    Proper streaming reader for .pgn.zst.

    This avoids extracting the entire compressed
    database onto disk.
    """

    compressed_file = open(
        filename,
        "rb"
    )

    decompressor = zstd.ZstdDecompressor()

    binary_stream = decompressor.stream_reader(
        compressed_file
    )

    text_stream = (
        binary_stream
        # Wrap binary decompression stream as text
    )

    import io

    text_stream = io.TextIOWrapper(
        text_stream,
        encoding="utf-8",
        errors="ignore"
    )

    return (
        text_stream,
        compressed_file,
        binary_stream
    )


# ============================================================
# LOAD ONE PGN FILE
# ============================================================

def process_pgn_file(
    filename,
    positions,
    stats
):

    print()
    print(
        "Processing:",
        os.path.basename(filename)
    )

    is_zst = filename.lower().endswith(
        ".pgn.zst"
    )


    if is_zst:

        pgn_stream, compressed_file, binary_stream = (
            open_pgn_zst_stream(filename)
        )

    else:

        pgn_stream = open(
            filename,
            "r",
            encoding="utf-8",
            errors="ignore"
        )

        compressed_file = None
        binary_stream = None


    try:

        while True:

            # Stop if maximum number of games reached.

            if (
                MAX_GAMES is not None
                and
                stats["games"] >= MAX_GAMES
            ):

                break


            # Stop if maximum number of positions reached.

            if (
                MAX_POSITIONS is not None
                and
                len(positions) >= MAX_POSITIONS
            ):

                break


            game = chess.pgn.read_game(
                pgn_stream
            )


            if game is None:
                break


            stats["games"] += 1


            result = game.headers.get(
                "Result",
                "*"
            )


            board = game.board()


            # ------------------------------------------------
            # Process every position in the game
            # ------------------------------------------------

            for move in game.mainline_moves():

                if (
                    MAX_POSITIONS is not None
                    and
                    len(positions) >= MAX_POSITIONS
                ):

                    break


                # Position BEFORE the move.

                tensor = board_to_tensor(
                    board
                )


                move_index = move_to_index(
                    move
                )


                value_target = get_value_target(
                    result,
                    board.turn
                )


                positions.append(
                    (
                        tensor,
                        move_index,
                        value_target
                    )
                )


                stats["positions"] += 1


                board.push(move)


            # Progress

            if stats["games"] % 100 == 0:

                print(
                    f"  Games: {stats['games']:,}"
                    f" | Positions: "
                    f"{stats['positions']:,}"
                )


    finally:

        pgn_stream.close()

        if binary_stream is not None:
            binary_stream.close()

        if compressed_file is not None:
            compressed_file.close()


# ============================================================
# FIND TRAINING FILES
# ============================================================

def find_training_files(folder):

    pgn_files = glob.glob(
        os.path.join(
            folder,
            "*.pgn"
        )
    )

    zst_files = glob.glob(
        os.path.join(
            folder,
            "*.pgn.zst"
        )
    )

    files = (
        pgn_files +
        zst_files
    )

    files.sort()

    return files


# ============================================================
# LOAD ALL TRAINING DATA
# ============================================================

def load_training_data(folder):

    files = find_training_files(
        folder
    )


    if not files:

        raise FileNotFoundError(
            f"No .pgn or .pgn.zst files found "
            f"in '{folder}'."
        )


    print()
    print("==============================")
    print("       TRAINING DATA")
    print("==============================")
    print()

    print(
        "Found",
        len(files),
        "training file(s)."
    )

    print()


    positions = []

    stats = {
        "games": 0,
        "positions": 0
    }


    for filename in files:

        if (
            MAX_GAMES is not None
            and
            stats["games"] >= MAX_GAMES
        ):
            break


        if (
            MAX_POSITIONS is not None
            and
            len(positions) >= MAX_POSITIONS
        ):
            break


        process_pgn_file(
            filename,
            positions,
            stats
        )


    print()
    print("==============================")
    print("       DATA LOADED")
    print("==============================")
    print()

    print(
        f"Games:     {stats['games']:,}"
    )

    print(
        f"Positions: {stats['positions']:,}"
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


    for batch_number, batch in enumerate(loader):

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
        # Forward pass
        # ----------------------------------------------------

        policy, value = model(
            inputs
        )


        value = value.squeeze(
            1
        )


        # ----------------------------------------------------
        # Loss
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


        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )


        optimizer.step()


        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        batch_size = inputs.size(0)


        total_loss += (
            loss.item()
            * batch_size
        )


        total_policy_loss += (
            policy_loss.item()
            * batch_size
        )


        total_value_loss += (
            value_loss.item()
            * batch_size
        )


        predictions = policy.argmax(
            dim=1
        )


        correct_moves += (
            (predictions == target_moves)
            .sum()
            .item()
        )


        total_positions += batch_size


        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            batch_number + 1
        ) % 50 == 0:

            accuracy = (
                correct_moves
                /
                max(
                    1,
                    total_positions
                )
            ) * 100


            print(
                f"    Batch "
                f"{batch_number + 1:,}"
                f" | Loss "
                f"{loss.item():.4f}"
                f" | Accuracy "
                f"{accuracy:.2f}%"
            )


    average_loss = (
        total_loss
        /
        max(
            1,
            total_positions
        )
    )


    average_policy_loss = (
        total_policy_loss
        /
        max(
            1,
            total_positions
        )
    )


    average_value_loss = (
        total_value_loss
        /
        max(
            1,
            total_positions
        )
    )


    accuracy = (
        correct_moves
        /
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
# MAIN TRAINING
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
    # Load games
    # --------------------------------------------------------

    positions = load_training_data(
        folder_path
    )


    if not positions:

        print(
            "No training positions found."
        )

        return


    # --------------------------------------------------------
    # Shuffle
    # --------------------------------------------------------

    print(
        "Shuffling training positions..."
    )

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
    # Scheduler
    # --------------------------------------------------------

    scheduler = (
        optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs
        )
    )


    # --------------------------------------------------------
    # Loss functions
    # --------------------------------------------------------

    policy_loss_fn = nn.CrossEntropyLoss()

    value_loss_fn = nn.MSELoss()


    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    best_loss = float("inf")


    for epoch in range(epochs):

        print()
        print("==============================")
        print(
            f"        EPOCH {epoch + 1}/{epochs}"
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
            f"Loss:         {loss:.4f}"
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
                "New best model saved:"
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
        "Model:",
        MODEL_PATH
    )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    train_on_folder(
        TRAIN_FOLDER,
        epochs=EPOCHS
    )