import os
import glob

import torch
import torch.nn as nn
import torch.optim as optim
import chess.pgn

from utils import board_to_tensor, move_to_index
from model import ChessNet


# ============================================================
# SETTINGS
# ============================================================

TRAIN_FOLDER = "train"

EPOCHS = 3

LEARNING_RATE = 0.001

SAVE_FILE = "chess_model_weights.pth"


# ============================================================
# TRAIN
# ============================================================

def train_on_folder(
    folder_path,
    epochs=EPOCHS
):

    print()
    print("==============================")
    print("       CHESS AI TRAINING")
    print("==============================")
    print()


    # ========================================================
    # FIND PGN FILES
    # ========================================================

    pgn_files = glob.glob(
        os.path.join(
            folder_path,
            "*.pgn"
        )
    )


    if not pgn_files:

        print(
            f"No .pgn files found in '{folder_path}'."
        )

        return


    print(
        f"Found {len(pgn_files)} PGN file(s)."
    )

    print()


    # ========================================================
    # MODEL
    # ========================================================

    model = ChessNet()

    model.train()


    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )


    # ========================================================
    # POLICY LOSS
    # ========================================================

    policy_loss_function = nn.CrossEntropyLoss()


    # ========================================================
    # EPOCHS
    # ========================================================

    for epoch in range(epochs):

        total_loss = 0.0

        moves_processed = 0

        games_processed = 0


        print(
            f"========== Epoch "
            f"{epoch + 1}/{epochs} =========="
        )


        # ====================================================
        # PGN FILES
        # ====================================================

        for pgn_filename in pgn_files:

            print(
                "Processing:",
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

                    game = chess.pgn.read_game(
                        pgn_file
                    )


                    if game is None:

                        break


                    games_processed += 1

                    board = game.board()


                    # ========================================
                    # MOVES
                    # ========================================

                    for move in game.mainline_moves():

                        input_tensor = (
                            board_to_tensor(board)
                            .unsqueeze(0)
                        )


                        target = torch.tensor(
                            [
                                move_to_index(move)
                            ],
                            dtype=torch.long
                        )


                        # Forward

                        policy, value = model(
                            input_tensor
                        )


                        # Loss

                        loss = policy_loss_function(
                            policy,
                            target
                        )


                        # Backpropagation

                        optimizer.zero_grad()

                        loss.backward()

                        optimizer.step()


                        # Statistics

                        total_loss += (
                            loss.item()
                        )

                        moves_processed += 1


                        # Advance board

                        board.push(move)


        # ====================================================
        # STATISTICS
        # ====================================================

        average_loss = (
            total_loss /
            max(
                1,
                moves_processed
            )
        )


        print()

        print(
            "Games:",
            games_processed
        )

        print(
            "Moves:",
            moves_processed
        )

        print(
            f"Average loss: "
            f"{average_loss:.6f}"
        )


        # ====================================================
        # SAVE
        # ====================================================

        torch.save(
            model.state_dict(),
            SAVE_FILE
        )


        print(
            f"Saved: {SAVE_FILE}"
        )

        print()


    print("==============================")
    print("       TRAINING FINISHED")
    print("==============================")


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    train_on_folder(
        TRAIN_FOLDER,
        epochs=EPOCHS
    )
