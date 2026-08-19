import os
import glob

import torch
import torch.nn as nn
import torch.optim as optim
import chess.pgn

from utils import board_to_tensor, move_to_index
from model import ChessNet


TRAIN_FOLDER = "train"

EPOCHS = 3

LEARNING_RATE = 0.001

SAVE_FILE = "chess_model_weights.pth"


def train_on_folder(folder_path, epochs=3):

    print()
    print("==============================")
    print("       CHESS AI TRAINING")
    print("==============================")
    print()


    # ==========================================
    # FIND PGN FILES
    # ==========================================

    pgn_files = glob.glob(
        os.path.join(
            folder_path,
            "*.pgn"
        )
    )

    if not pgn_files:

        print(
            f"No PGN files found in '{folder_path}'."
        )

        print(
            "Put your .pgn files inside the train folder."
        )

        return


    print(
        f"Found {len(pgn_files)} PGN file(s)."
    )


    # ==========================================
    # MODEL
    # ==========================================

    model = ChessNet()

    model.train()


    # ==========================================
    # OPTIMIZER
    # ==========================================

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )


    # ==========================================
    # LOSS
    # ==========================================

    criterion = nn.CrossEntropyLoss()


    # ==========================================
    # TRAINING
    # ==========================================

    for epoch in range(epochs):

        total_loss = 0.0
        moves_processed = 0
        games_processed = 0


        print()
        print(
            f"Epoch {epoch + 1}/{epochs}"
        )


        for pgn_filename in pgn_files:

            print(
                "Processing:",
                os.path.basename(pgn_filename)
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


                        policy, value = model(
                            input_tensor
                        )


                        loss = criterion(
                            policy,
                            target
                        )


                        optimizer.zero_grad()

                        loss.backward()

                        optimizer.step()


                        total_loss += loss.item()

                        moves_processed += 1


                        board.push(move)


        average_loss = (
            total_loss /
            max(1, moves_processed)
        )


        print()
        print(
            f"Games: {games_processed}"
        )

        print(
            f"Moves: {moves_processed}"
        )

        print(
            f"Average loss: {average_loss:.6f}"
        )


        # Save after every epoch

        torch.save(
            model.state_dict(),
            SAVE_FILE
        )


        print(
            f"Saved {SAVE_FILE}"
        )


    print()
    print("==============================")
    print("       TRAINING FINISHED")
    print("==============================")


if __name__ == "__main__":

    train_on_folder(
        TRAIN_FOLDER,
        EPOCHS
    )
