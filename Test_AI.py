import os
import shutil

import chess
import chess.engine
import torch

from model import ChessNet
from utils import board_to_tensor, move_to_index


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "chess_model_weights.pth"

# How long Stockfish thinks per move.
#
# Smaller = faster
# Larger = stronger
#
STOCKFISH_TIME = 0.2

# Maximum number of full moves.
MAX_MOVES = 200

# Your neural network plays White.
# Stockfish plays Black.
AI_COLOR = chess.WHITE


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print()
print("==============================")
print("       CHESS AI TEST")
print("==============================")
print()

print(
    "Device:",
    DEVICE
)


# ============================================================
# FIND STOCKFISH
# ============================================================

def find_stockfish():

    # --------------------------------------------------------
    # First check whether "stockfish" is already in PATH.
    # --------------------------------------------------------

    stockfish = shutil.which(
        "stockfish"
    )

    if stockfish is not None:

        return stockfish


    # --------------------------------------------------------
    # Common locations.
    # --------------------------------------------------------

    possible_paths = [

        "./stockfish",

        "./stockfish/stockfish",

        "./stockfish/stockfish-ubuntu-x86-64-avx2",

        "./stockfish/stockfish-ubuntu-x86-64-avx512",

        "/usr/bin/stockfish",

        "/usr/local/bin/stockfish",

        "/usr/games/stockfish"
    ]


    for path in possible_paths:

        if os.path.isfile(path):

            return path


    # --------------------------------------------------------
    # Search current directory.
    # --------------------------------------------------------

    for root, dirs, files in os.walk("."):

        for filename in files:

            if filename == "stockfish":

                path = os.path.join(
                    root,
                    filename
                )

                return path


            if filename.startswith(
                "stockfish-"
            ):

                path = os.path.join(
                    root,
                    filename
                )

                return path


    return None


# ============================================================
# LOAD NEURAL NETWORK
# ============================================================

def load_model():

    print()
    print("Loading neural network...")
    print()

    if not os.path.exists(
        MODEL_PATH
    ):

        raise FileNotFoundError(
            f"Could not find {MODEL_PATH}"
        )


    # --------------------------------------------------------
    # Create the exact same architecture used during training.
    # --------------------------------------------------------

    model = ChessNet()


    # --------------------------------------------------------
    # Load weights.
    # --------------------------------------------------------

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )


    # --------------------------------------------------------
    # Support both:
    #
    # torch.save(model.state_dict(), ...)
    #
    # and:
    #
    # torch.save({
    #     "model_state_dict": ...
    # }, ...)
    # --------------------------------------------------------

    if (
        isinstance(checkpoint, dict)
        and
        "model_state_dict" in checkpoint
    ):

        state_dict = checkpoint[
            "model_state_dict"
        ]

    else:

        state_dict = checkpoint


    model.load_state_dict(
        state_dict
    )


    model.to(
        DEVICE
    )


    model.eval()


    print(
        "Neural network loaded successfully."
    )

    print(
        "Weights:",
        MODEL_PATH
    )

    print()


    return model


# ============================================================
# GET NEURAL NETWORK MOVE
# ============================================================

def get_ai_move(
    model,
    board
):

    # --------------------------------------------------------
    # Get all legal moves.
    # --------------------------------------------------------

    legal_moves = list(
        board.legal_moves
    )


    if not legal_moves:

        return None


    # --------------------------------------------------------
    # Convert board to neural-network input.
    # --------------------------------------------------------

    tensor = board_to_tensor(
        board
    )


    # Add batch dimension.
    #
    # Before:
    #
    #   18 x 8 x 8
    #
    # After:
    #
    #   1 x 18 x 8 x 8

    tensor = tensor.unsqueeze(
        0
    )


    tensor = tensor.to(
        DEVICE
    )


    # --------------------------------------------------------
    # Neural network inference.
    # --------------------------------------------------------

    with torch.no_grad():

        policy, value = model(
            tensor
        )


    # --------------------------------------------------------
    # Policy scores.
    #
    # Shape:
    #
    #   [1, 4096]
    # --------------------------------------------------------

    scores = policy[0]


    # --------------------------------------------------------
    # Find highest-scoring LEGAL move.
    # --------------------------------------------------------

    best_move = None

    best_score = float(
        "-inf"
    )


    for move in legal_moves:

        move_index = move_to_index(
            move
        )


        score = scores[
            move_index
        ].item()


        if score > best_score:

            best_score = score

            best_move = move


    return best_move


# ============================================================
# GET STOCKFISH MOVE
# ============================================================

def get_stockfish_move(
    engine,
    board
):

    result = engine.play(

        board,

        chess.engine.Limit(
            time=STOCKFISH_TIME
        )
    )


    return result.move


# ============================================================
# PRINT BOARD
# ============================================================

def print_board(
    board
):

    print()

    print(
        board
    )

    print()


# ============================================================
# PLAY GAME
# ============================================================

def play_game(
    model,
    stockfish_path
):

    print()
    print("==============================")
    print("    NEURAL AI vs STOCKFISH")
    print("==============================")
    print()

    print(
        "White: Neural Chess AI"
    )

    print(
        "Black: Stockfish"
    )

    print(
        "Stockfish time:",
        STOCKFISH_TIME,
        "seconds/move"
    )

    print()


    # --------------------------------------------------------
    # Start Stockfish.
    # --------------------------------------------------------

    print(
        "Starting Stockfish..."
    )

    engine = chess.engine.SimpleEngine.popen_uci(
        stockfish_path
    )


    print(
        "Stockfish started successfully."
    )

    print()


    # --------------------------------------------------------
    # Create new chess game.
    # --------------------------------------------------------

    board = chess.Board()


    move_count = 0


    try:

        while (
            not board.is_game_over()
            and
            move_count < MAX_MOVES
        ):

            # =================================================
            # NEURAL AI
            # =================================================

            if board.turn == AI_COLOR:

                print(
                    f"Move {move_count + 1}: "
                    "Neural AI thinking..."
                )


                move = get_ai_move(
                    model,
                    board
                )


                if move is None:

                    print(
                        "Neural AI could not find a move."
                    )

                    break


                san = board.san(
                    move
                )


                print(
                    "Neural AI:",
                    san
                )


                board.push(
                    move
                )


            # =================================================
            # STOCKFISH
            # =================================================

            else:

                print(
                    f"Move {move_count + 1}: "
                    "Stockfish thinking..."
                )


                move = get_stockfish_move(
                    engine,
                    board
                )


                if move is None:

                    print(
                        "Stockfish could not find a move."
                    )

                    break


                san = board.san(
                    move
                )


                print(
                    "Stockfish:",
                    san
                )


                board.push(
                    move
                )


            move_count += 1


            # ------------------------------------------------
            # Print board every move.
            # ------------------------------------------------

            print_board(
                board
            )


        # =====================================================
        # GAME RESULT
        # =====================================================

        print()
        print("==============================")
        print("          GAME OVER")
        print("==============================")
        print()


        print_board(
            board
        )


        # -----------------------------------------------------
        # Checkmate
        # -----------------------------------------------------

        if board.is_checkmate():

            # The side whose turn it is has been checkmated.

            winner = not board.turn


            if winner == AI_COLOR:

                print(
                    "RESULT: YOUR AI WON!"
                )

            else:

                print(
                    "RESULT: STOCKFISH WON."
                )


        # -----------------------------------------------------
        # Draw
        # -----------------------------------------------------

        elif board.is_stalemate():

            print(
                "RESULT: DRAW - STALEMATE"
            )


        elif board.is_insufficient_material():

            print(
                "RESULT: DRAW - INSUFFICIENT MATERIAL"
            )


        elif board.is_fivefold_repetition():

            print(
                "RESULT: DRAW - FIVEFOLD REPETITION"
            )


        elif board.is_seventyfive_moves():

            print(
                "RESULT: DRAW - 75 MOVE RULE"
            )


        elif move_count >= MAX_MOVES:

            print(
                "RESULT: GAME STOPPED - MAX MOVES"
            )


        else:

            print(
                "RESULT:",
                board.result()
            )


        # =====================================================
        # GAME INFORMATION
        # =====================================================

        print()

        print(
            "Total plies:",
            move_count
        )

        print(
            "FEN:"
        )

        print(
            board.fen()
        )

        print()


        # -----------------------------------------------------
        # Print PGN-style move list.
        # -----------------------------------------------------

        print(
            "Game moves:"
        )

        print(
            board.move_stack
        )

        print()


    finally:

        print(
            "Shutting down Stockfish..."
        )

        engine.quit()

        print(
            "Stockfish stopped."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Find Stockfish.
    # --------------------------------------------------------

    stockfish_path = find_stockfish()


    if stockfish_path is None:

        print()
        print("==============================")
        print("       STOCKFISH NOT FOUND")
        print("==============================")
        print()

        print(
            "The Python package may be installed, "
            "but the Stockfish executable could not be found."
        )

        print()

        print(
            "Try:"
        )

        print(
            "which stockfish"
        )

        print()

        return


    print(
        "Stockfish:",
        stockfish_path
    )


    # --------------------------------------------------------
    # Load model.
    # --------------------------------------------------------

    try:

        model = load_model()

    except Exception as e:

        print()
        print("==============================")
        print("       MODEL LOAD ERROR")
        print("==============================")
        print()

        print(
            repr(e)
        )

        print()

        return


    # --------------------------------------------------------
    # Run game.
    # --------------------------------------------------------

    try:

        play_game(
            model,
            stockfish_path
        )

    except Exception as e:

        print()
        print("==============================")
        print("          GAME ERROR")
        print("==============================")
        print()

        print(
            repr(e)
        )

        print()


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    main()