import pygame
import torch
import chess

from utils import board_to_tensor, move_to_index, index_to_move
from model import ChessNet


# =========================
# SETTINGS
# =========================

WIDTH = 640
HEIGHT = 640
SQUARE_SIZE = WIDTH // 8

LIGHT_SQUARE = (240, 217, 181)
DARK_SQUARE = (181, 136, 99)

SELECTED_COLOR = (246, 246, 105)
LEGAL_MOVE_COLOR = (100, 180, 100)

WINDOW_TITLE = "Chess AI"


# =========================
# PIECE IMAGES
# =========================

PIECE_UNICODE = {
    "P": "♙",
    "N": "♘",
    "B": "♗",
    "R": "♖",
    "Q": "♕",
    "K": "♔",

    "p": "♟",
    "n": "♞",
    "b": "♝",
    "r": "♜",
    "q": "♛",
    "k": "♚",
}


# =========================
# AI
# =========================

def select_best_legal_move(board: chess.Board, model: ChessNet) -> chess.Move:
    """
    Ask the neural network for a move and choose the
    highest-scoring legal move.
    """

    input_tensor = board_to_tensor(board).unsqueeze(0)

    with torch.no_grad():
        policy, _ = model(input_tensor)

    probabilities = policy.squeeze(0).cpu().numpy()

    best_move = None
    best_score = float("-inf")

    for move in board.legal_moves:
        index = move_to_index(move)
        score = probabilities[index]

        if score > best_score:
            best_score = score
            best_move = move

    return best_move


# =========================
# DRAW BOARD
# =========================

def draw_board(screen, board, selected_square=None, legal_moves=None, font=None):

    # Draw squares
    for row in range(8):
        for col in range(8):

            # Convert screen coordinates to chess coordinates
            square = chess.square(col, 7 - row)

            if (row + col) % 2 == 0:
                color = LIGHT_SQUARE
            else:
                color = DARK_SQUARE

            # Highlight selected square
            if square == selected_square:
                color = SELECTED_COLOR

            # Highlight legal destinations
            if legal_moves:
                for move in legal_moves:
                    if move.to_square == square:
                        color = LEGAL_MOVE_COLOR

            pygame.draw.rect(
                screen,
                color,
                (
                    col * SQUARE_SIZE,
                    row * SQUARE_SIZE,
                    SQUARE_SIZE,
                    SQUARE_SIZE
                )
            )

    # Draw pieces
    for square in chess.SQUARES:

        piece = board.piece_at(square)

        if piece is None:
            continue

        col = chess.square_file(square)
        row = 7 - chess.square_rank(square)

        piece_symbol = piece.symbol()
        text = PIECE_UNICODE[piece_symbol]

        # White pieces have dark outline
        if piece.color == chess.WHITE:
            piece_color = (255, 255, 255)
            outline_color = (20, 20, 20)
        else:
            piece_color = (20, 20, 20)
            outline_color = (240, 240, 240)

        # Draw outline
        outline = font.render(text, True, outline_color)
        screen.blit(
            outline,
            (
                col * SQUARE_SIZE + SQUARE_SIZE // 2 - outline.get_width() // 2 + 2,
                row * SQUARE_SIZE + SQUARE_SIZE // 2 - outline.get_height() // 2 + 2
            )
        )

        # Draw piece
        piece_surface = font.render(text, True, piece_color)

        screen.blit(
            piece_surface,
            (
                col * SQUARE_SIZE + SQUARE_SIZE // 2 - piece_surface.get_width() // 2,
                row * SQUARE_SIZE + SQUARE_SIZE // 2 - piece_surface.get_height() // 2
            )
        )


# =========================
# SCREEN COORDINATES
# =========================

def screen_to_square(x, y):
    """
    Convert mouse coordinates into a python-chess square.
    """

    col = x // SQUARE_SIZE
    row = y // SQUARE_SIZE

    if col < 0 or col > 7 or row < 0 or row > 7:
        return None

    return chess.square(col, 7 - row)


# =========================
# MAIN GAME
# =========================

def main():

    pygame.init()

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(WINDOW_TITLE)

    # Large Unicode font for chess pieces
    piece_font = pygame.font.SysFont(
        "dejavusans",
        64
    )

    clock = pygame.time.Clock()

    # Create chess board
    board = chess.Board()

    # Create AI
    model = ChessNet()

    # Load trained weights
    try:
        model.load_state_dict(
            torch.load(
                "chess_model_weights.pth",
                map_location="cpu"
            )
        )

        print("Loaded trained AI.")

    except FileNotFoundError:
        print("WARNING: chess_model_weights.pth not found.")
        print("The AI will use random weights.")

    model.eval()

    selected_square = None
    running = True

    while running:

        # =========================
        # EVENTS
        # =========================

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                running = False

            # Mouse click
            elif event.type == pygame.MOUSEBUTTONDOWN:

                # Only allow player to move as White
                if board.turn != chess.WHITE:
                    continue

                x, y = pygame.mouse.get_pos()

                clicked_square = screen_to_square(x, y)

                if clicked_square is None:
                    continue

                # --------------------------------
                # First click: select a piece
                # --------------------------------

                if selected_square is None:

                    piece = board.piece_at(clicked_square)

                    if piece is not None and piece.color == chess.WHITE:

                        selected_square = clicked_square

                # --------------------------------
                # Second click: select destination
                # --------------------------------

                else:

                    move = chess.Move(
                        selected_square,
                        clicked_square
                    )

                    # Handle pawn promotion automatically
                    piece = board.piece_at(selected_square)

                    if (
                        piece is not None
                        and piece.piece_type == chess.PAWN
                        and chess.square_rank(clicked_square) in (0, 7)
                    ):
                        move = chess.Move(
                            selected_square,
                            clicked_square,
                            promotion=chess.QUEEN
                        )

                    # Check if legal
                    if move in board.legal_moves:

                        board.push(move)

                        print("You played:", move)

                        selected_square = None

                    else:

                        # If clicking another friendly piece,
                        # select that instead.
                        piece = board.piece_at(clicked_square)

                        if piece is not None and piece.color == chess.WHITE:
                            selected_square = clicked_square
                        else:
                            selected_square = None

        # =========================
        # AI TURN
        # =========================

        if board.turn == chess.BLACK and not board.is_game_over():

            print("AI is thinking...")

            ai_move = select_best_legal_move(
                board,
                model
            )

            if ai_move is not None:

                print("AI played:", ai_move)

                board.push(ai_move)

        # =========================
        # DRAW
        # =========================

        legal_moves = []

        if selected_square is not None:

            legal_moves = [
                move
                for move in board.legal_moves
                if move.from_square == selected_square
            ]

        draw_board(
            screen,
            board,
            selected_square,
            legal_moves,
            piece_font
        )

        pygame.display.flip()

        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
