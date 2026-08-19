import numpy as np
import torch
import chess


# ============================================================
# BOARD -> TENSOR
# ============================================================

def board_to_tensor(board: chess.Board):

    matrix = np.zeros(
        (14, 8, 8),
        dtype=np.float32
    )


    piece_to_layer = {

        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5

    }


    # ========================================================
    # PIECES
    # ========================================================

    for square in chess.SQUARES:

        piece = board.piece_at(square)

        if piece is None:
            continue


        row = chess.square_rank(square)

        col = chess.square_file(square)


        if piece.color == chess.WHITE:

            color_offset = 0

        else:

            color_offset = 6


        layer = (
            piece_to_layer[piece.piece_type]
            + color_offset
        )


        matrix[layer][row][col] = 1.0


    # ========================================================
    # SIDE TO MOVE
    # ========================================================

    if board.turn == chess.WHITE:

        matrix[12][:, :] = 1.0

    else:

        matrix[12][:, :] = 0.0


    # ========================================================
    # CASTLING RIGHTS
    # ========================================================

    has_castling = (

        board.has_kingside_castling_rights(
            chess.WHITE
        )

        or

        board.has_queenside_castling_rights(
            chess.WHITE
        )

        or

        board.has_kingside_castling_rights(
            chess.BLACK
        )

        or

        board.has_queenside_castling_rights(
            chess.BLACK
        )

    )


    if has_castling:

        matrix[13][:, :] = 1.0


    return torch.from_numpy(matrix)


# ============================================================
# MOVE -> INDEX
# ============================================================

def move_to_index(move: chess.Move) -> int:

    return (
        move.from_square * 64
        + move.to_square
    )


# ============================================================
# INDEX -> MOVE
# ============================================================

def index_to_move(index: int) -> chess.Move:

    if index < 0 or index >= 4096:

        raise ValueError(
            "Move index must be between 0 and 4095."
        )


    from_square = index // 64

    to_square = index % 64


    return chess.Move(
        from_square,
        to_square
    )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    board = chess.Board()

    tensor = board_to_tensor(board)


    print(
        "Tensor shape:",
        tensor.shape
    )


    test_move = chess.Move.from_uci(
        "e2e4"
    )


    index = move_to_index(
        test_move
    )


    restored = index_to_move(
        index
    )


    print(
        "Move:",
        test_move
    )

    print(
        "Index:",
        index
    )

    print(
        "Restored:",
        restored
    )
