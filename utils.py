import numpy as np
import torch
import chess


# ============================================================
# CONFIGURATION
# ============================================================

# 12 piece planes
# + 1 side-to-move plane
# + 4 castling-rights planes
# + 1 en-passant plane
#
# TOTAL = 18 CHANNELS

INPUT_CHANNELS = 18


# ============================================================
# PIECE CHANNELS
# ============================================================

# White:
#   0 = pawn
#   1 = knight
#   2 = bishop
#   3 = rook
#   4 = queen
#   5 = king
#
# Black:
#   6 = pawn
#   7 = knight
#   8 = bishop
#   9 = rook
#   10 = queen
#   11 = king

PIECE_TO_LAYER = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}


# ============================================================
# BOARD -> TENSOR
# ============================================================

def board_to_tensor(board: chess.Board) -> torch.Tensor:
    """
    Convert a chess.Board into an 18 x 8 x 8 float tensor.

    Channels:

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

    matrix = np.zeros(
        (INPUT_CHANNELS, 8, 8),
        dtype=np.float32
    )


    # ========================================================
    # PIECES
    # ========================================================

    for square in chess.SQUARES:

        piece = board.piece_at(square)

        if piece is None:
            continue


        # python-chess:
        #
        # rank 0 = rank 1
        # rank 7 = rank 8
        #
        # file 0 = a
        # file 7 = h

        row = chess.square_rank(square)
        col = chess.square_file(square)


        base_layer = PIECE_TO_LAYER[
            piece.piece_type
        ]


        # White = channels 0-5
        # Black = channels 6-11

        if piece.color == chess.WHITE:

            layer = base_layer

        else:

            layer = base_layer + 6


        matrix[layer, row, col] = 1.0


    # ========================================================
    # SIDE TO MOVE
    # ========================================================

    # Entire plane is 1 when White is to move.
    # Entire plane is 0 when Black is to move.

    if board.turn == chess.WHITE:

        matrix[12, :, :] = 1.0

    else:

        matrix[12, :, :] = 0.0


    # ========================================================
    # CASTLING RIGHTS
    # ========================================================

    # White kingside: O-O

    if board.has_kingside_castling_rights(
        chess.WHITE
    ):

        matrix[13, :, :] = 1.0


    # White queenside: O-O-O

    if board.has_queenside_castling_rights(
        chess.WHITE
    ):

        matrix[14, :, :] = 1.0


    # Black kingside: O-O

    if board.has_kingside_castling_rights(
        chess.BLACK
    ):

        matrix[15, :, :] = 1.0


    # Black queenside: O-O-O

    if board.has_queenside_castling_rights(
        chess.BLACK
    ):

        matrix[16, :, :] = 1.0


    # ========================================================
    # EN-PASSANT
    # ========================================================

    # Normally there is no en-passant square.

    if board.ep_square is not None:

        row = chess.square_rank(
            board.ep_square
        )

        col = chess.square_file(
            board.ep_square
        )

        matrix[
            17,
            row,
            col
        ] = 1.0


    # ========================================================
    # NUMPY -> PYTORCH
    # ========================================================

    return torch.from_numpy(matrix)


# ============================================================
# MOVE -> INDEX
# ============================================================

def move_to_index(move: chess.Move) -> int:
    """
    Convert a chess move to an index from 0 to 4095.

    Formula:

        from_square * 64 + to_square

    This gives:

        64 * 64 = 4096 possible indices.

    IMPORTANT:
    This encoding does NOT distinguish promotion pieces.

    For example:

        e7e8=Q
        e7e8=R
        e7e8=B
        e7e8=N

    all have the same base index.

    The Flask game currently promotes automatically to a queen,
    so this is compatible with your current application.
    """

    if not isinstance(move, chess.Move):
        raise TypeError(
            "move must be a chess.Move"
        )


    from_square = move.from_square
    to_square = move.to_square


    return (
        from_square * 64
        + to_square
    )


# ============================================================
# INDEX -> MOVE
# ============================================================

def index_to_move(index: int) -> chess.Move:
    """
    Convert a 0-4095 policy index back into a chess.Move.
    """

    if not isinstance(index, (int, np.integer)):
        raise TypeError(
            "index must be an integer"
        )


    if index < 0 or index >= 4096:
        raise ValueError(
            "index must be between 0 and 4095"
        )


    from_square = index // 64
    to_square = index % 64


    return chess.Move(
        from_square,
        to_square
    )


# ============================================================
# SAN / UCI TEST
# ============================================================

def test_board_encoding():
    """
    Basic sanity check for the board representation.
    """

    board = chess.Board()

    tensor = board_to_tensor(board)


    assert tensor.shape == (
        18,
        8,
        8
    ), (
        f"Wrong tensor shape: {tensor.shape}"
    )


    assert tensor.dtype == torch.float32


    print("Board tensor test: PASSED")
    print("Shape:", tensor.shape)


# ============================================================
# MOVE ENCODING TEST
# ============================================================

def test_move_encoding():
    """
    Test that moves can be converted to an index
    and back.
    """

    moves = [
        "e2e4",
        "g1f3",
        "b1c3",
        "e1g1",
        "e7e8",
    ]


    for uci in moves:

        move = chess.Move.from_uci(uci)

        index = move_to_index(move)

        restored = index_to_move(index)


        assert (
            restored.from_square
            == move.from_square
        )

        assert (
            restored.to_square
            == move.to_square
        )


        print(
            f"{uci} -> {index} -> {restored}"
        )


    print("Move encoding test: PASSED")


# ============================================================
# COMPLETE TEST
# ============================================================

def test_everything():

    print()
    print("==============================")
    print("      UTILS.PY TEST")
    print("==============================")
    print()


    test_board_encoding()

    print()

    test_move_encoding()

    print()

    # Test starting position

    board = chess.Board()

    print(
        "Starting position:"
    )

    print(board)

    print()

    tensor = board_to_tensor(board)

    print(
        "Tensor shape:",
        tensor.shape
    )

    print(
        "Tensor channels:",
        tensor.shape[0]
    )

    print()

    # Starting position should have:
    #
    # 8 white pawns
    # 2 white knights
    # 2 white bishops
    # 2 white rooks
    # 1 white queen
    # 1 white king
    #
    # and the same for black.

    print(
        "White pawns:",
        int(tensor[0].sum())
    )

    print(
        "Black pawns:",
        int(tensor[6].sum())
    )

    print(
        "White king:",
        int(tensor[5].sum())
    )

    print(
        "Black king:",
        int(tensor[11].sum())
    )

    print(
        "White to move plane:",
        int(tensor[12].sum())
    )

    print(
        "White kingside castling:",
        int(tensor[13].sum())
    )

    print(
        "White queenside castling:",
        int(tensor[14].sum())
    )

    print(
        "Black kingside castling:",
        int(tensor[15].sum())
    )

    print(
        "Black queenside castling:",
        int(tensor[16].sum())
    )

    print(
        "En-passant squares:",
        int(tensor[17].sum())
    )

    print()
    print("==============================")
    print("       ALL TESTS PASSED")
    print("==============================")
    print()


# ============================================================
# RUN TESTS
# ============================================================

if __name__ == "__main__":

    test_everything()