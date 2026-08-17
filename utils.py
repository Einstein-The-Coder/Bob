import numpy as np
import torch
import chess

def board_to_tensor(board: chess.Board):
    # 1. Initialize an empty 14x8x8 matrix filled with zeros
    matrix = np.zeros((14, 8, 8), dtype=np.float32)
    
    # 2. Map piece types to specific layer indices
    # python-chess constants: PAWN=1, KNIGHT=2, BISHOP=3, ROOK=4, QUEEN=5, KING=6
    piece_to_layer = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5
    }
    
    # 3. Fill layers 0 to 11 with piece locations
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            # Get 2D grid coordinates (0-7, 0-7) from 1D square index (0-63)
            row = chess.square_rank(square)
            col = chess.square_file(square)
            
            # Determine base layer based on color (White = 0, Black = 6)
            color_offset = 0 if piece.color == chess.WHITE else 6
            layer_index = piece_to_layer[piece.piece_type] + color_offset
            
            # Place a 1 where the piece exists
            matrix[layer_index][row][col] = 1.0
            
    # 4. Fill Layer 12: Active player turn
    if board.turn == chess.WHITE:
        matrix[12][:, :] = 1.0  # Fill the entire 8x8 grid with 1s
    else:
        matrix[12][:, :] = 0.0  # Fill the entire 8x8 grid with 0s
        
    # 5. Fill Layer 13: Castling rights availability
    has_castling = (
        board.has_kingside_castling_rights(chess.WHITE) or
        board.has_queenside_castling_rights(chess.WHITE) or
        board.has_kingside_castling_rights(chess.BLACK) or
        board.has_queenside_castling_rights(chess.BLACK)
    )
    if has_castling:
        matrix[13][:, :] = 1.0
        
    # 6. Convert the NumPy matrix into a PyTorch tensor
    return torch.from_numpy(matrix)

# --- Quick Test Example ---
if __name__ == "__main__":
    # Create a standard starting board
    test_board = chess.Board()
    
    # Convert it
    board_tensor = board_to_tensor(test_board)
    
    # Expected output shape: torch.Size([14, 8, 8])
    print("Tensor Shape:", board_tensor.shape) 

def move_to_index(move: chess.Move) -> int:
    """Converts a chess.Move object into a unique index between 0 and 4095."""
    from_square = move.from_square  # Number from 0 to 63
    to_square = move.to_square      # Number from 0 to 63
    return from_square * 64 + to_square

def index_to_move(index: int) -> chess.Move:
    """Converts an index between 0 and 4095 back into a chess.Move object."""
    from_square = index // 64
    to_square = index % 64
    return chess.Move(from_square, to_square)