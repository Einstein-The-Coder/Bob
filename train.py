import torch
import chess
import numpy as np
from utils import board_to_tensor, move_to_index, index_to_move
from model import ChessNet

def select_best_legal_move(board: chess.Board, model: ChessNet) -> chess.Move:
    """Passes the board to the AI model and extracts the best legal move."""
    # 1. Format the board and pass it through the model
    input_tensor = board_to_tensor(board).unsqueeze(0)
    
    with torch.no_grad():
        policy, _ = model(input_tensor)
    
    # 2. Extract probabilities and gather legal moves
    probabilities = policy.squeeze(0).cpu().numpy()
    legal_moves = list(board.legal_moves)
    
    # 3. Mask out all illegal moves
    legal_move_scores = np.full(4096, -np.inf)
    for move in legal_moves:
        idx = move_to_index(move)
        legal_move_scores[idx] = probabilities[idx]
        
    # 4. Select and return the highest-scoring legal move
    best_move_idx = np.argmax(legal_move_scores)
    return index_to_move(best_move_idx)

def main():
    # Initialize the board and the AI
    board = chess.Board()
    model = ChessNet()
    model.eval() # Set model to evaluation mode
    
    print("=" * 40)
    print("      CHESS AI INTERACTIVE GAME")
    print("=" * 40)
    print("Type your moves using 4-character notation (e.g., e2e4, g1f3).")
    print("Type 'quit' at any time to exit the game.\n")

    # Main gameplay loop
    while not board.is_game_over():
        # Print the current board state to the terminal
        print("\nCurrent Board State:")
        print(board)
        print("-" * 20)
        
        if board.turn == chess.WHITE:
            # --- HUMAN PLAYER TURN ---
            user_input = input("Your Move (White): ").strip()
            
            if user_input.lower() == 'quit':
                print("Game ended by player.")
                return
                
            try:
                # Convert the input text into a chess.Move object
                move = chess.Move.from_uci(user_input)
                if move in board.legal_moves:
                    board.push(move)
                else:
                    print("❌ That move is illegal! Try again.")
            except ValueError:
                print("❌ Invalid input format. Use coordinates like 'e2e4'.")
        else:
            # --- AI PLAYER TURN ---
            print("AI (Black) is thinking...")
            ai_move = select_best_legal_move(board, model)
            print(f"AI Played: {ai_move}")
            board.push(ai_move)

    # Game Over handling
    print("\n" + "=" * 40)
    print("GAME OVER")
    print(f"Result: {board.result()}")
    print("=" * 40)

if __name__ == "__main__":
    main()
