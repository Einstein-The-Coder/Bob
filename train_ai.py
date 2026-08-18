import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import chess.pgn
from utils import board_to_tensor, move_to_index
from model import ChessNet

def train_on_folder(folder_path, epochs=5):
    # 1. Initialize model, optimizer, and loss function
    model = ChessNet()
    model.train()
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    policy_criterion = nn.CrossEntropyLoss()

    # 2. Find all PGN files inside the target folder
    # This searches for anything ending in .pgn inside the folder
    search_path = os.path.join(folder_path, "*.pgn")
    pgn_files = glob.glob(search_path)
    
    if not pgn_files:
        print(f"❌ Error: No .pgn files found in the folder '{folder_path}'!")
        print("Please make sure you created the folder and placed your files inside it.")
        return

    print(f"Found {len(pgn_files)} PGN file(s) for training.")

    # 3. Main training loop across epochs
    for epoch in range(epochs):
        total_loss = 0.0
        moves_processed = 0
        
        # Loop through every single PGN file found in the folder
        for pgn_filename in pgn_files:
            print(f"  → Processing file: {os.path.basename(pgn_filename)}")
            
            with open(pgn_filename) as pgn_file:
                while True:
                    game = chess.pgn.read_game(pgn_file)
                    if game is None:
                        break  # Out of games in this specific file
                    
                    board = game.board()
                    
                    # Process every move in the current game
                    for move in game.mainline_moves():
                        input_tensor = board_to_tensor(board).unsqueeze(0)
                        target_move_idx = torch.tensor([move_to_index(move)], dtype=torch.long)
                        
                        # Forward pass
                        policy, value = model(input_tensor)
                        loss = policy_criterion(policy, target_move_idx)
                        
                        # Backward pass (Learning step)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        
                        total_loss += loss.item()
                        moves_processed += 1
                        board.push(move)
                        
        # Print progress status for the entire epoch
        avg_loss = total_loss / max(1, moves_processed)
        print(f"✨ Epoch {epoch+1}/{epochs} Complete | Total Moves: {moves_processed} | Avg Loss: {avg_loss:.4f}\n")
        
    # Save the updated smart weights
    torch.save(model.state_dict(), "chess_model_weights.pth")
    print("Training finished! Saved smart weights to 'chess_model_weights.pth'")

if __name__ == "__main__":
    # Point the training function to your "train" folder
    train_on_folder("train", epochs=3)
