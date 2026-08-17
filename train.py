import torch
import chess
# Import the function we made in the previous step
from utils import board_to_tensor 
# Import the AI model architecture
from model import ChessNet 

def main():
    # 1. Initialize the components
    board = chess.Board()       # Standard chess game starting position
    model = ChessNet()          # Your AI brain
    
    # 2. Convert the board using NumPy (handled inside board_to_tensor)
    # The output is a PyTorch tensor with a shape of [14, 8, 8]
    raw_tensor = board_to_tensor(board)
    
    # 3. Prepare the data for PyTorch
    # Neural networks expect data in "batches" (groups of boards).
    # unsqueeze(0) changes the shape from [14, 8, 8] to [1, 14, 8, 8] (a batch of 1 board).
    input_tensor = raw_tensor.unsqueeze(0)
    
    # 4. Feed the data into the AI
    # We turn off gradient tracking (eval mode) just to test the forward pass
    model.eval()
    with torch.no_grad():
        policy, value = model(input_tensor)
        
    # 5. See what the AI thought
    print("--- AI Output Success ---")
    print("Raw Input Shape to AI:", input_tensor.shape)
    print("Move Predictions Matrix Shape (Policy):", policy.shape)
    print("Board Evaluation Value (Value):", value.item())

if __name__ == "__main__":
    main()
