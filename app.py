from flask import Flask, render_template_string, request, jsonify
import chess
import torch

from model import ChessNet
from utils import board_to_tensor, move_to_index


app = Flask(__name__)


# =========================
# LOAD AI
# =========================

model = ChessNet()

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
    print("AI will use random weights.")

model.eval()


# =========================
# GAME
# =========================

board = chess.Board()


def get_ai_move():
    """Choose the highest-scoring legal move."""

    tensor = board_to_tensor(board).unsqueeze(0)

    with torch.no_grad():
        policy, value = model(tensor)

    probabilities = policy[0].cpu().numpy()

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
# WEB PAGE
# =========================

HTML = """
<!DOCTYPE html>
<html>

<head>

    <title>Chess AI</title>

    <style>

        body {
            background: #222;
            color: white;
            font-family: Arial, sans-serif;
            text-align: center;
        }

        h1 {
            margin-top: 20px;
        }

        #board {
            width: 640px;
            height: 640px;
            margin: 20px auto;
            display: grid;
            grid-template-columns: repeat(8, 1fr);
            border: 4px solid #111;
        }

        .square {
            width: 80px;
            height: 80px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 55px;
            cursor: pointer;
            user-select: none;
        }

        .light {
            background: #f0d9b5;
        }

        .dark {
            background: #b58863;
        }

        .selected {
            background: #f6f669 !important;
        }

        .legal {
            box-shadow: inset 0 0 0 6px rgba(50, 180, 50, 0.6);
        }

        #status {
            font-size: 20px;
            margin: 15px;
        }

        button {
            font-size: 18px;
            padding: 10px 20px;
            cursor: pointer;
        }

    </style>

</head>


<body>

    <h1>♟ Chess AI</h1>

    <div id="status">
        Your turn
    </div>

    <div id="board"></div>

    <button onclick="newGame()">
        New Game
    </button>


<script>

let boardData = null;
let selectedSquare = null;


// =========================
// GET BOARD
// =========================

async function getBoard() {

    const response = await fetch("/board");

    boardData = await response.json();

    drawBoard();
}


// =========================
// DRAW BOARD
// =========================

function drawBoard() {

    const boardElement = document.getElementById("board");

    boardElement.innerHTML = "";

    for (let row = 0; row < 8; row++) {

        for (let col = 0; col < 8; col++) {

            const square = row * 8 + col;

            const div = document.createElement("div");

            div.classList.add("square");

            if ((row + col) % 2 === 0) {
                div.classList.add("light");
            } else {
                div.classList.add("dark");
            }

            if (square === selectedSquare) {
                div.classList.add("selected");
            }

            if (
                boardData.legal_moves &&
                boardData.legal_moves.includes(square)
            ) {
                div.classList.add("legal");
            }

            const piece = boardData.board[row][col];

            if (piece) {
                div.innerText = piece.symbol;
    
                if (piece.color === "black") {
                        div.classList.add("black-piece");
                } else {
                    div.classList.add("white-piece");
    }
}


            div.onclick = function() {
                clickSquare(square);
            };

            boardElement.appendChild(div);
        }
    }

    document.getElementById("status").innerText =
        boardData.status;
}


// =========================
// CLICK SQUARE
// =========================

async function clickSquare(square) {

    // First click selects a piece
    if (selectedSquare === null) {

        selectedSquare = square;

        drawBoard();

        return;
    }


    // Second click attempts a move

    const response = await fetch("/move", {

        method: "POST",

        headers: {
            "Content-Type": "application/json"
        },

        body: JSON.stringify({
            from: selectedSquare,
            to: square
        })

    });

    const result = await response.json();

    selectedSquare = null;

    if (!result.success) {

        document.getElementById("status").innerText =
            result.message;

        drawBoard();

        return;
    }

    boardData = result;

    drawBoard();


    // AI move

    if (!result.game_over) {

        document.getElementById("status").innerText =
            "AI is thinking...";

        const aiResponse = await fetch("/ai_move", {
            method: "POST"
        });

        const aiResult = await aiResponse.json();

        boardData = aiResult;

        drawBoard();
    }
}


// =========================
// NEW GAME
// =========================

async function newGame() {

    await fetch("/new_game", {
        method: "POST"
    });

    selectedSquare = null;

    await getBoard();
}


getBoard();

</script>

</body>

</html>
"""


# =========================
# HELPERS
# =========================

PIECES = {
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
    "k": "♚"
}


def board_json():

    board_array = []

    for row in range(8):

        row_data = []

        for col in range(8):

            square = chess.square(col, 7 - row)

            piece = board.piece_at(square)

            if piece:
                row_data.append(
                    PIECES[piece.symbol()]
                )
            else:
                row_data.append("")

        board_array.append(row_data)


    legal_moves = []

    if board.turn == chess.WHITE:

        for move in board.legal_moves:

            legal_moves.append(
                move.to_square
            )


    if board.is_game_over():

        status = "Game Over: " + board.result()

    elif board.turn == chess.WHITE:

        status = "Your turn"

    else:

        status = "AI's turn"


    return {
        "board": board_array,
        "legal_moves": legal_moves,
        "status": status,
        "game_over": board.is_game_over()
    }


# =========================
# ROUTES
# =========================

@app.route("/")
def index():

    return render_template_string(HTML)


@app.route("/board")
def get_board():

    return jsonify(board_json())


@app.route("/new_game", methods=["POST"])
def new_game():

    global board

    board = chess.Board()

    return jsonify(board_json())


@app.route("/move", methods=["POST"])
def player_move():

    data = request.json

    from_square = data["from"]
    to_square = data["to"]


    # Convert screen square to chess square

    from_col = from_square % 8
    from_row = from_square // 8

    to_col = to_square % 8
    to_row = to_square // 8


    from_chess_square = chess.square(
        from_col,
        7 - from_row
    )

    to_chess_square = chess.square(
        to_col,
        7 - to_row
    )


    move = chess.Move(
        from_chess_square,
        to_chess_square
    )


    # Handle promotion
    piece = board.piece_at(from_chess_square)

    if (
        piece is not None
        and piece.piece_type == chess.PAWN
        and chess.square_rank(to_chess_square) in (0, 7)
    ):

        move = chess.Move(
            from_chess_square,
            to_chess_square,
            promotion=chess.QUEEN
        )


    if move not in board.legal_moves:

        return jsonify({
            "success": False,
            "message": "Illegal move."
        })


    board.push(move)

    result = board_json()

    result["success"] = True

    return jsonify(result)


@app.route("/ai_move", methods=["POST"])
def ai_move():

    if board.is_game_over():

        return jsonify(board_json())


    if board.turn != chess.BLACK:

        return jsonify(board_json())


    move = get_ai_move()

    if move is not None:

        board.push(move)


    return jsonify(board_json())


# =========================
# START SERVER
# =========================

if __name__ == "__main__":

    print()
    print("==============================")
    print("       CHESS AI SERVER")
    print("==============================")
    print()
    print("Open the forwarded port 8000")
    print()

    app.run(
        host="0.0.0.0",
        port=8000,
        debug=False
    )
