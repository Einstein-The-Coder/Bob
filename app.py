from flask import Flask, render_template_string, request, jsonify
import chess
import torch

from model import ChessNet
from utils import board_to_tensor, move_to_index


app = Flask(__name__)


# ============================================================
# LOAD AI
# ============================================================

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

except Exception as error:
    print("WARNING: Could not load AI weights.")
    print("Error:", error)

model.eval()


# ============================================================
# CHESS BOARD
# ============================================================

board = chess.Board()


# ============================================================
# PIECE SYMBOLS
# ============================================================

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


# ============================================================
# AI MOVE
# ============================================================

def get_ai_move():
    """Choose the highest-scoring legal black move."""

    input_tensor = board_to_tensor(board).unsqueeze(0)

    with torch.no_grad():
        policy, _ = model(input_tensor)

    scores = policy[0].cpu().numpy()

    best_move = None
    best_score = float("-inf")

    for move in board.legal_moves:
        index = move_to_index(move)
        score = float(scores[index])

        if score > best_score:
            best_score = score
            best_move = move

    return best_move


# ============================================================
# CONVERT BOARD TO JSON
# ============================================================

def board_json():

    board_array = []

    for row in range(8):

        row_data = []

        for col in range(8):

            # Browser coordinates:
            # row 0 = rank 8
            # row 7 = rank 1

            square = chess.square(
                col,
                7 - row
            )

            piece = board.piece_at(square)

            if piece is None:

                row_data.append(None)

            else:

                row_data.append({
                    "symbol": PIECES[piece.symbol()],
                    "color": (
                        "white"
                        if piece.color == chess.WHITE
                        else "black"
                    )
                })

        board_array.append(row_data)


    # ========================================================
    # LEGAL MOVES
    # ========================================================

    legal_moves = []

    if board.turn == chess.WHITE:

        for move in board.legal_moves:

            legal_moves.append({
                "from": move.from_square,
                "to": move.to_square
            })


    # ========================================================
    # GAME STATUS
    # ========================================================

    if board.is_checkmate():

        if board.turn == chess.WHITE:
            status = "Checkmate — AI wins!"
        else:
            status = "Checkmate — You win!"

    elif board.is_stalemate():

        status = "Draw — Stalemate"

    elif board.is_insufficient_material():

        status = "Draw — Insufficient material"

    elif board.is_check():

        if board.turn == chess.WHITE:
            status = "Your turn — CHECK!"
        else:
            status = "AI's turn — CHECK!"

    elif board.turn == chess.WHITE:

        status = "Your turn"

    else:

        status = "AI's turn"


    return {
        "board": board_array,
        "legal_moves": legal_moves,
        "status": status,
        "game_over": board.is_game_over(),
        "turn": (
            "white"
            if board.turn == chess.WHITE
            else "black"
        )
    }


# ============================================================
# HTML PAGE
# ============================================================

HTML = """
<!DOCTYPE html>

<html>

<head>

    <meta charset="UTF-8">

    <meta name="viewport"
          content="width=device-width, initial-scale=1.0">

    <title>Chess AI</title>

    <style>

        * {
            box-sizing: border-box;
        }

        body {
            margin: 0;
            min-height: 100vh;
            background: #222;
            color: white;
            font-family: Arial, sans-serif;
            text-align: center;
        }

        h1 {
            margin: 20px 0 10px;
        }

        #status {
            font-size: 20px;
            margin: 15px;
            min-height: 25px;
        }

        #board {
            width: min(640px, 92vw);
            height: min(640px, 92vw);
            margin: 20px auto;

            display: grid;
            grid-template-columns: repeat(8, 1fr);

            border: 4px solid #111;
        }

        .square {
            display: flex;
            align-items: center;
            justify-content: center;

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
            box-shadow:
                inset 0 0 0 6px
                rgba(50, 180, 50, 0.65);
        }

        .capture {
            box-shadow:
                inset 0 0 0 6px
                rgba(200, 50, 50, 0.7);
        }

        .piece {
            font-family:
                "DejaVu Sans",
                "Segoe UI Symbol",
                "Noto Sans Symbols 2",
                sans-serif;

            font-size: clamp(35px, 8vw, 64px);

            line-height: 1;

            pointer-events: none;
        }

        /*
         * WHITE PIECES
         */

        .white-piece {
            color: #ffffff;

            text-shadow:
                -2px -2px 0 #111,
                 2px -2px 0 #111,
                -2px  2px 0 #111,
                 2px  2px 0 #111;
        }

        /*
         * BLACK PIECES
         */

        .black-piece {
            color: #111111;

            text-shadow:
                -2px -2px 0 #ffffff,
                 2px -2px 0 #ffffff,
                -2px  2px 0 #ffffff,
                 2px  2px 0 #ffffff;
        }

        button {
            font-size: 18px;
            padding: 10px 20px;
            margin-bottom: 25px;
            cursor: pointer;
        }

    </style>

</head>


<body>

    <h1>♟ Chess AI</h1>

    <div id="status">
        Loading...
    </div>

    <div id="board"></div>

    <button onclick="newGame()">
        New Game
    </button>


<script>

let boardData = null;
let selectedSquare = null;


// ==========================================================
// GET BOARD
// ==========================================================

async function getBoard() {

    try {

        const response = await fetch("/board");

        boardData = await response.json();

        selectedSquare = null;

        drawBoard();

    } catch (error) {

        console.error(error);

        document.getElementById("status").innerText =
            "Could not connect to the chess server.";

    }
}


// ==========================================================
// DRAW BOARD
// ==========================================================

function drawBoard() {

    const boardElement =
        document.getElementById("board");

    boardElement.innerHTML = "";


    for (let row = 0; row < 8; row++) {

        for (let col = 0; col < 8; col++) {

            const square = row * 8 + col;

            const div = document.createElement("div");

            div.classList.add("square");


            // Board color

            if ((row + col) % 2 === 0) {

                div.classList.add("light");

            } else {

                div.classList.add("dark");

            }


            // Selected square

            if (square === selectedSquare) {

                div.classList.add("selected");

            }


            // Find legal move to this square

            const legalMove =
                boardData.legal_moves.find(
                    move => move.to === square
                );


            if (legalMove) {

                const piece =
                    boardData.board[row][col];

                if (piece) {

                    div.classList.add("capture");

                } else {

                    div.classList.add("legal");

                }
            }


            // Piece

            const piece =
                boardData.board[row][col];


            if (piece) {

                const pieceElement =
                    document.createElement("span");

                pieceElement.classList.add("piece");


                if (piece.color === "white") {

                    pieceElement.classList.add(
                        "white-piece"
                    );

                } else {

                    pieceElement.classList.add(
                        "black-piece"
                    );

                }


                pieceElement.innerText =
                    piece.symbol;


                div.appendChild(pieceElement);
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


// ==========================================================
// CLICK SQUARE
// ==========================================================

async function clickSquare(square) {

    // Don't allow moves after game ends

    if (boardData.game_over) {
        return;
    }


    // Player controls White

    if (boardData.turn !== "white") {
        return;
    }


    // ======================================================
    // FIRST CLICK
    // ======================================================

    if (selectedSquare === null) {

        const row = Math.floor(square / 8);

        const col = square % 8;

        const piece =
            boardData.board[row][col];


        if (
            piece &&
            piece.color === "white"
        ) {

            const canMove =
                boardData.legal_moves.some(
                    move => move.from === square
                );


            if (canMove) {

                selectedSquare = square;

                drawBoard();
            }
        }

        return;
    }


    // ======================================================
    // SECOND CLICK
    // ======================================================

    const from = selectedSquare;


    // Click selected square again

    if (square === selectedSquare) {

        selectedSquare = null;

        drawBoard();

        return;
    }


    try {

        const response = await fetch(
            "/move",
            {
                method: "POST",

                headers: {
                    "Content-Type":
                        "application/json"
                },

                body: JSON.stringify({
                    from: from,
                    to: square
                })
            }
        );


        const result =
            await response.json();


        selectedSquare = null;


        boardData = result;

        drawBoard();


        // ==================================================
        // AI TURN
        // ==================================================

        if (
            result.success &&
            !result.game_over
        ) {

            document.getElementById(
                "status"
            ).innerText =
                "AI is thinking...";


            const aiResponse =
                await fetch(
                    "/ai_move",
                    {
                        method: "POST"
                    }
                );


            const aiResult =
                await aiResponse.json();


            boardData = aiResult;

            drawBoard();
        }

    } catch (error) {

        console.error(error);

        document.getElementById(
            "status"
        ).innerText =
            "Error communicating with server.";

    }
}


// ==========================================================
// NEW GAME
// ==========================================================

async function newGame() {

    try {

        await fetch(
            "/new_game",
            {
                method: "POST"
            }
        );

        selectedSquare = null;

        await getBoard();

    } catch (error) {

        console.error(error);

    }
}


// ==========================================================
// START
// ==========================================================

getBoard();

</script>

</body>

</html>
"""


# ============================================================
# ROUTES
# ============================================================

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

    global board


    # ========================================================
    # READ REQUEST
    # ========================================================

    data = request.get_json()


    if not data:

        result = board_json()

        result["success"] = False

        result["message"] = "No move data received."

        return jsonify(result)


    try:

        from_square = int(data["from"])
        to_square = int(data["to"])

    except (KeyError, TypeError, ValueError):

        result = board_json()

        result["success"] = False

        result["message"] = "Invalid move data."

        return jsonify(result)


    # ========================================================
    # ONLY WHITE CAN BE CONTROLLED BY PLAYER
    # ========================================================

    if board.turn != chess.WHITE:

        result = board_json()

        result["success"] = False

        result["message"] = "It is the AI's turn."

        return jsonify(result)


    # ========================================================
    # CHECK RANGE
    # ========================================================

    if (
        from_square < 0
        or from_square > 63
        or to_square < 0
        or to_square > 63
    ):

        result = board_json()

        result["success"] = False

        result["message"] = "Invalid square."

        return jsonify(result)


    # ========================================================
    # BROWSER SQUARE -> CHESS SQUARE
    # ========================================================

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


    # ========================================================
    # CREATE MOVE
    # ========================================================

    move = chess.Move(
        from_chess_square,
        to_chess_square
    )


    # ========================================================
    # AUTOMATIC QUEEN PROMOTION
    # ========================================================

    piece = board.piece_at(
        from_chess_square
    )


    if (
        piece is not None
        and piece.piece_type == chess.PAWN
        and chess.square_rank(
            to_chess_square
        ) in (0, 7)
    ):

        move = chess.Move(
            from_chess_square,
            to_chess_square,
            promotion=chess.QUEEN
        )


    # ========================================================
    # CHECK LEGAL MOVE
    # ========================================================

    if move not in board.legal_moves:

        result = board_json()

        result["success"] = False

        result["message"] = "Illegal move."

        return jsonify(result)


    # ========================================================
    # MAKE PLAYER MOVE
    # ========================================================

    board.push(move)


    result = board_json()

    result["success"] = True

    result["message"] = "Move played."

    return jsonify(result)


@app.route("/ai_move", methods=["POST"])
def ai_move():

    global board


    # Don't move if game is over

    if board.is_game_over():

        return jsonify(board_json())


    # AI controls Black

    if board.turn != chess.BLACK:

        return jsonify(board_json())


    move = get_ai_move()


    if move is not None:

        board.push(move)


    return jsonify(board_json())


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print()
    print("==============================")
    print("       CHESS AI SERVER")
    print("==============================")
    print()
    print("Open port 8000 in your browser.")
    print()


    app.run(
        host="0.0.0.0",
        port=8000,
        debug=False
    )
