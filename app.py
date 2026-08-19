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

except Exception as e:

    print("WARNING: Could not load AI weights.")
    print("Error:", e)

model.eval()


# ============================================================
# GAME
# ============================================================

board = chess.Board()


# ============================================================
# PIECES
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

    input_tensor = (
        board_to_tensor(board)
        .unsqueeze(0)
    )

    with torch.no_grad():

        policy, _ = model(
            input_tensor
        )

    scores = (
        policy[0]
        .cpu()
        .numpy()
    )

    best_move = None
    best_score = float("-inf")

    for move in board.legal_moves:

        index = move_to_index(move)

        if index >= len(scores):
            continue

        score = float(
            scores[index]
        )

        if score > best_score:

            best_score = score
            best_move = move

    return best_move


# ============================================================
# BOARD JSON
# ============================================================

def board_json():

    board_array = []

    for row in range(8):

        row_data = []

        for col in range(8):

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
                    ),
                    "type": piece.piece_type
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
    # STATUS
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


    elif board.is_game_over():

        status = (
            "Game Over: "
            + board.result()
        )


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
# HTML
# ============================================================

HTML = r"""
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

            width: min(640px, 90vw);

            height: min(640px, 90vw);

            margin: 20px auto;

            display: grid;

            grid-template-columns:
                repeat(8, 1fr);

            border: 4px solid #111;

        }


        .square {

            position: relative;

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
                rgba(200, 50, 50, 0.75);

        }


        .piece {

            font-family:
                "DejaVu Sans",
                "Segoe UI Symbol",
                "Noto Sans Symbols 2",
                sans-serif;

            font-size: clamp(
                35px,
                8vw,
                64px
            );

            line-height: 1;

            pointer-events: none;

        }


        /*
         * White pieces
         */

        .white-piece {

            color: #ffffff;

            text-shadow:
                -2px -2px 0 #111,
                 2px -2px 0 #111,
                -2px  2px 0 #111,
                 2px  2px 0 #111,
                 0px  0px 4px #111;

        }


        /*
         * Black pieces
         */

        .black-piece {

            color: #111111;

            text-shadow:
                -2px -2px 0 #ffffff,
                 2px -2px 0 #ffffff,
                -2px  2px 0 #ffffff,
                 2px  2px 0 #ffffff,
                 0px  0px 4px #ffffff;

        }


        button {

            font-size: 18px;

            padding: 10px 20px;

            margin-bottom: 25px;

            cursor: pointer;

            border: none;

            border-radius: 5px;

        }


        button:hover {

            background: #ddd;

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

        const response = await fetch(
            "/board"
        );

        boardData = await response.json();

        selectedSquare = null;

        drawBoard();

    }

    catch (error) {

        console.error(error);

        document.getElementById(
            "status"
        ).innerText =
            "Could not connect to server.";

    }

}


// ==========================================================
// DRAW BOARD
// ==========================================================

function drawBoard() {

    const boardElement =
        document.getElementById(
            "board"
        );


    boardElement.innerHTML = "";


    for (
        let row = 0;
        row < 8;
        row++
    ) {


        for (
            let col = 0;
            col < 8;
            col++
        ) {


            const square =
                row * 8 + col;


            const div =
                document.createElement(
                    "div"
                );


            div.classList.add(
                "square"
            );


            // Board colors

            if (
                (row + col) % 2 === 0
            ) {

                div.classList.add(
                    "light"
                );

            }

            else {

                div.classList.add(
                    "dark"
                );

            }


            // Selected square

            if (
                square === selectedSquare
            ) {

                div.classList.add(
                    "selected"
                );

            }


            // Legal moves

            const legalMove =
                boardData.legal_moves.find(
                    move =>
                        move.to === square
                );


            if (legalMove) {

                const piece =
                    boardData.board[row][col];


                if (piece) {

                    div.classList.add(
                        "capture"
                    );

                }

                else {

                    div.classList.add(
                        "legal"
                    );

                }

            }


            // Piece

            const piece =
                boardData.board[row][col];


            if (piece) {

                const pieceElement =
                    document.createElement(
                        "span"
                    );


                pieceElement.classList.add(
                    "piece"
                );


                if (
                    piece.color === "black"
                ) {

                    pieceElement.classList.add(
                        "black-piece"
                    );

                }

                else {

                    pieceElement.classList.add(
                        "white-piece"
                    );

                }


                pieceElement.innerText =
                    piece.symbol;


                div.appendChild(
                    pieceElement
                );

            }


            // Click

            div.onclick = function() {

                clickSquare(square);

            };


            boardElement.appendChild(
                div
            );

        }

    }


    document.getElementById(
        "status"
    ).innerText =
        boardData.status;

}


// ==========================================================
// CLICK SQUARE
// ==========================================================

async function clickSquare(square) {


    if (
        boardData.game_over
    ) {

        return;

    }


    if (
        boardData.turn !== "white"
    ) {

        return;

    }


    // ======================================================
    // FIRST CLICK
    // ======================================================

    if (
        selectedSquare === null
    ) {

        const row =
            Math.floor(
                square / 8
            );

        const col =
            square % 8;


        const piece =
            boardData.board[row][col];


        if (
            piece &&
            piece.color === "white"
        ) {

            const canMove =
                boardData.legal_moves.some(
                    move =>
                        move.from === square
                );


            if (canMove) {

                selectedSquare =
                    square;

                drawBoard();

            }

        }


        return;

    }


    // ======================================================
    // SECOND CLICK
    // ======================================================

    const from =
        selectedSquare;


    // Clicking the same square

    if (
        square === selectedSquare
    ) {

        selectedSquare = null;

        drawBoard();

        return;

    }


    const response =
        await fetch(
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


    if (!result.success) {

        boardData = result;

        drawBoard();

        return;

    }


    boardData = result;

    drawBoard();


    // ======================================================
    // AI TURN
    // ======================================================

    if (
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

}


// ==========================================================
// NEW GAME
// ==========================================================

async function newGame() {

    await fetch(
        "/new_game",
        {
            method: "POST"
        }
    );


    selectedSquare = null;

    await getBoard();

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

    return render_template_string(
        HTML
    )


# ============================================================
# BOARD
# ============================================================

@app.route("/board")
def get_board():

    return jsonify(
        board_json()
    )


# ============================================================
# NEW GAME
# ============================================================

@app.route(
    "/new_game",
    methods=["POST"]
)
def new_game():

    global board

    board = chess.Board()

    return jsonify(
        board_json()
    )


# ============================================================
# PLAYER MOVE
# ============================================================

@app.route(
    "/move",
    methods=["POST"]
)
def player_move():

    global board


    data = request.get_json()


    if not data:

        return jsonify({
            "success": False,
            "message": "No move data received."
        })


    try:

        from_square = int(
            data["from"]
        )

        to_square = int(
            data["to"]
        )

    except (
        KeyError,
        TypeError,
        ValueError
    ):

        return jsonify({
            "success": False,
            "message": "Invalid move data."
        })


    # Only White is controlled by player.

    if board.turn != chess.WHITE:

        result = board_json()

        result["success"] = False

        result["message"] = (
            "It is the AI's turn."
        )

        return jsonify(result)


    # ========================================================
    # SCREEN -> CHESS
    # ========================================================

    from_col = (
        from_square % 8
    )

    from_row = (
        from_square // 8
    )


    to_col = (
        to_square % 8
    )

    to_row = (
        to_square // 8
    )


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


    # ========================================================
    # PROMOTION
    # ========================================================

    piece = board.piece_at(
        from_chess_square
    )


    if (
        piece is not None
        and piece.piece_type
        == chess.PAWN
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
    # CHECK LEGAL
    # ========================================================

    if move not in board.legal_moves:

        result = board_json()

        result["success"] = False

        result["message"] = (
            "Illegal move."
        )

        return jsonify(result)


    # ========================================================
    # MAKE MOVE
    # ========================================================

    board.push(move)


    result = board_json()

    result["success"] = True


    return jsonify(result)


# ============================================================
# AI MOVE
# ============================================================

@app.route(
    "/ai_move",
    methods=["POST"]
)
def ai_move():

    global board


    if board.is_game_over():

        return jsonify(
            board_json()
        )


    if board.turn != chess.BLACK:

        return jsonify(
            board_json()
        )


    move = get_ai_move()


    if move is not None:

        board.push(move)


    return jsonify(
        board_json()
    )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    print()
    print("==============================")
    print("       BOB AI SERVER")
    print("==============================")
    print()
    print("Starting server...")
    print("Open the forwarded port 8000.")
    print()


    app.run(
        host="0.0.0.0",
        port=8000,
        debug=False
    )
