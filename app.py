from flask import Flask, render_template_string, request, jsonify
import chess
import torch

from model import ChessNet
from utils import board_to_tensor, move_to_index


app = Flask(__name__)


# ============================================================
# AI
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

    tensor = board_to_tensor(board).unsqueeze(0)

    with torch.no_grad():
        policy, _ = model(tensor)

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
# BOARD DATA FOR BROWSER
# ============================================================

def board_json():

    board_array = []

    # Browser displays:
    #
    # 8 7 6 5 4 3 2 1
    #
    # from top to bottom.
    #
    # Files:
    #
    # a b c d e f g h

    for row in range(8):

        row_data = []

        rank = 7 - row

        for col in range(8):

            square = chess.square(
                col,
                rank
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


    # Legal moves for White

    legal_moves = []

    if board.turn == chess.WHITE:

        for move in board.legal_moves:

            from_col = chess.square_file(
                move.from_square
            )

            from_rank = chess.square_rank(
                move.from_square
            )

            from_row = 7 - from_rank


            to_col = chess.square_file(
                move.to_square
            )

            to_rank = chess.square_rank(
                move.to_square
            )

            to_row = 7 - to_rank


            from_browser_square = (
                from_row * 8
                + from_col
            )

            to_browser_square = (
                to_row * 8
                + to_col
            )


            legal_moves.append({
                "from": from_browser_square,
                "to": to_browser_square
            })


    # Status

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
# HTML
# ============================================================

HTML = """
<!DOCTYPE html>

<html>

<head>

    <meta charset="UTF-8">

    <meta
        name="viewport"
        content="width=device-width, initial-scale=1.0"
    >

    <title>Chess AI</title>


    <style>

        * {
            box-sizing: border-box;
        }


        html,
        body {

            margin: 0;
            padding: 0;

            background: #222;

            color: white;

            font-family: Arial, sans-serif;

            text-align: center;

        }


        body {

            min-height: 100vh;

            padding-top: 15px;

        }


        h1 {

            margin: 10px 0;

        }


        #status {

            font-size: 20px;

            margin: 10px 0 15px;

            min-height: 25px;

        }


        /*
         * IMPORTANT:
         *
         * The board is always a square.
         * Each square is exactly 1/8 of the board.
         */

        #board {

            width: 640px;

            height: 640px;

            max-width: 90vw;

            max-height: 90vw;

            aspect-ratio: 1 / 1;

            margin: 0 auto 20px;

            display: grid;

            grid-template-columns:
                repeat(8, 1fr);

            grid-template-rows:
                repeat(8, 1fr);

            border: 4px solid #111;

        }


        .square {

            width: 100%;

            height: 100%;

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
                rgba(40, 180, 40, 0.75);

        }


        .capture {

            box-shadow:
                inset 0 0 0 6px
                rgba(210, 50, 50, 0.8);

        }


        .piece {

            font-family:
                "DejaVu Sans",
                "Segoe UI Symbol",
                "Noto Sans Symbols 2",
                Arial,
                sans-serif;

            font-size: clamp(
                36px,
                8vw,
                62px
            );

            line-height: 1;

            pointer-events: none;

        }


        .white-piece {

            color: white;

            text-shadow:
                -2px -2px 0 #111,
                 2px -2px 0 #111,
                -2px  2px 0 #111,
                 2px  2px 0 #111;

        }


        .black-piece {

            color: #111;

            text-shadow:
                -2px -2px 0 white,
                 2px -2px 0 white,
                -2px  2px 0 white,
                 2px  2px 0 white;

        }


        button {

            font-size: 18px;

            padding: 10px 22px;

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

        const response =
            await fetch("/board");

        boardData =
            await response.json();

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
        document.getElementById("board");

    boardElement.innerHTML = "";


    for (let row = 0; row < 8; row++) {

        for (let col = 0; col < 8; col++) {

            const square =
                row * 8 + col;


            const div =
                document.createElement("div");


            div.classList.add("square");


            // Board color

            if (
                (row + col) % 2 === 0
            ) {

                div.classList.add("light");

            } else {

                div.classList.add("dark");

            }


            // Selected square

            if (
                square === selectedSquare
            ) {

                div.classList.add("selected");

            }


            // Is this a legal destination?

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

                } else {

                    div.classList.add(
                        "legal"
                    );

                }

            }


            // Get piece

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
                    piece.color === "white"
                ) {

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


                div.appendChild(
                    pieceElement
                );

            }


            // Click handler

            div.addEventListener(
                "click",
                function() {

                    clickSquare(square);

                }
            );


            boardElement.appendChild(div);

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


    // Game over

    if (
        boardData.game_over
    ) {

        return;

    }


    // Only player controls White

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
            Math.floor(square / 8);


        const col =
            square % 8;


        const piece =
            boardData.board[row][col];


        // Only select White pieces

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


    // Clicking selected square again

    if (
        square === selectedSquare
    ) {

        selectedSquare = null;

        drawBoard();

        return;

    }


    // ======================================================
    // SEND MOVE TO PYTHON
    // ======================================================

    try {

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

        boardData = result;

        drawBoard();


        // ==================================================
        // AI MOVE
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


            boardData =
                aiResult;


            drawBoard();

        }

    }

    catch (error) {

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

    return render_template_string(HTML)


@app.route("/board")
def get_board():

    return jsonify(
        board_json()
    )


@app.route("/new_game", methods=["POST"])
def new_game():

    global board

    board = chess.Board()

    return jsonify(
        board_json()
    )


@app.route("/move", methods=["POST"])
def player_move():

    global board


    # ========================================================
    # READ JSON
    # ========================================================

    data = request.get_json()


    if not data:

        result = board_json()

        result["success"] = False

        result["message"] = (
            "No move data received."
        )

        return jsonify(result)


    try:

        from_browser = int(
            data["from"]
        )

        to_browser = int(
            data["to"]
        )

    except (
        KeyError,
        TypeError,
        ValueError
    ):

        result = board_json()

        result["success"] = False

        result["message"] = (
            "Invalid move data."
        )

        return jsonify(result)


    # ========================================================
    # VALIDATE BROWSER SQUARES
    # ========================================================

    if (
        from_browser < 0
        or from_browser > 63
        or to_browser < 0
        or to_browser > 63
    ):

        result = board_json()

        result["success"] = False

        result["message"] = (
            "Invalid board square."
        )

        return jsonify(result)


    # ========================================================
    # PLAYER MUST BE WHITE
    # ========================================================

    if board.turn != chess.WHITE:

        result = board_json()

        result["success"] = False

        result["message"] = (
            "It is the AI's turn."
        )

        return jsonify(result)


    # ========================================================
    # BROWSER -> CHESS COORDINATES
    # ========================================================

    from_row = from_browser // 8
    from_col = from_browser % 8

    to_row = to_browser // 8
    to_col = to_browser % 8


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
    # AUTO-PROMOTE TO QUEEN
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

        result["message"] = (
            "Illegal move."
        )

        return jsonify(result)


    # ========================================================
    # PLAY MOVE
    # ========================================================

    board.push(move)


    result = board_json()

    result["success"] = True

    result["message"] = (
        "Move played."
    )

    return jsonify(result)


# ============================================================
# AI ROUTE
# ============================================================

@app.route("/ai_move", methods=["POST"])
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
# START SERVER
# ============================================================

if __name__ == "__main__":

    print()
    print("==============================")
    print("       BOB AI SERVER")
    print("==============================")
    print()
    print("Open the forwarded port 8000.")
    print()


    app.run(
        host="0.0.0.0",
        port=8000,
        debug=False
    )
