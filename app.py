from flask import Flask, render_template_string, request, jsonify
import chess
import torch

from model import ChessNet
from utils import board_to_tensor, move_to_index


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", DEVICE)


# ============================================================
# LOAD MODEL
# ============================================================

model = ChessNet()

MODEL_PATH = "chess_model_weights.pth"

try:

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    # Support either:
    #
    # torch.save(model.state_dict(), ...)
    #
    # or:
    #
    # torch.save({"model_state_dict": ...}, ...)

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:

            model.load_state_dict(
                checkpoint["model_state_dict"]
            )

        else:

            model.load_state_dict(
                checkpoint
            )

    else:

        raise RuntimeError(
            "Unsupported model checkpoint format."
        )

    print(
        f"Loaded trained AI from {MODEL_PATH}"
    )

except FileNotFoundError:

    print()
    print(
        "WARNING: chess_model_weights.pth was not found."
    )

    print(
        "The AI will use random weights."
    )

    print(
        "Train the model before expecting useful play."
    )

    print()


except RuntimeError as e:

    print()
    print("ERROR: Could not load model weights.")
    print(e)
    print()
    print(
        "This usually means the checkpoint was trained "
        "with a different model architecture."
    )
    print()


model.to(DEVICE)
model.eval()


# ============================================================
# GAME STATE
# ============================================================

board = chess.Board()

# Player color:
#
# chess.WHITE
# or
# chess.BLACK

player_color = chess.WHITE


# ============================================================
# PIECE LETTERS
# ============================================================

# We deliberately DO NOT use Unicode chess pieces.

PIECES = {
    "P": "P",
    "N": "N",
    "B": "B",
    "R": "R",
    "Q": "Q",
    "K": "K",

    "p": "p",
    "n": "n",
    "b": "b",
    "r": "r",
    "q": "q",
    "k": "k"
}


# ============================================================
# GAME STATUS
# ============================================================

def get_status():

    if board.is_checkmate():

        winner = (
            "White"
            if board.turn == chess.BLACK
            else "Black"
        )

        return f"Checkmate! {winner} wins."

    if board.is_stalemate():

        return "Draw by stalemate."

    if board.is_insufficient_material():

        return "Draw by insufficient material."

    if board.is_fivefold_repetition():

        return "Draw by fivefold repetition."

    if board.is_seventyfive_moves():

        return "Draw by 75-move rule."

    if board.is_check():

        if board.turn == player_color:

            return "Check! Your turn."

        return "Check! AI is thinking..."

    if board.turn == player_color:

        return "Your turn."

    return "AI's turn."


# ============================================================
# BOARD JSON
# ============================================================

def board_json(success=None, message=None):

    board_array = []

    # --------------------------------------------------------
    # Board orientation
    # --------------------------------------------------------
    #
    # If player is White:
    #
    #   White pieces appear at bottom.
    #
    # If player is Black:
    #
    #   Black pieces appear at bottom.
    #
    # This makes the board feel like a real chess board
    # from the player's perspective.
    #

    if player_color == chess.WHITE:

        ranks = range(7, -1, -1)
        files = range(0, 8)

    else:

        ranks = range(0, 8)
        files = range(7, -1, -1)


    for rank in ranks:

        row_data = []

        for file in files:

            square = chess.square(
                file,
                rank
            )

            piece = board.piece_at(square)

            if piece is None:

                row_data.append({
                    "symbol": "",
                    "color": ""
                })

            else:

                row_data.append({
                    "symbol": PIECES[
                        piece.symbol()
                    ],
                    "color": (
                        "white"
                        if piece.color == chess.WHITE
                        else "black"
                    )
                })

        board_array.append(row_data)


    # --------------------------------------------------------
    # Legal moves
    # --------------------------------------------------------

    legal_moves = []

    if board.turn == player_color:

        for move in board.legal_moves:

            # Send the actual destination square.
            #
            # JavaScript converts this to the displayed
            # board coordinate system.

            legal_moves.append({
                "from": move.from_square,
                "to": move.to_square
            })


    result = {
        "board": board_array,

        "legal_moves": legal_moves,

        "status": get_status(),

        "game_over": board.is_game_over(),

        "player_color": (
            "white"
            if player_color == chess.WHITE
            else "black"
        ),

        "turn": (
            "white"
            if board.turn == chess.WHITE
            else "black"
        )
    }


    if success is not None:

        result["success"] = success


    if message is not None:

        result["message"] = message


    return result


# ============================================================
# AI MOVE
# ============================================================

def get_ai_move():

    """
    Select the highest-scoring legal move from the network.

    The network produces 4096 policy scores.

    We NEVER allow the neural network to make an illegal move.
    """

    if board.is_game_over():

        return None


    if board.turn != (
        chess.BLACK
        if player_color == chess.WHITE
        else chess.WHITE
    ):

        return None


    # --------------------------------------------------------
    # Convert board to tensor
    # --------------------------------------------------------

    tensor = board_to_tensor(
        board
    )


    tensor = tensor.unsqueeze(0)


    tensor = tensor.to(
        DEVICE
    )


    # --------------------------------------------------------
    # Neural network
    # --------------------------------------------------------

    with torch.no_grad():

        policy, value = model(
            tensor
        )


    # policy shape:
    #
    # [1, 4096]

    scores = policy[0]


    # --------------------------------------------------------
    # Find best LEGAL move
    # --------------------------------------------------------

    legal_moves = list(
        board.legal_moves
    )


    if not legal_moves:

        return None


    best_move = None

    best_score = float("-inf")


    for move in legal_moves:

        index = move_to_index(
            move
        )

        score = scores[
            index
        ].item()


        if score > best_score:

            best_score = score

            best_move = move


    return best_move


# ============================================================
# HTML
# ============================================================

HTML = r"""
<!DOCTYPE html>

<html>

<head>

<meta charset="UTF-8">

<title>Chess AI</title>


<style>

/* =========================================================
   PAGE
   ========================================================= */

* {
    box-sizing: border-box;
}

body {

    margin: 0;

    min-height: 100vh;

    background:
        radial-gradient(
            circle at top,
            #333,
            #151515 70%
        );

    color: white;

    font-family:
        Arial,
        Helvetica,
        sans-serif;

    text-align: center;
}


/* =========================================================
   TITLE
   ========================================================= */

h1 {

    margin: 25px 0 5px;

    font-size: 36px;

    letter-spacing: 1px;
}


/* =========================================================
   STATUS
   ========================================================= */

#status {

    min-height: 28px;

    margin: 10px;

    font-size: 19px;

    color: #eee;
}


/* =========================================================
   GAME INFO
   ========================================================= */

#info {

    margin-bottom: 10px;

    color: #bbb;

    font-size: 15px;
}


/* =========================================================
   BOARD
   ========================================================= */

#board {

    width: min(
        640px,
        90vw
    );

    height: min(
        640px,
        90vw
    );

    margin: 20px auto;

    display: grid;

    grid-template-columns:
        repeat(8, 1fr);

    grid-template-rows:
        repeat(8, 1fr);

    border: 5px solid #111;

    box-shadow:
        0 15px 40px
        rgba(0, 0, 0, 0.5);
}


/* =========================================================
   SQUARE
   ========================================================= */

.square {

    position: relative;

    display: flex;

    align-items: center;

    justify-content: center;

    cursor: pointer;

    user-select: none;

    font-size: clamp(
        30px,
        8vw,
        58px
    );

    font-weight: bold;

    transition:
        filter 0.08s ease;
}


.square:hover {

    filter: brightness(1.12);
}


.light {

    background: #f0d9b5;
}


.dark {

    background: #b58863;
}


/* =========================================================
   PIECES
   ========================================================= */

.white-piece {

    color: #ffffff;

    text-shadow:
        2px 2px 0 #111,
        -1px -1px 0 #111,
        1px -1px 0 #111,
        -1px 1px 0 #111;
}


.black-piece {

    color: #111111;

    text-shadow:
        1px 1px 0 #555;
}


/* =========================================================
   SELECTED
   ========================================================= */

.selected {

    background:
        #f6f669 !important;
}


/* =========================================================
   LEGAL MOVE
   ========================================================= */

.legal {

    box-shadow:
        inset 0 0 0 5px
        rgba(40, 190, 60, 0.75);
}


/* =========================================================
   BUTTON
   ========================================================= */

button {

    border: none;

    border-radius: 8px;

    background: #4caf50;

    color: white;

    font-size: 17px;

    padding: 11px 24px;

    cursor: pointer;

    margin: 5px;

    transition:
        background 0.15s ease;
}


button:hover {

    background: #43a047;
}


button:disabled {

    background: #555;

    cursor: not-allowed;
}


/* =========================================================
   PROMOTION
   ========================================================= */

#promotion {

    display: none;

    position: fixed;

    inset: 0;

    background:
        rgba(0, 0, 0, 0.7);

    align-items: center;

    justify-content: center;

    z-index: 10;
}


#promotion-box {

    background: #222;

    padding: 25px;

    border-radius: 12px;

    box-shadow:
        0 15px 40px
        rgba(0, 0, 0, 0.7);
}


#promotion-box h2 {

    margin-top: 0;
}


.promotion-button {

    background: #555;

    margin: 5px;

    min-width: 70px;
}


.promotion-button:hover {

    background: #666;
}


</style>

</head>


<body>


<h1>Chess AI</h1>


<div id="status">
    Loading...
</div>


<div id="info">
    Loading game...
</div>


<div id="board"></div>


<button
    id="new-game-button"
    onclick="newGame()"
>
    New Game
</button>


<!-- =======================================================
     PROMOTION DIALOG
     ======================================================= -->

<div id="promotion">

    <div id="promotion-box">

        <h2>
            Choose promotion
        </h2>

        <button
            class="promotion-button"
            onclick="choosePromotion('q')"
        >
            Queen
        </button>

        <button
            class="promotion-button"
            onclick="choosePromotion('r')"
        >
            Rook
        </button>

        <button
            class="promotion-button"
            onclick="choosePromotion('b')"
        >
            Bishop
        </button>

        <button
            class="promotion-button"
            onclick="choosePromotion('n')"
        >
            Knight
        </button>

    </div>

</div>


<script>


// ==========================================================
// STATE
// ==========================================================

let boardData = null;

let selectedSquare = null;

let displayedSelectedSquare = null;

let waitingForAI = false;

let pendingPromotion = null;


// ==========================================================
// GET BOARD
// ==========================================================

async function getBoard() {

    try {

        const response =
            await fetch("/board");

        if (!response.ok) {

            throw new Error(
                "Server returned " +
                response.status
            );
        }


        boardData =
            await response.json();


        drawBoard();


    } catch (error) {

        console.error(error);

        document.getElementById(
            "status"
        ).innerText =
            "Could not connect to server.";
    }
}


// ==========================================================
// CONVERT DISPLAY SQUARE -> CHESS SQUARE
// ==========================================================

function displayToChessSquare(
    displayIndex
) {

    const row =
        Math.floor(
            displayIndex / 8
        );

    const col =
        displayIndex % 8;


    if (
        boardData.player_color ===
        "white"
    ) {

        const rank =
            7 - row;

        const file =
            col;

        return rank * 8 + file;

    } else {

        const rank =
            row;

        const file =
            7 - col;

        return rank * 8 + file;
    }
}


// ==========================================================
// CHESS SQUARE -> DISPLAY SQUARE
// ==========================================================

function chessToDisplaySquare(
    chessSquare
) {

    const rank =
        Math.floor(
            chessSquare / 8
        );

    const file =
        chessSquare % 8;


    let row;
    let col;


    if (
        boardData.player_color ===
        "white"
    ) {

        row = 7 - rank;

        col = file;

    } else {

        row = rank;

        col = 7 - file;
    }


    return row * 8 + col;
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


    // ------------------------------------------------------
    // Legal destinations
    // ------------------------------------------------------

    const legalTargets =
        new Set();


    if (
        boardData &&
        boardData.legal_moves
    ) {

        for (
            const move
            of boardData.legal_moves
        ) {

            if (
                selectedSquare ===
                move.from
            ) {

                legalTargets.add(
                    move.to
                );
            }
        }
    }


    // ------------------------------------------------------
    // Create squares
    // ------------------------------------------------------

    for (
        let displayRow = 0;
        displayRow < 8;
        displayRow++
    ) {

        for (
            let displayCol = 0;
            displayCol < 8;
            displayCol++
        ) {

            const displayIndex =
                displayRow * 8 +
                displayCol;


            const chessSquare =
                displayToChessSquare(
                    displayIndex
                );


            const div =
                document.createElement(
                    "div"
                );


            div.classList.add(
                "square"
            );


            if (
                (displayRow +
                 displayCol) % 2 === 0
            ) {

                div.classList.add(
                    "light"
                );

            } else {

                div.classList.add(
                    "dark"
                );
            }


            // Selected square

            if (
                chessSquare ===
                selectedSquare
            ) {

                div.classList.add(
                    "selected"
                );
            }


            // Legal destination

            if (
                legalTargets.has(
                    chessSquare
                )
            ) {

                div.classList.add(
                    "legal"
                );
            }


            // Piece

            const piece =
                boardData.board[
                    displayRow
                ][
                    displayCol
                ];


            if (piece &&
                piece.symbol) {

                div.innerText =
                    piece.symbol;


                if (
                    piece.color ===
                    "white"
                ) {

                    div.classList.add(
                        "white-piece"
                    );

                } else {

                    div.classList.add(
                        "black-piece"
                    );
                }
            }


            div.onclick =
                function() {

                    clickSquare(
                        chessSquare
                    );
                };


            boardElement.appendChild(
                div
            );
        }
    }


    // ------------------------------------------------------
    // Status
    // ------------------------------------------------------

    document.getElementById(
        "status"
    ).innerText =
        boardData.status;


    document.getElementById(
        "info"
    ).innerText =
        "You are playing " +
        (
            boardData.player_color ===
            "white"
                ? "White"
                : "Black"
        );


    document.getElementById(
        "new-game-button"
    ).disabled =
        waitingForAI;
}


// ==========================================================
// CLICK SQUARE
// ==========================================================

async function clickSquare(
    square
) {

    // Don't allow clicks while AI is moving

    if (waitingForAI) {

        return;
    }


    // Game over

    if (
        boardData.game_over
    ) {

        return;
    }


    // Must be player's turn

    if (
        boardData.turn !==
        boardData.player_color
    ) {

        return;
    }


    // ------------------------------------------------------
    // First click
    // ------------------------------------------------------

    if (
        selectedSquare === null
    ) {

        const piece =
            getPieceAtChessSquare(
                square
            );


        if (!piece) {

            return;
        }


        if (
            piece.color !==
            boardData.player_color
        ) {

            return;
        }


        // Only select if the piece has
        // at least one legal move.

        const canMove =
            boardData.legal_moves.some(
                move =>
                    move.from === square
            );


        if (!canMove) {

            return;
        }


        selectedSquare =
            square;


        drawBoard();

        return;
    }


    // ------------------------------------------------------
    // Clicking same square
    // ------------------------------------------------------

    if (
        selectedSquare === square
    ) {

        selectedSquare = null;

        drawBoard();

        return;
    }


    // ------------------------------------------------------
    // Check if destination is legal
    // ------------------------------------------------------

    const possibleMoves =
        boardData.legal_moves.filter(
            move =>
                move.from ===
                    selectedSquare
                &&
                move.to === square
        );


    if (
        possibleMoves.length === 0
    ) {

        // If clicked another friendly piece,
        // select that instead.

        const piece =
            getPieceAtChessSquare(
                square
            );


        if (
            piece &&
            piece.color ===
            boardData.player_color
        ) {

            const canMove =
                boardData.legal_moves.some(
                    move =>
                        move.from ===
                        square
                );


            if (canMove) {

                selectedSquare =
                    square;

                drawBoard();

                return;
            }
        }


        return;
    }


    // ------------------------------------------------------
    // Promotion
    // ------------------------------------------------------

    const piece =
        getPieceAtChessSquare(
            selectedSquare
        );


    if (
        piece &&
        piece.symbol.toUpperCase() ===
        "P"
    ) {

        const targetRank =
            Math.floor(
                square / 8
            );


        if (
            targetRank === 0 ||
            targetRank === 7
        ) {

            pendingPromotion = {

                from: selectedSquare,

                to: square
            };


            showPromotion();

            return;
        }
    }


    await sendMove(
        selectedSquare,
        square,
        null
    );
}


// ==========================================================
// GET PIECE
// ==========================================================

function getPieceAtChessSquare(
    chessSquare
) {

    const displaySquare =
        chessToDisplaySquare(
            chessSquare
        );


    const row =
        Math.floor(
            displaySquare / 8
        );

    const col =
        displaySquare % 8;


    return boardData.board[
        row
    ][
        col
    ];
}


// ==========================================================
// SEND MOVE
// ==========================================================

async function sendMove(
    from,
    to,
    promotion
) {

    selectedSquare = null;

    waitingForAI = true;

    drawBoard();


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

                    body:
                        JSON.stringify({
                            from: from,
                            to: to,
                            promotion:
                                promotion
                        })
                }
            );


        const result =
            await response.json();


        if (!response.ok ||
            !result.success) {

            alert(
                result.message ||
                "Illegal move."
            );

            boardData =
                result.board
                    ? result
                    : boardData;

            waitingForAI = false;

            drawBoard();

            return;
        }


        boardData = result;

        drawBoard();


        // --------------------------------------------------
        // AI TURN
        // --------------------------------------------------

        if (
            !result.game_over &&
            result.turn !==
            result.player_color
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


            if (
                !aiResponse.ok ||
                aiResult.success === false
            ) {

                console.error(
                    aiResult
                );


                document.getElementById(
                    "status"
                ).innerText =
                    aiResult.message ||
                    "AI error.";

                return;
            }


            boardData =
                aiResult;


            drawBoard();
        }


    } catch (error) {

        console.error(
            "Move error:",
            error
        );


        document.getElementById(
            "status"
        ).innerText =
            "Connection error.";
    }


    waitingForAI = false;

    drawBoard();
}


// ==========================================================
// PROMOTION UI
// ==========================================================

function showPromotion() {

    document.getElementById(
        "promotion"
    ).style.display =
        "flex";
}


// ==========================================================
// CHOOSE PROMOTION
// ==========================================================

async function choosePromotion(
    piece
) {

    document.getElementById(
        "promotion"
    ).style.display =
        "none";


    if (
        pendingPromotion === null
    ) {

        return;
    }


    const from =
        pendingPromotion.from;

    const to =
        pendingPromotion.to;


    pendingPromotion = null;


    await sendMove(
        from,
        to,
        piece
    );
}


// ==========================================================
// NEW GAME
// ==========================================================

async function newGame() {

    if (waitingForAI) {

        return;
    }


    try {

        const response =
            await fetch(
                "/new_game",
                {
                    method: "POST"
                }
            );


        if (!response.ok) {

            throw new Error(
                "New game failed."
            );
        }


        boardData =
            await response.json();


        selectedSquare = null;

        waitingForAI = false;


        drawBoard();


        // --------------------------------------------------
        // If player is Black, AI goes first.
        // --------------------------------------------------

        if (
            boardData.player_color ===
                "black"
            &&
            !boardData.game_over
        ) {

            await makeAIFirstMove();
        }


    } catch (error) {

        console.error(error);

        document.getElementById(
            "status"
        ).innerText =
            "Could not start new game.";
    }
}


// ==========================================================
// AI FIRST MOVE
// ==========================================================

async function makeAIFirstMove() {

    waitingForAI = true;

    document.getElementById(
        "status"
    ).innerText =
        "AI is thinking...";


    drawBoard();


    try {

        const response =
            await fetch(
                "/ai_move",
                {
                    method: "POST"
                }
            );


        const result =
            await response.json();


        if (
            !response.ok ||
            result.success === false
        ) {

            throw new Error(
                result.message ||
                "AI failed."
            );
        }


        boardData = result;

        drawBoard();


    } catch (error) {

        console.error(error);

        document.getElementById(
            "status"
        ).innerText =
            "AI error. Check the server console.";
    }


    waitingForAI = false;

    drawBoard();
}


// ==========================================================
// INITIAL LOAD
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
    global player_color


    # New board

    board = chess.Board()


    # Randomly choose player's color

    if torch.rand(1).item() < 0.5:

        player_color = chess.WHITE

    else:

        player_color = chess.BLACK


    result = board_json(
        success=True
    )


    return jsonify(
        result
    )


# ============================================================
# PLAYER MOVE
# ============================================================

@app.route(
    "/move",
    methods=["POST"]
)
def player_move():

    try:

        data = request.get_json(
            silent=True
        )


        if data is None:

            return jsonify({
                "success": False,
                "message":
                    "Invalid JSON."
            }), 400


        # ----------------------------------------------------
        # Validate input
        # ----------------------------------------------------

        if "from" not in data:

            return jsonify({
                "success": False,
                "message":
                    "Missing 'from' square."
            }), 400


        if "to" not in data:

            return jsonify({
                "success": False,
                "message":
                    "Missing 'to' square."
            }), 400


        from_square = int(
            data["from"]
        )

        to_square = int(
            data["to"]
        )


        if not (
            0 <= from_square < 64
            and
            0 <= to_square < 64
        ):

            return jsonify({
                "success": False,
                "message":
                    "Invalid square."
            }), 400


        # ----------------------------------------------------
        # Verify player's turn
        # ----------------------------------------------------

        if board.turn != player_color:

            return jsonify({
                "success": False,
                "message":
                    "It is not your turn."
            })


        if board.is_game_over():

            return jsonify({
                "success": False,
                "message":
                    "The game is already over."
            })


        # ----------------------------------------------------
        # Convert square indexes
        # ----------------------------------------------------

        from_chess_square = (
            from_square
        )

        to_chess_square = (
            to_square
        )


        # ----------------------------------------------------
        # Promotion
        # ----------------------------------------------------

        promotion = data.get(
            "promotion"
        )


        promotion_piece = None


        if promotion:

            promotion = str(
                promotion
            ).lower()


            promotion_map = {

                "q": chess.QUEEN,

                "r": chess.ROOK,

                "b": chess.BISHOP,

                "n": chess.KNIGHT
            }


            if promotion not in promotion_map:

                return jsonify({
                    "success": False,
                    "message":
                        "Invalid promotion piece."
                }), 400


            promotion_piece = promotion_map[promotion]


        # ----------------------------------------------------
        # Create move
        # ----------------------------------------------------

        move = chess.Move(
            from_chess_square,
            to_chess_square,
            promotion=promotion_piece
        )


        # ----------------------------------------------------
        # If pawn reaches last rank and no promotion was
        # supplied, automatically promote to queen.
        # ----------------------------------------------------

        piece = board.piece_at(
            from_chess_square
        )


        if (
            piece is not None
            and
            piece.piece_type ==
                chess.PAWN
            and
            chess.square_rank(
                to_chess_square
            ) in (0, 7)
            and
            promotion_piece is None
        ):

            move = chess.Move(
                from_chess_square,
                to_chess_square,
                promotion=chess.QUEEN
            )


        # ----------------------------------------------------
        # Verify legality
        # ----------------------------------------------------

        if move not in board.legal_moves:

            return jsonify({
                "success": False,
                "message":
                    "Illegal move."
            })


        # ----------------------------------------------------
        # Make move
        # ----------------------------------------------------

        board.push(move)


        result = board_json(
            success=True
        )


        return jsonify(
            result
        )


    except Exception as e:

        print(
            "PLAYER MOVE ERROR:",
            repr(e)
        )


        return jsonify({
            "success": False,
            "message":
                "Server error: " +
                str(e)
        }), 500


# ============================================================
# AI MOVE
# ============================================================

@app.route(
    "/ai_move",
    methods=["POST"]
)
def ai_move():

    try:

        # ----------------------------------------------------
        # Game over
        # ----------------------------------------------------

        if board.is_game_over():

            return jsonify(
                board_json(
                    success=True
                )
            )


        # ----------------------------------------------------
        # Verify AI turn
        # ----------------------------------------------------

        if board.turn == player_color:

            return jsonify({
                **board_json(
                    success=False
                ),
                "message":
                    "It is the player's turn."
            })


        # ----------------------------------------------------
        # Get AI move
        # ----------------------------------------------------

        move = get_ai_move()


        if move is None:

            return jsonify({
                **board_json(
                    success=False
                ),
                "message":
                    "AI could not find a legal move."
            })


        print(
            "AI move:",
            board.san(move)
        )


        # ----------------------------------------------------
        # Make AI move
        # ----------------------------------------------------

        board.push(move)


        return jsonify(
            board_json(
                success=True
            )
        )


    except Exception as e:

        print()
        print(
            "=============================="
        )
        print(
            "          AI ERROR"
        )
        print(
            "=============================="
        )
        print(
            repr(e)
        )
        print(
            "=============================="
        )
        print()


        return jsonify({
            **board_json(
                success=False
            ),
            "message":
                "AI error: " +
                str(e)
        }), 500


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print()
    print("==============================")
    print("        CHESS AI SERVER")
    print("==============================")
    print()
    print(
        "Device:",
        DEVICE
    )
    print()
    print(
        "Board input: 18 channels"
    )
    print(
        "Policy output: 4096 moves"
    )
    print(
        "Random player color: enabled"
    )
    print()
    print(
        "Open:"
    )
    print(
        "http://localhost:8000"
    )
    print()
    print("==============================")
    print()


    # Start with White by default for the first game.
    # The New Game button will randomly choose a color.

    player_color = chess.WHITE


    app.run(
        host="0.0.0.0",
        port=8000,
        debug=False
    )