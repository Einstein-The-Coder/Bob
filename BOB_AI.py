from ollama import chat, ResponseError
from pathlib import Path

MODEL = "gemma3:1b"

# ============================================================
# MEMORY PATHS - KEEPING YOUR EXACT FILES
# ============================================================

MEMORY_FILE = Path("data.txt")
CHESS_MEMORY = Path("Data/chessgame.pgn")


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are a helpful personal AI assistant.

You have access to memory supplied by the user.

IMPORTANT RULES:

- Only use memory that is provided in the current prompt.
- Use memory when it is relevant to the user's question.
- Do not invent memories.
- If the memory directly answers the question, use it.
- Do not discuss unrelated memories.
- If no memory is provided, simply answer normally.
- Keep simple questions concise.
"""


# ============================================================
# MEMORY LOADING
# ============================================================

def load_memory():
    """Load personal memory from data.txt."""

    if not MEMORY_FILE.exists():
        MEMORY_FILE.write_text("", encoding="utf-8")
        return ""

    return MEMORY_FILE.read_text(encoding="utf-8")


def load_chess_memory():
    """Load chess memory from Data/chessgame.pgn."""

    if not CHESS_MEMORY.exists():
        return ""

    return CHESS_MEMORY.read_text(encoding="utf-8")


# ============================================================
# MEMORY SAVING
# ============================================================

def save_memory(text):
    """Add something to personal memory."""

    with MEMORY_FILE.open("a", encoding="utf-8") as file:
        file.write(text.strip() + "\n")


# ============================================================
# MEMORY CLEARING
# ============================================================

def clear_memory():
    """Delete all personal memory."""

    MEMORY_FILE.write_text("", encoding="utf-8")


# ============================================================
# MEMORY ROUTER
# ============================================================

def choose_memory(user_message):
    """
    Decide which memory is relevant.

    Returns:
        "personal"
        "chess"
        "none"
    """

    question = user_message.lower()

    # --------------------------------------------------------
    # Chess-related words
    # --------------------------------------------------------

    chess_keywords = [
        "chess",
        "chess game",
        "chess match",
        "pgn",
        "opening",
        "checkmate",
        "check",
        "king",
        "queen",
        "rook",
        "bishop",
        "knight",
        "pawn",
        "white",
        "black",
        "carlsen",
        "shevchenko",
        "moves",
        "move",
        "game result",
        "who won",
    ]

    for keyword in chess_keywords:
        if keyword in question:
            return "chess"

    # --------------------------------------------------------
    # Personal memory words
    # --------------------------------------------------------

    personal_keywords = [
        "my name",
        "what's my name",
        "what is my name",
        "who am i",
        "about me",
        "my favorite",
        "my favourite",
        "my hobby",
        "my hobbies",
        "my friend",
        "my friends",
        "i like",
        "i love",
        "i hate",
        "i prefer",
        "do you remember me",
        "remember me",
        "what do you know about me",
    ]

    for keyword in personal_keywords:
        if keyword in question:
            return "personal"

    # --------------------------------------------------------
    # Explicit memory requests
    # --------------------------------------------------------

    memory_keywords = [
        "according to my memory",
        "in my memory",
        "from memory",
        "what did i tell you",
        "what have i told you",
        "do you remember",
    ]

    for keyword in memory_keywords:
        if keyword in question:
            return "personal"

    # --------------------------------------------------------
    # No memory needed
    # --------------------------------------------------------

    return "none"


# ============================================================
# BUILD PROMPT
# ============================================================

def build_prompt(user_message):
    """
    Build a prompt using ONLY the memory selected by the router.
    """

    memory_type = choose_memory(user_message)

    # --------------------------------------------------------
    # No memory
    # --------------------------------------------------------

    if memory_type == "none":

        return f"""
{SYSTEM_PROMPT}

No memory is needed for this question.

User:
{user_message}
"""

    # --------------------------------------------------------
    # Personal memory
    # --------------------------------------------------------

    if memory_type == "personal":

        memory = load_memory()

        return f"""
{SYSTEM_PROMPT}

===== PERSONAL MEMORY =====
{memory if memory.strip() else "(Personal memory is empty)"}
===== END PERSONAL MEMORY =====

User:
{user_message}
"""

    # --------------------------------------------------------
    # Chess memory
    # --------------------------------------------------------

    if memory_type == "chess":

        chess_memory = load_chess_memory()

        return f"""
{SYSTEM_PROMPT}

===== CHESS MEMORY =====
{chess_memory if chess_memory.strip() else "(Chess memory is empty)"}
===== END CHESS MEMORY =====

User:
{user_message}
"""

    return f"""
{SYSTEM_PROMPT}

User:
{user_message}
"""


# ============================================================
# ASK OLLAMA
# ============================================================

def ask_ai(user_message):
    """
    Send the question to Ollama using only relevant memory.
    """

    memory_type = choose_memory(user_message)

    # Useful for testing the memory router.
    if memory_type == "personal":
        print("[Using personal memory]")

    elif memory_type == "chess":
        print("[Using chess memory]")

    else:
        print("[No memory needed]")

    prompt = build_prompt(user_message)

    response = chat(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        stream=True
    )

    answer = ""

    for chunk in response:

        text = chunk.message.content

        print(text, end="", flush=True)

        answer += text

    print()

    return answer


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    print("================================")
    print("       Local Ollama AI")
    print("================================")
    print("Model:", MODEL)
    print()

    print("Memory files:")
    print("  Personal:", MEMORY_FILE)
    print("  Chess:", CHESS_MEMORY)
    print()

    print("Commands:")
    print("  /remember <text>  - Save something to memory")
    print("  /memory           - Show personal memory")
    print("  /chess            - Show chess memory")
    print("  /clear            - Clear personal memory")
    print("  /exit             - Exit")
    print()

    while True:

        try:

            user_input = input("You: ").strip()

            if not user_input:
                continue

            # ------------------------------------------------
            # EXIT
            # ------------------------------------------------

            if user_input == "/exit":

                print("Goodbye!")

                break

            # ------------------------------------------------
            # REMEMBER
            # ------------------------------------------------

            if user_input.startswith("/remember "):

                memory = user_input[len("/remember "):].strip()

                if memory:

                    save_memory(memory)

                    print("AI: Saved to memory.")

                else:

                    print("AI: Nothing to save.")

                continue

            # ------------------------------------------------
            # SHOW PERSONAL MEMORY
            # ------------------------------------------------

            if user_input == "/memory":

                memory = load_memory()

                if memory.strip():

                    print("\n===== PERSONAL MEMORY =====")
                    print(memory)
                    print("===========================\n")

                else:

                    print("AI: Personal memory is empty.")

                continue

            # ------------------------------------------------
            # SHOW CHESS MEMORY
            # ------------------------------------------------

            if user_input == "/chess":

                chess_memory = load_chess_memory()

                if chess_memory.strip():

                    print("\n===== CHESS MEMORY =====")
                    print(chess_memory)
                    print("========================\n")

                else:

                    print(
                        "AI: Chess memory is empty "
                        "or the file was not found."
                    )

                continue

            # ------------------------------------------------
            # CLEAR PERSONAL MEMORY
            # ------------------------------------------------

            if user_input == "/clear":

                clear_memory()

                print("AI: Personal memory cleared.")

                continue

            # ------------------------------------------------
            # NORMAL AI
            # ------------------------------------------------

            print("AI: ", end="", flush=True)

            ask_ai(user_input)

        # ----------------------------------------------------
        # CTRL+C
        # ----------------------------------------------------

        except KeyboardInterrupt:

            print("\nGoodbye!")

            break

        # ----------------------------------------------------
        # OLLAMA ERROR
        # ----------------------------------------------------

        except ResponseError as error:

            print(f"\nOllama error: {error}")

        # ----------------------------------------------------
        # OTHER ERROR
        # ----------------------------------------------------

        except Exception as error:

            print(f"\nError: {error}")


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    main()