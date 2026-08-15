from ollama import chat, ResponseError
from pathlib import Path

MODEL = "gemma3:1b"

# Memory paths
MEMORY_FILE = Path("data.txt")
CHESS_MEMORY = Path("Data/chessgame.pgn")

SYSTEM_PROMPT = """
You are a helpful personal AI assistant.

You have access to multiple memory sources.

Use the memories when they are relevant to the user's question.
Do not assume something is true if it is not present in the memories.

The memories can contain personal information, facts, and chess games.
"""


def load_memory():
    """Load the normal memory from data.txt."""
    if not MEMORY_FILE.exists():
        MEMORY_FILE.write_text("", encoding="utf-8")
        return ""

    return MEMORY_FILE.read_text(encoding="utf-8")


def load_chess_memory():
    """Load the chess memory from the PGN file."""
    if not CHESS_MEMORY.exists():
        return ""

    return CHESS_MEMORY.read_text(encoding="utf-8")


def save_memory(text):
    """Add something to the normal memory."""
    with MEMORY_FILE.open("a", encoding="utf-8") as file:
        file.write(text.strip() + "\n")


def clear_memory():
    """Delete all normal memory."""
    MEMORY_FILE.write_text("", encoding="utf-8")


def ask_ai(user_message):
    """Send the user message and both memories to Ollama."""

    memory = load_memory()
    chess_memory = load_chess_memory()

    prompt = f"""
{SYSTEM_PROMPT}

===== PERSONAL MEMORY =====
{memory if memory.strip() else "(No personal memory)"}
===== END PERSONAL MEMORY =====

===== CHESS MEMORY =====
{chess_memory if chess_memory.strip() else "(No chess memory)"}
===== END CHESS MEMORY =====

User:
{user_message}
"""

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

            if user_input == "/exit":
                print("Goodbye!")
                break

            # Save normal memory
            if user_input.startswith("/remember "):
                memory = user_input[len("/remember "):].strip()

                if memory:
                    save_memory(memory)
                    print("AI: Saved to memory.")
                else:
                    print("AI: Nothing to save.")

                continue

            # Show normal memory
            if user_input == "/memory":
                memory = load_memory()

                if memory.strip():
                    print("\n===== PERSONAL MEMORY =====")
                    print(memory)
                    print("===========================\n")
                else:
                    print("AI: Personal memory is empty.")

                continue

            # Show chess memory
            if user_input == "/chess":
                chess_memory = load_chess_memory()

                if chess_memory.strip():
                    print("\n===== CHESS MEMORY =====")
                    print(chess_memory)
                    print("========================\n")
                else:
                    print("AI: Chess memory is empty or the file was not found.")

                continue

            # Clear normal memory
            if user_input == "/clear":
                clear_memory()
                print("AI: Personal memory cleared.")
                continue

            # Normal AI conversation
            print("AI: ", end="", flush=True)
            ask_ai(user_input)

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break

        except ResponseError as error:
            print(f"\nOllama error: {error}")

        except Exception as error:
            print(f"\nError: {error}")


if __name__ == "__main__":
    main()