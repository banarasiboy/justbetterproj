"""
CodSoft ML Internship - Task 2
Tic-Tac-Toe AI (Minimax with Alpha-Beta Pruning)

Board positions are numbered 1-9 like a phone keypad, left to right,
top to bottom:

 1 | 2 | 3
 4 | 5 | 6
 7 | 8 | 9

The AI ('O') plays optimally. Since Tic-Tac-Toe is a solved game, a
perfect Minimax implementation should never lose - at best the human
can force a draw.
"""

import math

HUMAN = "X"
AI = "O"
EMPTY = " "


def print_board(board):
    print()
    for row in range(3):
        cells = board[row * 3:row * 3 + 3]
        print(f" {cells[0]} | {cells[1]} | {cells[2]} ")
        if row < 2:
            print("---+---+---")
    print()


def available_moves(board):
    return [i for i, cell in enumerate(board) if cell == EMPTY]


def check_winner(board):
    lines = [
        (0, 1, 2), (3, 4, 5), (6, 7, 8),   # rows
        (0, 3, 6), (1, 4, 7), (2, 5, 8),   # columns
        (0, 4, 8), (2, 4, 6),              # diagonals
    ]
    for a, b, c in lines:
        if board[a] != EMPTY and board[a] == board[b] == board[c]:
            return board[a]

    if EMPTY not in board:
        return "DRAW"

    return None


def minimax(board, depth, is_maximizing, alpha, beta):
    """
    is_maximizing == True  -> AI's ('O') turn, trying to maximize score
    is_maximizing == False -> Human's ('X') turn, trying to minimize score

    Score convention: +10 for an AI win, -10 for a human win, 0 for a draw.
    Scores are adjusted slightly by depth so the AI prefers faster wins
    and slower losses.
    """
    result = check_winner(board)
    if result == AI:
        return 10 - depth
    if result == HUMAN:
        return depth - 10
    if result == "DRAW":
        return 0

    if is_maximizing:
        best_score = -math.inf
        for move in available_moves(board):
            board[move] = AI
            score = minimax(board, depth + 1, False, alpha, beta)
            board[move] = EMPTY
            best_score = max(best_score, score)
            alpha = max(alpha, best_score)
            if beta <= alpha:
                break  # prune - human would never allow this branch
        return best_score
    else:
        best_score = math.inf
        for move in available_moves(board):
            board[move] = HUMAN
            score = minimax(board, depth + 1, True, alpha, beta)
            board[move] = EMPTY
            best_score = min(best_score, score)
            beta = min(beta, best_score)
            if beta <= alpha:
                break  # prune - AI would never allow this branch
        return best_score


def best_ai_move(board):
    best_score = -math.inf
    move = None
    for candidate in available_moves(board):
        board[candidate] = AI
        score = minimax(board, 0, False, -math.inf, math.inf)
        board[candidate] = EMPTY
        if score > best_score:
            best_score = score
            move = candidate
    return move


def human_turn(board):
    while True:
        try:
            choice = int(input("Your move (1-9): ")) - 1
        except ValueError:
            print("Please enter a number between 1 and 9.")
            continue

        if choice not in range(9) or board[choice] != EMPTY:
            print("That spot isn't available, try again.")
            continue

        board[choice] = HUMAN
        return


def play():
    board = [EMPTY] * 9
    print("Tic-Tac-Toe: You are X, the AI is O.")
    print("Positions are numbered 1-9, left to right, top to bottom.")
    print_board([str(i + 1) if c == EMPTY else c for i, c in enumerate(board)])

    turn = HUMAN
    while True:
        if turn == HUMAN:
            human_turn(board)
        else:
            print("AI is thinking...")
            move = best_ai_move(board)
            board[move] = AI

        print_board(board)

        result = check_winner(board)
        if result == "DRAW":
            print("It's a draw!")
            break
        elif result in (HUMAN, AI):
            print("You win!" if result == HUMAN else "AI wins!")
            break

        turn = AI if turn == HUMAN else HUMAN


if __name__ == "__main__":
    play()