from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import chess
import chess.svg
import tensorflow as tf
import numpy as np
import asyncio
import io
import cairosvg
import base64

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup Jinja2 templates
templates = Jinja2Templates(directory="templates")

app.mount("/img", StaticFiles(directory="img"), name="img")

# Load the model
model = tf.keras.models.load_model('improved_chess_ai_model.h5')

# Chess evaluation functions (copy from the provided code)
def evaluate_board(board):
    if board.is_checkmate():
        return -1000 if board.turn else 1000
    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    material_score = sum(
        len(board.pieces(piece_type, chess.WHITE)) * value -
        len(board.pieces(piece_type, chess.BLACK)) * value
        for piece_type, value in PIECE_VALUES.items()
    )

    white_king_square = board.king(chess.WHITE)
    black_king_square = board.king(chess.BLACK)
    white_king_safety = evaluate_king_safety(board, white_king_square, chess.WHITE)
    black_king_safety = evaluate_king_safety(board, black_king_square, chess.BLACK)

    king_safety_score = white_king_safety - black_king_safety
    protection_score = evaluate_piece_protection(board)

    total_score = (
        material_score * 1.0 +
        king_safety_score * 2.5 +
        protection_score * 0.5
    )

    return total_score if board.turn == chess.WHITE else -total_score

def evaluate_king_safety(board, king_square, color):
    safety_score = 0
    opponent_color = not color

    if board.is_check():
        safety_score -= 5

    pawn_shield_score = sum(1 for sq in chess.SQUARES if
                            chess.square_distance(sq, king_square) <= 2 and
                            board.piece_at(sq) == chess.Piece(chess.PAWN, color))
    safety_score += pawn_shield_score * 0.5

    for piece_type in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
        attackers = board.attackers(opponent_color, king_square)
        safety_score -= len([sq for sq in attackers if board.piece_type_at(sq) == piece_type]) * PIECE_VALUES[piece_type]

    return safety_score

def evaluate_piece_protection(board):
    protection_score = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            attackers = board.attackers(piece.color, square)
            defenders = board.attackers(not piece.color, square)
            protection_score += (len(defenders) - len(attackers)) * PIECE_VALUES[piece.piece_type]
    return protection_score

def is_capture_safe(board, move):
    board.push(move)
    is_safe = evaluate_board(board) >= 0 if board.turn == chess.BLACK else evaluate_board(board) <= 0
    board.pop()
    return is_safe

def predict_move_improved(model, board, move_history):
    # Opening moves
    if board.fullmove_number == 1:
        if board.turn == chess.WHITE:
            return chess.Move.from_uci("e2e4")
        else:
            return chess.Move.from_uci("e7e5")
    elif board.fullmove_number == 2:
        if board.turn == chess.WHITE:
            return chess.Move.from_uci("g1f3")
        else:
            return chess.Move.from_uci("b8c6")

    # Prioritize castling if available
    castling_moves = [move for move in board.legal_moves if board.is_castling(move)]
    if castling_moves:
        return castling_moves[0]

    legal_moves = list(board.legal_moves)
    capture_moves = [move for move in legal_moves if board.is_capture(move)]
    non_capture_moves = [move for move in legal_moves if not board.is_capture(move)]

    best_move = None
    best_score = float('-inf') if board.turn == chess.WHITE else float('inf')

    # Evaluate capture moves first
    for move in capture_moves:
        if is_capture_safe(board, move):
            board.push(move)
            score = evaluate_board(board)
            board.pop()

            if board.turn == chess.WHITE:
                if score > best_score:
                    best_score = score
                    best_move = move
            else:
                if score < best_score:
                    best_score = score
                    best_move = move

    # If no good capture move, evaluate non-capture moves
    if best_move is None:
        for move in non_capture_moves:
            # Avoid move repetition
            if move.uci() in [m.uci() for m in move_history[-4:]]:
                continue

            board.push(move)
            score = evaluate_board(board)
            board.pop()

            if board.turn == chess.WHITE:
                if score > best_score:
                    best_score = score
                    best_move = move
            else:
                if score < best_score:
                    best_score = score
                    best_move = move

    # If still no good move, use model's prediction
    if best_move is None:
        board_planes = board_to_planes(board.fen()).reshape(1, 8, 8, 12)
        meta_features = get_meta_features(board.fen()).reshape(1, 7)
        policy, _ = model.predict([board_planes, meta_features])
        legal_move_indices = [move.from_square * 64 + move.to_square for move in legal_moves]
        legal_move_probs = policy[0][legal_move_indices]
        best_move_index = np.argmax(legal_move_probs)
        best_move = legal_moves[best_move_index]

    return best_move

def get_meta_features(fen):
    board = chess.Board(fen)
    return np.array([
        int(board.has_kingside_castling_rights(chess.WHITE)),
        int(board.has_queenside_castling_rights(chess.WHITE)),
        int(board.has_kingside_castling_rights(chess.BLACK)),
        int(board.has_queenside_castling_rights(chess.BLACK)),
        int(board.has_legal_en_passant()),
        board.halfmove_clock / 100.0,
        board.fullmove_number / 100.0
    ], dtype=np.float32)

def board_to_planes(fen):
    board = chess.Board(fen)
    planes = np.zeros((8, 8, 12), dtype=np.float32)
    piece_dict = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
    }
    for i in range(64):
        piece = board.piece_at(i)
        if piece:
            planes[i // 8, i % 8, piece_dict[piece.symbol()]] = 1
    return planes


PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0
}


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/play")
async def play_game():
    board = chess.Board()
    move_count = 0
    max_moves = 150
    move_history = []
    game_html = []

    while not board.is_game_over() and move_count < max_moves:
        game_html.append(f"<h2>Move {move_count + 1}</h2>")
        
        # Convert SVG to PNG and then to base64
        svg_data = chess.svg.board(board=board).encode('utf-8')
        png_data = cairosvg.svg2png(bytestring=svg_data)
        base64_img = base64.b64encode(png_data).decode('utf-8')
        game_html.append(f'<img src="data:image/png;base64,{base64_img}" alt="Chess board" style="width: 400px;">')

        if board.turn == chess.WHITE:
            ai_move = predict_move_improved(model, board, move_history)
            try:
                board.push(ai_move)
                game_html.append(f"<p>AI move: {ai_move.uci()}</p>")
                move_history.append(ai_move)
            except chess.IllegalMoveError:
                game_html.append(f"<p>AI attempted illegal move: {ai_move.uci()}. Choosing random move.</p>")
                legal_moves = list(board.legal_moves)
                if legal_moves:
                    random_move = np.random.choice(legal_moves)
                    board.push(random_move)
                    game_html.append(f"<p>Random move: {random_move.uci()}</p>")
                    move_history.append(random_move)
                else:
                    game_html.append("<p>No legal moves available. Game over.</p>")
                    break
        else:
            legal_moves = list(board.legal_moves)
            if legal_moves:
                random_move = np.random.choice(legal_moves)
                board.push(random_move)
                game_html.append(f"<p>Random move: {random_move.uci()}</p>")
                move_history.append(random_move)
            else:
                game_html.append("<p>No legal moves available. Game over.</p>")
                break

        move_count += 1
        
        if len(move_history) > 10:
            move_history = move_history[-10:]
        
        # Add a small delay to make the game progression visible
        await asyncio.sleep(0.5)

    game_html.append("<h2>Game over</h2>")
    game_html.append(f"<h2>Result: {board.result()}</h2>")

    return HTMLResponse(content="".join(game_html))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)