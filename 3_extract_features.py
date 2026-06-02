import chess
import chess.pgn
import chess.engine
import pandas as pd
import io
import os
import multiprocessing
import atexit
import math
from tqdm import tqdm

# ==========================================
# 1. CẤU HÌNH ĐƯỜNG DẪN & THAM SỐ
# ==========================================
STOCKFISH_PATH = r"D:\stockfish\stockfish-windows-x86-64-avx2.exe"
CHEAT_CSV = r"D:\Downloads\Games_2000.csv"
CLEAN_PGN = r"D:\Downloads\Clean_2000.pgn"
OUTPUT_CSV = "dataset_ml_depth12.csv"

# TỈ LỆ VÀNG: Ép tính tới Depth 12. Nếu thế cờ quá khó, ngắt rụp ở 0.2 giây để tránh treo máy.
TIME_LIMIT = 0.2
DEPTH_LIMIT = 12
MULTIPV = 3

# ==========================================
# 2. CÁC HÀM TOÁN HỌC THỐNG KÊ
# ==========================================
def calc_shannon_entropy(legal_moves_count, board):
    if legal_moves_count <= 0: return 0.0
    probabilities = []
    total_weight = 0
    for move in board.legal_moves:
        piece = board.piece_at(move.from_square)
        weight = 1 if piece is None else piece.piece_type
        probabilities.append(weight)
        total_weight += weight
        
    if total_weight == 0: return 0.0
    entropy = sum(-(w/total_weight) * math.log2(w/total_weight) for w in probabilities if w > 0)
    max_entropy = math.log2(max(1, legal_moves_count))
    return round(min(1.0, max(0.0, entropy / max_entropy)), 4) if max_entropy > 0 else 0.0

def calc_tree_complexity(branching_factor):
    if branching_factor <= 0: return 0.0
    try:
        raw = math.pow(branching_factor, 10) # Chiều sâu giả định
        return round(math.log10(raw), 4) if raw > 0 else 0.0
    except OverflowError:
        return 30.0

def calc_std_dev(data_list, mean_val):
    if len(data_list) <= 1: return 0.0
    variance = sum((x - mean_val) ** 2 for x in data_list) / (len(data_list) - 1)
    return round(math.sqrt(variance), 4)

# ==========================================
# 3. HÀM PHỤ TRỢ (CHẠY TRÊN TỪNG NHÂN CPU)
# ==========================================
worker_engine = None

def init_worker(engine_path):
    global worker_engine
    worker_engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    # Tối ưu RAM cho đa luồng
    worker_engine.configure({"Threads": 1, "Hash": 64})
    atexit.register(worker_engine.quit)

def extract_features(game, elo, opp_elo, label, target_color):
    board = game.board()
    move_count, total_cp_loss = 0, 0
    blunders, mistakes = 0, 0
    best_move_matches, top3_matches = 0, 0
    
    cpl_array, entropy_array, complexity_array = [], [], []
    phase_losses = {'opening': [], 'middlegame': [], 'endgame': []}
    
    # DUYỆT TỪ NƯỚC ĐI ĐẦU TIÊN
    for node in game.mainline():
        turn_color = board.turn
        played_move = node.move
            
        if turn_color == target_color:
            # 1. Đặc trưng Toán học
            bf = board.legal_moves.count()
            entropy_array.append(calc_shannon_entropy(bf, board))
            complexity_array.append(calc_tree_complexity(bf))
            
            # 2. Phân tích Centipawn với Cơ chế Cân bằng (Cái nào đến trước thì ngắt)
            limit = chess.engine.Limit(time=TIME_LIMIT, depth=DEPTH_LIMIT)
            info_before = worker_engine.analyse(board, limit, multipv=MULTIPV)
            
            # Đếm tỷ lệ khớp nước đi
            top_moves = [pv.get('pv')[0] for pv in info_before if 'pv' in pv and len(pv.get('pv')) > 0]
            if top_moves:
                if played_move == top_moves[0]: best_move_matches += 1
                if played_move in top_moves[:3]: top3_matches += 1
            
            best_score = info_before[0]['score'].pov(target_color).score(mate_score=10000) or 0
            
            # Đẩy nước đi vào bàn cờ thực
            board.push(played_move)
            
            # Phân tích lại sau khi đi
            info_after = worker_engine.analyse(board, limit)
            score_after = info_after['score'].pov(target_color).score(mate_score=10000) or 0
            
            # Tính Centipawn Loss (CPL)
            cp_loss = max(0, min(best_score - score_after, 1000))
            cpl_array.append(cp_loss)
            total_cp_loss += cp_loss
            move_count += 1
            
            if cp_loss > 200: blunders += 1
            elif 50 < cp_loss <= 200: mistakes += 1
            
            # Ghi nhận CPL theo từng giai đoạn
            fullmove = board.fullmove_number
            if fullmove <= 10: phase_losses['opening'].append(cp_loss)
            elif 11 <= fullmove <= 40: phase_losses['middlegame'].append(cp_loss)
            else: phase_losses['endgame'].append(cp_loss)
            
        else:
            board.push(played_move)
            
    if move_count == 0: return None
    
    mean_cpl = round(total_cp_loss / move_count, 1)
    safe_mean = lambda lst: round(sum(lst) / len(lst), 1) if lst else 0.0
    
    return {
        "elo": elo, 
        "elo_diff": elo - opp_elo, 
        "move_count": move_count,
        "acpl": mean_cpl,
        "cpl_std": calc_std_dev(cpl_array, mean_cpl),
        "best_move_rate": round(best_move_matches / move_count, 2),
        "top3_rate": round(top3_matches / move_count, 2),
        "blunder_rate": round(blunders / move_count, 2),
        "mistake_rate": round(mistakes / move_count, 2),
        "opening_acpl": safe_mean(phase_losses['opening']),
        "middlegame_acpl": safe_mean(phase_losses['middlegame']),
        "endgame_acpl": safe_mean(phase_losses['endgame']),
        "avg_entropy": round(sum(entropy_array) / len(entropy_array), 4) if entropy_array else 0.0,
        "avg_complexity": round(sum(complexity_array) / len(complexity_array), 4) if complexity_array else 0.0,
        "label": label
    }

def process_single_game(task):
    game = chess.pgn.read_game(io.StringIO(task['pgn']))
    if not game: return []
    
    results = []
    feat_w = extract_features(game, task['elo_w'], task['elo_b'], task['label_w'], chess.WHITE)
    if feat_w: results.append(feat_w)
        
    feat_b = extract_features(game, task['elo_b'], task['elo_w'], task['label_b'], chess.BLACK)
    if feat_b: results.append(feat_b)
        
    return results

# ==========================================
# 4. LUỒNG CHẠY CHÍNH
# ==========================================
def main():
    print("Đang đọc toàn bộ dữ liệu vào RAM...")
    all_tasks = []
    
    if os.path.exists(CHEAT_CSV):
        df_cheat = pd.read_csv(CHEAT_CSV)
        for _, row in df_cheat.iterrows():
            all_tasks.append({
                'pgn': str(row.get("Game", "")),
                'elo_w': float(row.get("Elo White", 1500)),
                'elo_b': float(row.get("Elo Black", 1500)),
                'label_w': 1 if '1' in str(row.get("Liste cheat white", "")) else 0,
                'label_b': 1 if '1' in str(row.get("Liste cheat black", "")) else 0
            })
            
    if os.path.exists(CLEAN_PGN):
        with open(CLEAN_PGN, "r", encoding="utf-8") as f:
            while True:
                game = chess.pgn.read_game(f)
                if not game: break
                try:
                    elo_w = float(game.headers.get("WhiteElo", 1500))
                    elo_b = float(game.headers.get("BlackElo", 1500))
                except ValueError:
                    continue
                exporter = chess.pgn.StringExporter(headers=False, variations=False, comments=False)
                all_tasks.append({
                    'pgn': game.accept(exporter), 'elo_w': elo_w, 'elo_b': elo_b,
                    'label_w': 0, 'label_b': 0
                })

    print(f"Tổng số ván cờ cần xử lý: {len(all_tasks)}")
    num_cores = max(1, multiprocessing.cpu_count() - 1)
    print(f"Bắt đầu phân tích (Depth 12)! Đang chạy trên {num_cores} nhân CPU...")

    final_results = []
    with multiprocessing.Pool(processes=num_cores, initializer=init_worker, initargs=(STOCKFISH_PATH,)) as pool:
        for result in tqdm(pool.imap_unordered(process_single_game, all_tasks), total=len(all_tasks), desc="Tiến độ (Từng ván)"):
            final_results.extend(result)

    df_final = pd.DataFrame(final_results)
    columns_order = [
        "elo", "elo_diff", "move_count", 
        "acpl", "cpl_std", "opening_acpl", "middlegame_acpl", "endgame_acpl",
        "best_move_rate", "top3_rate", "blunder_rate", "mistake_rate", 
        "avg_entropy", "avg_complexity", "label"
    ]
    df_final = df_final[columns_order]
    df_final.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Hoàn tất! Tốc độ tối ưu mà data vẫn uy tín. Đã lưu vào {OUTPUT_CSV}.")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()