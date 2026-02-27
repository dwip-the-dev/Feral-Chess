import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import chess
import os
import time

print("="*80)
print("🐉 FERAL CHESS 950 🐉")
print("="*80)

class Config:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MCTS_SIMULATIONS = 500  # Strong
    EXPLORATION_CONSTANT = 1.0
    TEMPERATURE = 0.1
    MAX_MOVES = 4672

config = Config()
print(f"Using device: {config.DEVICE}")
print(f"MCTS strength: {config.MCTS_SIMULATIONS} simulations")

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = F.relu(x + residual)
        return x

class DynamicChessNet(nn.Module):
    def __init__(self, num_blocks, channels):
        super().__init__()
        self.num_blocks = num_blocks
        self.channels = channels

        self.conv_input = nn.Conv2d(25, channels, 3, padding=1)
        self.bn_input = nn.BatchNorm2d(channels)

        self.res_blocks = nn.ModuleList([
            ResidualBlock(channels) for _ in range(num_blocks)
        ])

        self.policy_conv = nn.Conv2d(channels, 32, 1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 64, config.MAX_MOVES)

        self.value_conv = nn.Conv2d(channels, 1, 1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(64, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.bn_input(self.conv_input(x)))

        for block in self.res_blocks:
            x = block(x)

        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)
        policy = self.policy_fc(policy)

        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return policy, value

class MoveEncoder:
    def __init__(self):
        self.move_to_idx = {}
        self.idx_to_move = []
        prom_map = {'n': chess.KNIGHT, 'b': chess.BISHOP, 'r': chess.ROOK, 'q': chess.QUEEN}

        print("🎯 Initializing move encoder...")
        for from_square in chess.SQUARES:
            for to_square in chess.SQUARES:
                if from_square == to_square:
                    continue
                move = chess.Move(from_square, to_square)
                self._add_move(move.uci())
                from_row, to_row = from_square // 8, to_square // 8
                if (from_row == 1 and to_row == 0) or (from_row == 6 and to_row == 7):
                    for prom_int in prom_map.values():
                        prom_move = chess.Move(from_square, to_square, promotion=prom_int)
                        self._add_move(prom_move.uci())

        while len(self.idx_to_move) < config.MAX_MOVES:
            self.idx_to_move.append('a1a1')

        print(f"✅ Move encoder ready: {len([m for m in self.idx_to_move if m != 'a1a1'])} moves")

    def _add_move(self, uci):
        if uci not in self.move_to_idx:
            idx = len(self.idx_to_move)
            self.move_to_idx[uci] = idx
            self.idx_to_move.append(uci)

    def encode(self, move_uci):
        return self.move_to_idx.get(move_uci, 0)

    def decode(self, idx):
        return self.idx_to_move[idx] if idx < len(self.idx_to_move) else 'a1a1'

    def get_legal_mask(self, legal_moves):
        mask = torch.zeros(config.MAX_MOVES)
        for move in legal_moves:
            if move in self.move_to_idx:
                mask[self.move_to_idx[move]] = 1.0
        return mask

def board_to_state(board):
    state = np.zeros((25, 8, 8), dtype=np.float32)

    piece_map = {1:0, 2:1, 3:2, 4:3, 5:4, 6:5}
    for color in (chess.WHITE, chess.BLACK):
        color_offset = 0 if color else 6
        for piece_type in range(1, 7):
            mask = board.pieces(piece_type, color)
            channel = piece_map[piece_type] + color_offset
            for sq in chess.SQUARES:
                if mask & (1 << sq):
                    row = 7 - (sq // 8)
                    col = sq % 8
                    state[channel, row, col] = 1.0

    if board.has_kingside_castling_rights(chess.WHITE):
        state[12, 7, 4] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        state[13, 7, 4] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        state[14, 0, 4] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        state[15, 0, 4] = 1.0

    state[16, :, :] = 1.0 if board.turn == chess.WHITE else 0.0
    state[17, :, :] = min(board.fullmove_number / 100, 1.0)

    if board.has_legal_en_passant():
        ep_square = board.ep_square
        if ep_square:
            row = 7 - (ep_square // 8)
            col = ep_square % 8
            state[18, row, col] = 1.0

    state[19, :, :] = 1.0 if board.is_check() else 0.0
    state[20, :, :] = min(board.halfmove_clock / 100, 1.0)

    rep = 0
    for i in range(3):
        if board.is_repetition(i+1):
            rep = i+1
    state[21, :, :] = min(rep / 3, 1.0)

    if board.move_stack:
        last = board.move_stack[-1]
        state[22, 7 - (last.from_square // 8), last.from_square % 8] = 1.0
        state[23, 7 - (last.to_square // 8), last.to_square % 8] = 1.0

    state[24, :, :] = 1.0 if board.halfmove_clock >= 100 else 0.0

    return state

class MCTSNode:
    def __init__(self, board, parent=None, action=None, prior=0.0):
        self.board = board
        self.parent = parent
        self.action = action
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children = {}
        self.is_expanded = False

    def value(self):
        return self.value_sum / self.visit_count if self.visit_count else 0.0

    def ucb_score(self, exploration=config.EXPLORATION_CONSTANT):
        if self.visit_count == 0:
            return float('inf')
        return self.value() + exploration * self.prior * (self.parent.visit_count ** 0.5) / (1 + self.visit_count)

def mcts_search(root_board, model, move_encoder, num_simulations=config.MCTS_SIMULATIONS):
    model.eval()
    root = MCTSNode(root_board)

    for _ in range(num_simulations):
        node = root
        board = node.board.copy()
        path = [node]

        while node.is_expanded and node.children:
            best = max(node.children.values(), key=lambda c: c.ucb_score())
            board.push(chess.Move.from_uci(best.action))
            node = best
            path.append(node)

        if not node.is_expanded and not board.is_game_over():
            state = board_to_state(board)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(config.DEVICE)
            with torch.no_grad():
                policy_logits, value = model(state_tensor)
            policy = F.softmax(policy_logits[0], dim=0).cpu().numpy()

            legal_moves = [m.uci() for m in board.legal_moves]
            mask = move_encoder.get_legal_mask(legal_moves).numpy()
            masked_policy = policy * mask
            if masked_policy.sum() > 0:
                masked_policy /= masked_policy.sum()

            for move in legal_moves:
                idx = move_encoder.encode(move)
                prior = masked_policy[idx]
                if prior > 1e-5:
                    child_board = board.copy()
                    child_board.push(chess.Move.from_uci(move))
                    node.children[idx] = MCTSNode(child_board, parent=node, action=move, prior=prior)
            node.is_expanded = True
            leaf_value = -value.item() if board.turn == chess.BLACK else value.item()
        else:
            if board.is_checkmate():
                leaf_value = -1 if board.turn == chess.WHITE else 1
            else:
                leaf_value = 0

        for n in reversed(path):
            n.visit_count += 1
            n.value_sum += leaf_value
            leaf_value = -leaf_value

    policy = np.zeros(config.MAX_MOVES)
    for idx, child in root.children.items():
        policy[idx] = child.visit_count
    if policy.sum() > 0:
        policy /= policy.sum()
    return policy
  
class AIPlayer:
    def __init__(self, model_path):
        print(f"\n📂 Loading 950 model from: {model_path}")

        if not os.path.exists(model_path):
            print(f"❌ Model not found at: {model_path}")
            print("Looking for .pt files in current directory...")
            pt_files = [f for f in os.listdir('.') if f.endswith('.pt')]
            if pt_files:
                print(f"Found: {pt_files}")
                model_path = pt_files[0]
                print(f"Using: {model_path}")
            else:
                raise FileNotFoundError("No .pt model file found!")

        import numpy as np
        import torch.serialization
        torch.serialization.add_safe_globals([np._core.multiarray.scalar])

        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

        state_dict = checkpoint['model_state_dict']
        if 'module.' in list(state_dict.keys())[0]:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]
                new_state_dict[name] = v
            state_dict = new_state_dict

        block_nums = set()
        for k in state_dict.keys():
            if 'res_blocks.' in k:
                try:
                    block_num = int(k.split('.')[1])
                    block_nums.add(block_num)
                except:
                    pass
        num_blocks = len(block_nums)

        conv_shape = state_dict['conv_input.weight'].shape
        channels = conv_shape[0]

        print(f"🔍 Detected: {num_blocks} blocks, {channels} channels")

        self.model = DynamicChessNet(num_blocks, channels)
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(config.DEVICE)
        self.model.eval()

        self.move_encoder = MoveEncoder()
        self.board = chess.Board()

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"✅ 950 ready! ({total_params/1e6:.1f}M parameters)")

    def get_best_move(self):
        policy = mcts_search(self.board, self.model, self.move_encoder)

        legal_moves = [m.uci() for m in self.board.legal_moves]
        mask = self.move_encoder.get_legal_mask(legal_moves).numpy()
        masked_policy = policy * mask

        if masked_policy.sum() > 0:
            masked_policy /= masked_policy.sum()
        else:
            masked_policy = mask / mask.sum()

        if config.TEMPERATURE != 1.0:
            masked_policy = masked_policy ** (1.0/config.TEMPERATURE)
            masked_policy /= masked_policy.sum()

        best_idx = np.argmax(masked_policy)
        return self.move_encoder.decode(best_idx), masked_policy[best_idx]

    def show_top_moves(self, n=5):
        state = board_to_state(self.board)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(config.DEVICE)

        with torch.no_grad():
            policy_logits, value = self.model(state_tensor)
            policy = F.softmax(policy_logits[0], dim=0).cpu().numpy()

        legal_moves = [m.uci() for m in self.board.legal_moves]
        move_probs = []

        for move in legal_moves:
            idx = self.move_encoder.encode(move)
            if idx < len(policy):
                move_probs.append((move, policy[idx]))

        move_probs.sort(key=lambda x: x[1], reverse=True)

        print(f"\n📊 Evaluation: {value.item():.3f}")
        print(f"Top {n} moves:")
        for i, (move, prob) in enumerate(move_probs[:n]):
            print(f"  {i+1}. {move} ({prob:.3f})")

        return value.item()
      
def main():
    print("\n🔍 Looking for 950 model in current directory...")

    # Try to find the model
    model_path = '/content/model_iter_950.pt'

    player = AIPlayer(model_path)

    print("\n" + "="*60)
    print("♜ YOU vs 950 ♞")
    print("="*60)
    print("Enter UCI moves (e.g., 'e2e4')")
    print("Special moves: 'O-O' (castling), 'e7e8q' (promotion)")
    print("Commands: 'top' for analysis, 'quit' to exit")
    print("="*60)

    player.board.reset()

    while True:
        print(f"\n{player.board}")

        if player.board.is_game_over():
            print(f"\n🏁 Game Over: {player.board.result()}")
            if player.board.is_checkmate():
                print("Checkmate! " + ("You win! 🎉" if player.board.turn == chess.BLACK else "950 wins! 🤖"))
            break

        if player.board.turn == chess.WHITE:
            cmd = input("\nYour move > ").strip().lower()

            if cmd == 'quit':
                break
            elif cmd == 'top':
                player.show_top_moves()
                continue

          
            if cmd == 'o-o' or cmd == '0-0':
                cmd = 'e1g1' if player.board.turn == chess.WHITE else 'e8g8'
            elif cmd == 'o-o-o' or cmd == '0-0-0':
                cmd = 'e1c1' if player.board.turn == chess.WHITE else 'e8c8'

            try:
                move = chess.Move.from_uci(cmd)
                if move in player.board.legal_moves:
                    player.board.push(move)
                else:
                    print("❌ Illegal move!")
                    print(f"Legal: {[m.uci() for m in player.board.legal_moves][:10]}...")
            except:
                print("❌ Invalid format! Use e2e4, O-O, e7e8q")
        else:
            print(f"\n🤖 950 thinking...")
            move, conf = player.get_best_move()
            print(f"🤖 950 plays: {move} (confidence: {conf:.3f})")
            player.board.push(chess.Move.from_uci(move))
          
    if input("\nPlay again? (y/n): ").lower() == 'y':
        main()

if __name__ == "__main__":
    main()
