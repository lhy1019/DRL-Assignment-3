# studentagent.py
import cv2, gym, torch, numpy as np
from collections import deque
from pathlib import Path

from models import DuelingQNetwork      # ship this file with submission

CKPT = Path(__file__).with_name("weights.pt")
DEV  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ────────── preprocessing 240×256×3 → 1×84×84 ──────────
def to_gray84(obs_rgb):
    g = cv2.cvtColor(obs_rgb, cv2.COLOR_RGB2GRAY)
    g = cv2.resize(g, (84, 84), interpolation=cv2.INTER_AREA)
    return g[None, ...]                 # (1,84,84)

# ────────── evaluation-only agent ──────────
class Agent:
    def __init__(self, skip: int = 4):
        self.action_space  = gym.spaces.Discrete(12)   # COMPLEX_MOVEMENT
        self.skip          = skip
        self.frame_counter = 0
        self.last_action   = 0

        # network
        self.net = DuelingQNetwork(4, 12).to(DEV)
        self.net.eval()
        ck = torch.load(CKPT, map_location=DEV)
        self.net.load_state_dict(ck["q_net"])          # <-- key from save()
        print("[studentagent] checkpoint loaded:", CKPT)

        # frame stack deque
        self.stack = deque(maxlen=4)

    # ---------- helper ----------
    def _state_from_obs(self, obs):
        """Return (4,84,84) uint8 stack, building it if needed."""
        if obs.shape == (4, 84, 84):           # grader already stacked
            return obs
        f = to_gray84(obs)
        if len(self.stack) == 0:
            for _ in range(4):
                self.stack.append(f)
        else:
            self.stack.append(f)
        return np.concatenate(list(self.stack), axis=0)

    # ---------- main API ----------
    def act(self, observation):
        """
        Called every env frame by the grader.  We repeat each chosen
        action for <skip> frames to mimic MaxAndSkip(4) used in training.
        """
        state = self._state_from_obs(observation)      # (4,84,84) uint8

        if self.frame_counter % self.skip == 0:
            s = torch.from_numpy(state).unsqueeze(0).to(DEV)  # (1,4,84,84)
            with torch.no_grad():
                q = self.net(s)
            self.last_action = int(q.argmax(1).item())

        self.frame_counter += 1
        return self.last_action
