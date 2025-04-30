# studentagent.py
import cv2, gym, torch, numpy as np
from collections import deque
from pathlib import Path

from models import DuelingQNetwork        # must be shipped too
CKPT = Path("./weights.pt")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ───────────────────────── preprocessing ──────────────────────────
def preprocess(obs):
    # obs: uint8 (H,W,3) = (240,256,3)
    obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)                 # (240,256)
    obs = cv2.resize(obs, (84, 84),interpolation=cv2.INTER_AREA)              # (84,84)
    obs = np.expand_dims(obs, 0)                                # (1,84,84)
    return obs.astype(np.uint8)

# ───────────────────────── Agent class ────────────────────────────
class Agent:
    """Greedy Dueling-Double-Noisy agent with internal skip-4."""
    def __init__(self, skip: int = 4):
        self.action_space   = gym.spaces.Discrete(12)   # COMPLEX_MOVEMENT
        self.skip           = skip                     # repeat horizon
        self.frame_counter  = 0                        # global frame index
        self.last_action    = 0                        # repeated action
        self.stack          = deque(maxlen=4)          # for 4-frame stack
        self.prev_raw       = None                     # previous raw RGB frame

        # build network
        self.net = DuelingQNetwork(in_channels=4, n_actions=12).to(DEVICE)
        self.net.eval()                                # Noisy → deterministic

        # load trained weights
        ckpt = torch.load(CKPT, map_location=DEVICE)
        key  = "q_net" if "q_net" in ckpt else "q_net_state"
        self.net.load_state_dict(ckpt[key])
        print(f"[studentagent] loaded weights from {CKPT} (key='{key}')")

    # ───────────────────────────────────────────────────────────────
    def _state_from_observation(self, obs: np.ndarray) -> np.ndarray:
        """
        Convert grader's observation to the (4,84,84) uint8 stack expected
        by the network.
        """
        if obs.shape == (4, 84, 84):          # grader already stacked
            return obs

        if self.prev_raw is None:
            self.prev_raw = obs
        
        if self.frame_counter % self.skip == 0:
            max_frame = np.maximum(self.prev_raw, obs)
            frame = preprocess(max_frame)        # (1,84,84)
            if len(self.stack) == 0:
                zero = np.zeros_like(frame)
                for _ in range(3):
                    self.stack.append(zero)
            self.stack.append(frame)             # (4,84,84)
        
        self.prev_raw = obs                    # save for next step
            


        return np.concatenate(self.stack, axis=0)      # (4,84,84)

    # ───────────────────────────────────────────────────────────────
    def act(self, observation: np.ndarray) -> int:
        """
        Called *every* NES frame by grader.  Only compute a new network
        action once every `skip` frames; in-between return the last one.
        """
        state = self._state_from_observation(observation)

        if self.frame_counter % self.skip == 0:
            with torch.no_grad():
                q = self.net(torch.from_numpy(state)
                                   .unsqueeze(0)       # (1,4,84,84)
                                   .to(DEVICE))
            self.last_action = int(q.argmax(1).item())

        self.frame_counter += 1
        return self.last_action
