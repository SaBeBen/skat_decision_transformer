from typing import Dict, List, Optional

import numpy as np


class TrajectoryReplayBuffer:
    """
    FIFO replay buffer over whole trajectories (games from the perspective of one player), as in the Online DT.

    Trajectories are dicts with "states" (12, state_dim), "actions" (12, 12), "rewards" (12,) and optionally the
    legal actions "legal" (12, 12) and the locations of unseen cards "hidden" (12, 32).
    The buffer can be passed directly to the DecisionTransformerSkatDataCollator, which samples games, cuts them at a
    random time step and computes the returns-to-go from the stored rewards. As the returns-to-go are always computed
    from the achieved rewards, this realises the hindsight return relabelling of the Online DT.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.trajectories: List[Dict[str, np.ndarray]] = []
        self._next = 0  # position of the oldest trajectory once the buffer is full

    def add(self, trajectory: Dict[str, np.ndarray]) -> None:
        if len(self.trajectories) < self.capacity:
            self.trajectories.append(trajectory)
        else:
            self.trajectories[self._next] = trajectory
            self._next = (self._next + 1) % self.capacity

    def extend(self, trajectories) -> None:
        for trajectory in trajectories:
            self.add(trajectory)

    def seed_from_offline(self, data: dict, top_k: Optional[int] = None) -> int:
        """
        Fills the buffer with offline trajectories from data_pipeline.get_states_actions_rewards.

        :param data: dict with "states", "actions" and "rewards" of the offline data.
        :param top_k: only keep the top_k trajectories with the highest return (as in the Online DT),
            all trajectories if None.
        :return: the number of added trajectories.
        """
        trajectories = []
        for i in range(len(data["states"])):
            trajectory = {
                "states": np.stack(data["states"][i]).astype(np.float32),
                "actions": np.asarray(data["actions"][i], dtype=np.float32),
                "rewards": np.asarray(data["rewards"][i], dtype=np.float32).reshape(-1),
            }
            for key in ("legal", "hidden"):
                if key in data:
                    trajectory[key] = np.asarray(data[key][i], dtype=np.int8)
            trajectories.append(trajectory)

        if top_k is not None:
            trajectories.sort(key=lambda traj: traj["rewards"].sum(), reverse=True)
            trajectories = trajectories[:top_k]

        trajectories = trajectories[-self.capacity:]
        self.extend(trajectories)
        return len(trajectories)

    def __len__(self) -> int:
        return len(self.trajectories)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        return self.trajectories[idx]
