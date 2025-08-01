import os, json
from stable_baselines3.common.callbacks import BaseCallback
from .utils import tokens_to_infix

class SaveBestProgramCallback(BaseCallback):
    """
    - 학습 중 reward가 갱신되면 best_program.json / .txt 저장
    """
    def __init__(self, save_dir: str, verbose: int = 0):
        super().__init__(verbose)
        self.save_dir = save_dir
        self.best_reward = -float("inf")
        os.makedirs(save_dir, exist_ok=True)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info, rew in zip(infos, self.locals["rewards"]):
            if info and "program" in info and rew > self.best_reward:
                self.best_reward = rew
                tokens = info["program"]
                # JSON
                with open(os.path.join(self.save_dir, "best_program.json"), "w") as f:
                    json.dump({"tokens": tokens}, f)
                # 사람이 읽을 수 있는 식
                with open(os.path.join(self.save_dir, "best_program.txt"), "w") as f:
                    f.write(tokens_to_infix(tokens))
                if self.verbose:
                    print(f"[BEST] reward={rew:.4f} → 저장 완료")
        return True
