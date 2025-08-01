
# Factor Factory – Constrained RL (Tree Assembly) v2.1 (fast)

### What's new
- **Fast eval**: subsampling (`eval_stride`) and truncation (`max_eval_bars`)
- **LRU cache** for repeated programs
- **np.clip** path to avoid pandas overhead
- Conservative PPO defaults for stability

### Quick start
```bash
pip install -r factor_factory/requirements.txt

python -m factor_factory.scripts.cli_rlc_train --symbol BTCUSDT --interval 1h   --timesteps 150000 --save models/ppo_program.zip --eval_stride 2 --max_eval_bars 20000

python -m factor_factory.scripts.cli_rlc_best --model models/ppo_program.zip   --symbol BTCUSDT --interval 1h --tries 256 --outdir rlc_out --eval_stride 2 --max_eval_bars 20000

python -m factor_factory.scripts.cli_rlc_eval --program rlc_out/best_program.json   --symbol BTCUSDT --interval 1h --outdir rlc_out
```
