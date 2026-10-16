# V-IMMUNE project conventions

- Python 3.11 venv at `.venv` (`source .venv/bin/activate`). CPU-only must always work; CUDA is auto-detected.
- Layout: `src/vimmune/<pkg>/` per report section (see ARCHITECTURE.md). Tests in `tests/`, outputs in `runs/` (gitignored).
- Every module docstring cites the report section it implements (`"""Implements report §2.x ..."""`).
- Numbers come from `configs/*.yaml` via `vimmune.config.load_config(profile)`. `full` = spec numbers,
  `demo` = scaled down (`inherit: full`). Never hard-code a tunable; never silently change a spec number —
  record deviations in DECISIONS.md.
- Determinism: one `seed` in config; call `seed_everything(cfg.seed)`; numpy RNGs are `default_rng([seed, ...])`.
- BDW tables are pandas DataFrames with the columns of `vimmune.io.bdw.BDW`; snippets live in a `BurstStore`.
- Never fabricate results: every reported number must be read from a file the code wrote under `runs/`.
- Scope: passive analysis of simulated, fictional protocols only. No transmit, jamming, spoofing,
  payload decoding or real-protocol modelling code.
- `make check` = ruff + mypy + pytest. Keep tests CPU-fast (tiny configs).
