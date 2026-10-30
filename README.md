# Identifiability and verification of epistemic–aleatoric uncertainty decomposition in probabilistic load forecasting

Code, data, trained models and raw result logs for the paper of the same title.
Every table and figure in the paper is produced by the scripts below from the data
in `data/`; all experiments run on CPU (two cores).

## Forecasting set-up

Both datasets are hourly. Each sample holds, for the target clock hour, the load,
temperature and calendar variables on each of the previous seven days, followed by
the calendar variables and the observed temperature of the target day (ex-post
setting). The target is the load at that hour on the next day (24 h ahead). The
target-day load entry is replaced by the previous day's load, so no target
information enters the input (`backbone.add_target_day`). Post-COVID loads are
stored in kW and reported in MW.

## Decompositions (`src/models.py`, `src/decomp.py`)

| Key | Decomposition |
|---|---|
| `FREE` | free epistemic/aleatoric outputs trained on their sum (non-identifiable control) |
| `MC` | MC dropout, M = 40, **all** dropout layers active (`model.train()` under `no_grad`) |
| `ENS` | deep ensemble, K = 5 |
| `LA-1` | last-layer Laplace (65 weights), heteroscedastic GGN, prior precision by evidence |
| `LA-2` | two-layer Laplace (4225 weights), same |
| `LCD` | learning-curve decomposition: aleatoric = asymptote of a cross-fitted learning curve (`e11_learning_curve.py`) |
| `MVE`, `CMC`, `QR`, `PERS` | baselines: deterministic network, classical MC dropout, quantile regression, persistence |

`e9_fairness_remedy.py` also trains the shared network with the beta-NLL loss
(Seitzer et al., 2022; `HeteroMCDropout.beta = 0.5`) as a remedy for the
aleatoric absorption found by T4.

Note on MC dropout: putting only the `nn.Dropout` modules in training mode is not
enough for `nn.TransformerEncoderLayer`; its inference fast path then skips the
layer's internal dropout. The whole model is put in training mode (there is no
batch normalization).

## Experiments

Table, figure, section and remark numbers refer to the IEEE PES Transactions version
(main paper: Tables I-VI, Figs. 1-4, Propositions 1-6; supplementary material: Sections S1-S8, Tables S1-S15, Fig. S1).

| Script | Paper |
|---|---|
| `e1_counterexample.py` | Table I (free outputs, initialization only) |
| `e2_dropout_sweep.py` | selection of p; Fig. S1(a); Table S12 |
| `e3_main.py gef\|covid` | Table S2 (T1), Tables V and S10 (accuracy, calibration), Fig. 4 |
| `e4_reducibility.py` | Table S3 (T2), Fig. 2(a) |
| `e5_support.py` | Table S4 (T3 distance probe, one-week block bootstrap), Fig. 2(b) |
| `e6_ood.py` | Table S5 (T3 out of support, validation-year stopping), Fig. 2(c) |
| `e7_semisynthetic.py gef\|covid` | Tables II, S6 and S7 (T4), Fig. 3 |
| `e8_passes.py` | Fig. S1(b) |
| `e9_fairness_remedy.py` | Table S8 (MC dropout at p = 0.1; beta-NLL), Section S4 |
| `e10_window_validation.py` | short-window conditions with early stopping on the last 20 % of the window: "window" rows of Tables II, S5-S8, Fig. 2(c) |
| `e11_learning_curve.py gef\|covid` | learning-curve decomposition: Tables II and S9 |
| `e12_absorption.py` | what the variance head estimates (Proposition 5): Table III |
| `e13_reserve.py` | reserve relevance of the total distributions: Tables V and S11 |
| `e14_timescale.py gef\|covid` | time scale of the retraining decision (Proposition 6): Tables VI, S14 and S15 |
| `oracle_ceiling.py OUTDIR` | dominant-source ceiling for an oracle that knows the expected estimation error (Proposition 6): Table S13 |
| `paper_assets.py OUTDIR` | builds every table body (`OUTDIR/tables/*.tex`), the rule-based verification summary (Table IV) and every figure |
| `ieee_assets.py OUTDIR` | compact tables of the main paper (`ieee_t4.tex`, `ieee_accuracy.tex`, `ieee_decision.tex`) and Tables S10, S11, S14, S15 |
| `fig_overview.py OUTDIR` | Fig. 1 |

The verification summary (Table IV) is computed by `paper_assets.py` from the result files
with the thresholds stated in its caption; `tables/scorecard.json` records the level of every
entry; the decision column (`T4d`) is computed from the E14 results. The hour-level dominant-source level
(`T4s`) is kept in the JSON but not shown in Table IV: by Proposition 6(i) even an oracle that knows the expected estimation error labels
hours imperfectly (`oracle_ceiling.py`).

```
pip install -r requirements.txt
cd src
python3 e1_counterexample.py
python3 e2_dropout_sweep.py          # selects p; later scripts read it
python3 e3_main.py gef
python3 e3_main.py covid
python3 e4_reducibility.py
python3 e6_ood.py
python3 e5_support.py
python3 e7_semisynthetic.py gef
python3 e7_semisynthetic.py covid
python3 e8_passes.py
python3 e9_fairness_remedy.py
python3 e10_window_validation.py
python3 e11_learning_curve.py gef
python3 e11_learning_curve.py covid
python3 e12_absorption.py
python3 e13_reserve.py               # reads the E3 predictions
python3 e14_timescale.py gef         # reuses the E7 / E10 models
python3 e14_timescale.py covid
python3 paper_assets.py ..
python3 ieee_assets.py ..
python3 fig_overview.py ..
python3 oracle_ceiling.py ..
```

Early stopping: every model is early-stopped on the validation year, except
that the short-window models (most recent one to three months) are reported
twice, once with that rule (E6, E7; they stop after one to three epochs) and
once stopped on the last 20 % of their own window (E10; 13 to 50 epochs).

Training is resume-safe: a model whose checkpoint and metadata exist in
`checkpoints/` is loaded instead of retrained. `results/` holds a text log and a
JSON file per experiment.
