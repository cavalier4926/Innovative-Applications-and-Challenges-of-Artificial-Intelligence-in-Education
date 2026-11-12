
## Data dictionary (`data/raw/`)

* `interactions.csv` — one row per attempt: `student_id, seq_idx, item_id, concept_id, correct (0/1),
  timestamp_h (hours since the learner's first attempt), session_id, true_p` (the simulator's exact
  probability of a correct answer — used only for the Bayes-ceiling and never by any model).
* `students.csv` — `student_id, school_id, group (0 = majority, 1 = low-connectivity / under-resourced
  access), theta, eta, phi, slip, n_interactions, dropout (1 = disengaged before completion), final_mastery`.
  Groups share the same ability distribution; they differ only in access conditions (session gaps, noise).
* `items.csv` — `item_id, concept_id, difficulty, discrimination, guess`.
* `concepts.csv` — `concept_id, base_difficulty, learning_gain, forgetting_rate_per_day, n_prerequisites`.
* `prerequisite_edges.csv` — the true prerequisite DAG (`child_concept, parent_concept`).
* `dataset_summary.json` — summary statistics.

To use a **real** dataset (e.g. ASSISTments, EdNet), export it in the same `interactions.csv` /
`students.csv` schema (set `true_p` to 0 and `group` to the audit attribute) and run the same scripts.

## Code (`code/`)

| File | Purpose |
|---|---|
| `simulator.py` | EduSim-GT ground-truth simulator; `python simulator.py --n_students 4000 --seed 2026 --out ../data/raw` |
| `autograd.py` | NumPy reverse-mode autodiff, fused biased attention, Adam |
| `test_autograd.py` | finite-difference gradient checks of every operator |
| `data.py` | loading, stratified learner-level splits, causal forgetting-aware query features |
| `models.py` | CAFE-KT, SAKT, AKT-s, DKT (+ ablation variants, online `predict_next`) |
| `classic.py` | BKT (EM), PFA, GBDT-FE, feature-based early-warning baselines |
| `train.py` | equity-aware objective, training loop (EMA, reduce-on-plateau, early stopping) |
| `metrics.py` | AUC/ACC/RMSE/NLL/ECE, group metrics, ABROCA, early-warning, conformal prediction |
| `run_experiments.py` | all main / ablation / equity / sparsity runs (`--list`, `--worker i --n_workers n`) |
| `federated.py` | school-level FedAvg and DP-FedAvg with Rényi-DP accounting |
| `tutor_sim.py` | closed-loop adaptive-practice study (ZPD and mastery-learning policies) |
| `interpret.py` | counterfactual forgetting curves, influence graph, error buckets, attention example |
| `analysis_tables.py`, `analysis_conformal.py` | aggregate tables and conformal analysis |
| `fig_diagrams.py`, `fig_results.py` | all figures |
| `make_paper.py`, `build_paper.js` | assemble the manuscript from the result files |

### Reproduce everything

```bash
cd code
python test_autograd.py
python simulator.py --n_students 4000 --seed 2026 --out ../data/raw
python run_experiments.py --worker 0 --n_workers 2 &   # run the two workers in parallel
python run_experiments.py --worker 1 --n_workers 2
python run_experiments.py --stage accmode   # CAFE-KT-Acc seeds 3-4
python federated.py --mode fedavg --unit school --q 0.3
python federated.py --mode fedavg --unit learner --q 0.05 --local_steps 2 --local_bs 1 --clip 0.5
python federated.py --mode dp --unit learner --sigma 1.0 --q 0.05 --local_steps 2 --local_bs 1 --clip 0.5
python federated.py --mode dp --unit learner --sigma 0.6 --q 0.05 --local_steps 2 --local_bs 1 --clip 0.5
python tutor_sim.py 300 60
python interpret.py; python analysis_conformal.py; python analysis_tables.py
python fig_diagrams.py; python fig_results.py
python make_paper.py && node build_paper.js
```

Total compute for the full study on a 2-core CPU: about 4–5 hours. Note: the scripts that build tables and figures read `results/runs/`, `results/conformal/`, `results/interpret/`, `results/tutoring/` and `results/federated/`; the shipped copies of those JSON files are in `results/analyses/`.
