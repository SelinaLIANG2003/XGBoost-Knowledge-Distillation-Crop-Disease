import os
import time
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, recall_score


# ============================================================
# Data loading (fixed split by data_seed)
# ============================================================
def load_and_split_data(
    csv_folder: str, train_size: int, test_size: int, data_seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not os.path.isdir(csv_folder):
        raise FileNotFoundError(f"[csv_folder not found] {csv_folder}")

    dfs = []
    for f in os.listdir(csv_folder):
        if f.endswith(".csv"):
            dfs.append(pd.read_csv(os.path.join(csv_folder, f)))

    if len(dfs) == 0:
        raise FileNotFoundError(f"No CSV files found in: {csv_folder}")

    data = pd.concat(dfs, ignore_index=True)

    train_list, test_list = [], []
    for c in sorted(data["Class_Label"].unique()):
        class_df = data[data["Class_Label"] == c].copy()
        class_df = class_df.sample(frac=1, random_state=data_seed).reset_index(drop=True)

        train_list.append(class_df.iloc[:train_size])
        test_list.append(class_df.iloc[train_size : train_size + test_size])

    train = pd.concat(train_list, ignore_index=True)
    test = pd.concat(test_list, ignore_index=True)
    return train, test


def prepare_xy(
    train: pd.DataFrame, test: pd.DataFrame
) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    Xtr = train.drop(columns=["Class_Label", "Image_Name"])
    ytr = train["Class_Label"].astype(int).to_numpy()
    Xte = test.drop(columns=["Class_Label", "Image_Name"])
    yte = test["Class_Label"].astype(int).to_numpy()
    return Xtr, ytr, Xte, yte


# ============================================================
# Evaluation
# ============================================================
def eval_probs(probs: np.ndarray, y_true: np.ndarray) -> Tuple[float, float, float]:
    """return: accuracy, macro-F1, disease-recall(Healthy vs Disease)"""
    y_pred = np.argmax(probs, axis=1)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    rec = recall_score((y_true != 0).astype(int), (y_pred != 0).astype(int))
    return acc, f1, rec


def predict_probs(model: xgb.Booster, X: pd.DataFrame) -> Tuple[np.ndarray, float]:
    dX = xgb.DMatrix(X)
    t0 = time.time()
    probs = model.predict(dX)
    infer_time_sec = time.time() - t0
    return probs, infer_time_sec


# ============================================================
# Feature transform
# ============================================================
def log1p_transform(X: pd.DataFrame) -> pd.DataFrame:
    """
    x -> log(1 + x) transform (elementwise)
    Assumes features are >= 0.
    """
    X = X.astype(np.float32)
    return np.log1p(X)
    # if tiny negatives can appear:
    # return np.log1p(X.clip(lower=0.0))


# ============================================================
# KD utilities
# ============================================================
def temperature_scale(probs: np.ndarray, T: float) -> np.ndarray:
    """scale probabilities with temperature T then renormalize"""
    T = float(max(1e-6, T))
    logp = np.log(np.clip(probs, 1e-12, 1.0)) / T
    logp -= logp.max(axis=1, keepdims=True)
    p = np.exp(logp)
    return p / np.clip(p.sum(axis=1, keepdims=True), 1e-12, None)


def expand_topm(
    X: pd.DataFrame, soft: np.ndarray, top_m: int
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """expand each sample into top-m classes -> (X_expanded, y_expanded, weight_expanded)"""
    top_m = int(top_m)
    idx = np.argsort(-soft, axis=1)[:, :top_m]

    Xs, ys, ws = [], [], []
    for i in range(len(X)):
        xi = X.iloc[i].values
        for k in idx[i]:
            Xs.append(xi)
            ys.append(int(k))
            ws.append(float(soft[i, k]))

    return (
        pd.DataFrame(Xs, columns=X.columns),
        np.array(ys, dtype=int),
        np.array(ws, dtype=float),
    )


# ============================================================
# Formatting helpers (force +/- sign)
# ============================================================
def pct_delta(new: float, base: float) -> float:
    """percent change: (new-base)/base*100; returns nan if base==0"""
    if base == 0:
        return float("nan")
    return (new - base) / base * 100.0


def fmt_abs6(x: float) -> str:
    return f"{x:+.6f}"


def fmt_time2(x: float) -> str:
    return f"{x:+.2f} sec"


def fmt_time6(x: float) -> str:
    return f"{x:+.6f} sec"


def fmt_pct2(x: float) -> str:
    if np.isnan(x):
        return "nan"
    return f"{x:+.2f}%"


# ============================================================
# Pretty printing helpers
# ============================================================
def print_delta_block(delta_abs: Dict[str, float], delta_pct: Dict[str, float]) -> None:
    print(f"Δ Accuracy:       {delta_abs['acc']:+.6f} ({delta_pct['acc']:+.2f}%)")
    print(f"Δ F1-macro:       {delta_abs['f1']:+.6f} ({delta_pct['f1']:+.2f}%)")
    print(f"Δ Disease Recall: {delta_abs['recall']:+.6f} ({delta_pct['recall']:+.2f}%)")
    print(f"Δ Train time:     {delta_abs['train_time_sec']:+.2f} sec ({delta_pct['train_time_sec']:+.2f}%)")
    print(f"Δ Infer time:     {delta_abs['infer_time_sec']:+.6f} sec ({delta_pct['infer_time_sec']:+.2f}%)\n")


def print_summary_table(
    teacher: Dict[str, float],
    base: Dict[str, float],
    kd: Dict[str, float],
    delta_abs: Dict[str, float],
) -> None:
    print("=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Model':<18} {'Accuracy':>10} {'F1-macro':>12} {'Recall':>10} {'Train(s)':>10} {'Infer(s)':>10}")
    print("-" * 80)
    print(f"{'Teacher':<18} {teacher['acc']:>10.6f} {teacher['f1']:>12.6f} {teacher['recall']:>10.6f} {teacher['train_time_sec']:>10.2f} {teacher['infer_time_sec']:>10.6f}")
    print(f"{'Student-Base(MEAN)':<18} {base['acc']:>10.6f} {base['f1']:>12.6f} {base['recall']:>10.6f} {base['train_time_sec']:>10.2f} {base['infer_time_sec']:>10.6f}")
    print(f"{'Student-KD(MEAN)':<18} {kd['acc']:>10.6f} {kd['f1']:>12.6f} {kd['recall']:>10.6f} {kd['train_time_sec']:>10.2f} {kd['infer_time_sec']:>10.6f}")
    print(f"{'Δ(KD-Base)':<18} {delta_abs['acc']:>10.6f} {delta_abs['f1']:>12.6f} {delta_abs['recall']:>10.6f} {delta_abs['train_time_sec']:>10.2f} {delta_abs['infer_time_sec']:>10.6f}")
    print("=" * 80)


# ============================================================
# Main
# ============================================================
def main() -> None:    # ------------- Path -------------
    # Update this path to point to your local dataset folder
    csv_folder = r"./cropcsv"

    # ------------- Data split -------------
    train_size = 770
    test_size = 330

    # data_seed fixed (Teacher/Student compare on same split)
    data_seed = 31

    # ------------- Seeds (5 seeds) -------------
    teacher_seed = 31
    base_seeds = [31, 32, 33, 34, 35]
    kd_seeds = [31, 32, 33, 34, 35]

    # ------------- KD grid -------------
    alpha_list = [0.0, 0.01, 0.03, 0.05, 0.07, 0.09, 0.1]
    topm_list = [1, 2, 3]
    T = 2  # temperature for soft targets

    # ------------- Params -------------
    teacher_params = {
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        "eta": 0.2,
        "max_depth": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "num_class": 10,
        "seed": teacher_seed,
    }

    student_params = dict(teacher_params)
    student_params["max_depth"] = 3  # smaller student

    teacher_num_round = 700
    student_num_round = 150

    # ========================================================
    # Containers for CSV outputs
    # ========================================================
    results_full: List[Dict[str, Any]] = []       # kd_final_results_full.csv
    base_mean_rows: List[Dict[str, Any]] = []     # summary_base_mean.csv
    kd_mean_rows: List[Dict[str, Any]] = []       # summary_kd_mean_by_alpha_topm.csv
    kd_delta_rows: List[Dict[str, Any]] = []      # summary_kd_delta_vs_base.csv
    ranked_rows: List[Dict[str, Any]] = []        # kd_ranked_by_improvement.csv

    # ========================================================
    # 0) Fixed dataset
    # ========================================================
    print("=" * 80)
    print(f"Loading data with fixed split (data_seed={data_seed})")
    print("=" * 80)

    train, test = load_and_split_data(csv_folder, train_size, test_size, data_seed)
    Xtr, ytr, Xte, yte = prepare_xy(train, test)

    # feature transform: log(1+x) for all models
    Xtr = log1p_transform(Xtr)
    Xte = log1p_transform(Xte)

    dtr = xgb.DMatrix(Xtr, label=ytr)

    print(f"Train samples: {len(Xtr)}")
    print(f"Test samples:  {len(Xte)}\n")

    # ========================================================
    # 1) Train Teacher ONCE
    # ========================================================
    print("=" * 80)
    print(f"Training Teacher (seed={teacher_seed}, num_round={teacher_num_round})")
    print("=" * 80)

    t0 = time.time()
    teacher = xgb.train(teacher_params, dtr, num_boost_round=teacher_num_round, verbose_eval=False)
    teacher_train_time = time.time() - t0

    t_probs, t_infer_sec = predict_probs(teacher, Xte)
    t_acc, t_f1, t_rec = eval_probs(t_probs, yte)

    teacher_stat = {
        "acc": float(t_acc),
        "f1": float(t_f1),
        "recall": float(t_rec),
        "train_time_sec": float(teacher_train_time),
        "infer_time_sec": float(t_infer_sec),
    }

    print("Teacher Results:")
    print(f"  Accuracy:       {teacher_stat['acc']:.6f}")
    print(f"  F1-macro:       {teacher_stat['f1']:.6f}")
    print(f"  Disease Recall: {teacher_stat['recall']:.6f}")
    print(f"  Train time:     {teacher_stat['train_time_sec']:.2f} sec")
    print(f"  Infer time:     {teacher_stat['infer_time_sec']:.6f} sec\n")

    # save Teacher to full
    results_full.append({
        "seed": teacher_seed,
        "alpha": np.nan,
        "top_m": np.nan,
        "model": "Teacher",
        **teacher_stat,
    })

    # Teacher soft targets on TRAIN
    soft_train, _ = predict_probs(teacher, Xtr)
    soft_train = temperature_scale(soft_train, T)

    # ========================================================
    # 2) Student-Base: 5 seeds -> MEAN baseline
    # ========================================================
    print("=" * 80)
    print(f"Training Student-Base ({len(base_seeds)} seeds)")
    print("=" * 80)

    base_rows: List[Dict[str, Any]] = []
    for s in base_seeds:
        params = dict(student_params)
        params["seed"] = int(s)

        t0 = time.time()
        base_model = xgb.train(params, dtr, num_boost_round=student_num_round, verbose_eval=False)
        base_train_time = time.time() - t0

        b_probs, b_infer_sec = predict_probs(base_model, Xte)
        b_acc, b_f1, b_rec = eval_probs(b_probs, yte)

        row = {
            "seed": int(s),
            "acc": float(b_acc),
            "f1": float(b_f1),
            "recall": float(b_rec),
            "train_time_sec": float(base_train_time),
            "infer_time_sec": float(b_infer_sec),
        }
        base_rows.append(row)

        print(f"  Seed {s}: acc={row['acc']:.6f}, f1={row['f1']:.6f}, recall={row['recall']:.6f}, train={row['train_time_sec']:.2f}s, infer={row['infer_time_sec']:.6f}s")

        # save Base seed row to full
        results_full.append({
            "seed": int(s),
            "alpha": np.nan,
            "top_m": np.nan,
            "model": "Student-Base",
            **{k: row[k] for k in ["acc", "f1", "recall", "train_time_sec", "infer_time_sec"]},
        })

    df_base = pd.DataFrame(base_rows)
    base_mean = {
        "acc": float(df_base["acc"].mean()),
        "f1": float(df_base["f1"].mean()),
        "recall": float(df_base["recall"].mean()),
        "train_time_sec": float(df_base["train_time_sec"].mean()),
        "infer_time_sec": float(df_base["infer_time_sec"].mean()),
    }

    print("\nStudent-Base MEAN:")
    print(f"  Accuracy:       {base_mean['acc']:.6f}")
    print(f"  F1-macro:       {base_mean['f1']:.6f}")
    print(f"  Disease Recall: {base_mean['recall']:.6f}")
    print(f"  Train time:     {base_mean['train_time_sec']:.2f} sec")
    print(f"  Infer time:     {base_mean['infer_time_sec']:.6f} sec\n")

    # save Base MEAN to full
    results_full.append({
        "seed": "BASE_MEAN",
        "alpha": np.nan,
        "top_m": np.nan,
        "model": "Student-Base(MEAN)",
        **base_mean,
    })

    # summary_base_mean.csv (Teacher + BaseMean)
    base_mean_rows.append({"model": "Teacher", **teacher_stat})
    base_mean_rows.append({"model": "Student-Base(MEAN)", **base_mean})

    # ========================================================
    # 3) KD loop + per-config summary (MEAN + absΔ + %Δ)
    # ========================================================
    print("=" * 80)
    print("Training Student-KD grid (5 seeds) + per-config summary")
    print("=" * 80)

    for top_m in topm_list:
        # pre-expand once per top_m (saves time; alpha only changes weights)
        Xs, ys, ws = expand_topm(Xtr, soft_train, top_m)
        X_all = pd.concat([Xtr, Xs], ignore_index=True)
        y_all = np.concatenate([ytr, ys])

        hard_w = np.ones(len(ytr), dtype=float)

        for alpha in alpha_list:
            alpha = float(alpha)

            # weights depend only on alpha -> build DMatrix once per (alpha, top_m)
            w_all = np.concatenate([
                hard_w * (1.0 - alpha),
                ws * alpha,
            ])
            dkd = xgb.DMatrix(X_all, label=y_all, weight=w_all)

            kd_rows_this: List[Dict[str, Any]] = []

            for seed in kd_seeds:
                params = dict(student_params)
                params["seed"] = int(seed)

                t0 = time.time()
                kd_model = xgb.train(params, dkd, num_boost_round=student_num_round, verbose_eval=False)
                kd_train_time = time.time() - t0

                k_probs, k_infer_sec = predict_probs(kd_model, Xte)
                k_acc, k_f1, k_rec = eval_probs(k_probs, yte)

                row = {
                    "seed": int(seed),
                    "alpha": alpha,
                    "top_m": int(top_m),
                    "model": "Student-KD",
                    "acc": float(k_acc),
                    "f1": float(k_f1),
                    "recall": float(k_rec),
                    "train_time_sec": float(kd_train_time),
                    "infer_time_sec": float(k_infer_sec),
                }
                kd_rows_this.append(row)

                print(f"\n[Seed {seed} | alpha={alpha} | top_m={top_m}]")
                print(
                    f"KD        : acc={row['acc']:.6f}  f1={row['f1']:.6f}  recall={row['recall']:.6f}  "
                    f"train={row['train_time_sec']:.2f} sec  infer={row['infer_time_sec']:.6f} sec"
                )
                print(
                    f"Δ(BaseMEAN): acc={row['acc'] - base_mean['acc']:+.6f}  "
                    f"f1={row['f1'] - base_mean['f1']:+.6f}  "
                    f"recall={row['recall'] - base_mean['recall']:+.6f}  "
                    f"train={row['train_time_sec'] - base_mean['train_time_sec']:+.2f} sec  "
                    f"infer={row['infer_time_sec'] - base_mean['infer_time_sec']:+.6f} sec"
                )
                print("-" * 70)

                # save per-seed KD to full
                results_full.append(row)

                # save per-seed delta to full
                results_full.append({
                    "seed": int(seed),
                    "alpha": alpha,
                    "top_m": int(top_m),
                    "model": "Δ(KD-BaseMEAN)",
                    "acc": row["acc"] - base_mean["acc"],
                    "f1": row["f1"] - base_mean["f1"],
                    "recall": row["recall"] - base_mean["recall"],
                    "train_time_sec": row["train_time_sec"] - base_mean["train_time_sec"],
                    "infer_time_sec": row["infer_time_sec"] - base_mean["infer_time_sec"],
                })

            # ---- after 5 seeds: KD MEAN ----
            df_kd = pd.DataFrame(kd_rows_this)
            kd_mean = {
                "acc": float(df_kd["acc"].mean()),
                "f1": float(df_kd["f1"].mean()),
                "recall": float(df_kd["recall"].mean()),
                "train_time_sec": float(df_kd["train_time_sec"].mean()),
                "infer_time_sec": float(df_kd["infer_time_sec"].mean()),
            }

            delta_abs = {
                "acc": kd_mean["acc"] - base_mean["acc"],
                "f1": kd_mean["f1"] - base_mean["f1"],
                "recall": kd_mean["recall"] - base_mean["recall"],
                "train_time_sec": kd_mean["train_time_sec"] - base_mean["train_time_sec"],
                "infer_time_sec": kd_mean["infer_time_sec"] - base_mean["infer_time_sec"],
            }

            delta_pct = {
                "acc": pct_delta(kd_mean["acc"], base_mean["acc"]),
                "f1": pct_delta(kd_mean["f1"], base_mean["f1"]),
                "recall": pct_delta(kd_mean["recall"], base_mean["recall"]),
                "train_time_sec": pct_delta(kd_mean["train_time_sec"], base_mean["train_time_sec"]),
                "infer_time_sec": pct_delta(kd_mean["infer_time_sec"], base_mean["infer_time_sec"]),
            }

            print("\n" + "=" * 80)
            print(f"[CONFIG SUMMARY] alpha={alpha} | top_m={top_m} | seeds={kd_seeds}")
            print("=" * 80)
            print_delta_block(delta_abs, delta_pct)
            print_summary_table(teacher_stat, base_mean, kd_mean, delta_abs)

            # save config mean + delta mean to full
            results_full.append({
                "seed": "KD_MEAN",
                "alpha": alpha,
                "top_m": int(top_m),
                "model": "Student-KD(MEAN)",
                **kd_mean,
            })
            results_full.append({
                "seed": "ΔMEAN(KD-Base)",
                "alpha": alpha,
                "top_m": int(top_m),
                "model": "Δ(KDMEAN-BaseMEAN)",
                "acc": delta_abs["acc"],
                "f1": delta_abs["f1"],
                "recall": delta_abs["recall"],
                "train_time_sec": delta_abs["train_time_sec"],
                "infer_time_sec": delta_abs["infer_time_sec"],
            })

            # ----------------------------
            # summary_kd_mean_by_alpha_topm.csv
            # ----------------------------
            kd_mean_rows.append({
                "alpha": alpha,
                "top_m": int(top_m),
                **kd_mean,
            })

            # ----------------------------
            # summary_kd_delta_vs_base.csv
            #  - numeric columns + signed-string columns (*_s)
            # ----------------------------
            kd_delta_rows.append({
                "alpha": alpha,
                "top_m": int(top_m),

                # absolute delta (numeric)
                "d_acc": delta_abs["acc"],
                "d_f1": delta_abs["f1"],
                "d_recall": delta_abs["recall"],
                "d_train_time_sec": delta_abs["train_time_sec"],
                "d_infer_time_sec": delta_abs["infer_time_sec"],

                # absolute delta (signed string)
                "d_acc_s": fmt_abs6(delta_abs["acc"]),
                "d_f1_s": fmt_abs6(delta_abs["f1"]),
                "d_recall_s": fmt_abs6(delta_abs["recall"]),
                "d_train_time_sec_s": fmt_time2(delta_abs["train_time_sec"]),
                "d_infer_time_sec_s": fmt_time6(delta_abs["infer_time_sec"]),

                # percent delta (numeric, unit = %)
                "pct_acc": delta_pct["acc"],
                "pct_f1": delta_pct["f1"],
                "pct_recall": delta_pct["recall"],
                "pct_train_time_sec": delta_pct["train_time_sec"],
                "pct_infer_time_sec": delta_pct["infer_time_sec"],

                # percent delta (signed string)
                "pct_acc_s": fmt_pct2(delta_pct["acc"]),
                "pct_f1_s": fmt_pct2(delta_pct["f1"]),
                "pct_recall_s": fmt_pct2(delta_pct["recall"]),
                "pct_train_time_sec_s": fmt_pct2(delta_pct["train_time_sec"]),
                "pct_infer_time_sec_s": fmt_pct2(delta_pct["infer_time_sec"]),
            })

            # ----------------------------
            # ranked table rows
            # ----------------------------
            ranked_rows.append({
                "alpha": alpha,
                "top_m": int(top_m),

                # percent deltas numeric
                "pct_f1": delta_pct["f1"],
                "pct_acc": delta_pct["acc"],
                "pct_recall": delta_pct["recall"],
                "pct_train_time_sec": delta_pct["train_time_sec"],
                "pct_infer_time_sec": delta_pct["infer_time_sec"],

                # percent deltas signed strings
                "pct_f1_s": fmt_pct2(delta_pct["f1"]),
                "pct_acc_s": fmt_pct2(delta_pct["acc"]),
                "pct_recall_s": fmt_pct2(delta_pct["recall"]),
                "pct_train_time_sec_s": fmt_pct2(delta_pct["train_time_sec"]),
                "pct_infer_time_sec_s": fmt_pct2(delta_pct["infer_time_sec"]),

                # absolute deltas numeric
                "d_f1": delta_abs["f1"],
                "d_acc": delta_abs["acc"],
                "d_recall": delta_abs["recall"],
                "d_train_time_sec": delta_abs["train_time_sec"],
                "d_infer_time_sec": delta_abs["infer_time_sec"],

                # absolute deltas signed strings
                "d_f1_s": fmt_abs6(delta_abs["f1"]),
                "d_acc_s": fmt_abs6(delta_abs["acc"]),
                "d_recall_s": fmt_abs6(delta_abs["recall"]),
                "d_train_time_sec_s": fmt_time2(delta_abs["train_time_sec"]),
                "d_infer_time_sec_s": fmt_time6(delta_abs["infer_time_sec"]),
            })

    # ========================================================
    # Save CSVs
    # ========================================================
    df_full = pd.DataFrame(results_full)
    df_full.to_csv("kd_final_results_full.csv", index=False)
    print("\nSaved: kd_final_results_full.csv")

    df_base_summary = pd.DataFrame(base_mean_rows)
    df_base_summary.to_csv("summary_base_mean.csv", index=False)
    print("Saved: summary_base_mean.csv")

    df_kd_mean = pd.DataFrame(kd_mean_rows)
    df_kd_mean.to_csv("summary_kd_mean_by_alpha_topm.csv", index=False)
    print("Saved: summary_kd_mean_by_alpha_topm.csv")

    df_kd_delta = pd.DataFrame(kd_delta_rows)
    df_kd_delta.to_csv("summary_kd_delta_vs_base.csv", index=False)
    print("Saved: summary_kd_delta_vs_base.csv")

    # ========================================================
    # Ranking: best -> worst (by pct_f1, pct_acc, pct_recall)
    # ========================================================
    df_rank = pd.DataFrame(ranked_rows)
    df_rank = df_rank.sort_values(
        by=["pct_f1", "pct_acc", "pct_recall"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    df_rank.to_csv("kd_ranked_by_improvement.csv", index=False)
    print("Saved: kd_ranked_by_improvement.csv")

    df_top3 = df_rank.head(3).copy()
    df_top3.to_csv("kd_top3_configs.csv", index=False)
    print("Saved: kd_top3_configs.csv")

    # show top-3 on terminal (compact)
    print("\n" + "=" * 80)
    print("TOP 3 CONFIGS (by %ΔF1 -> %ΔAcc -> %ΔRecall)")
    print("=" * 80)
    print(df_top3[
        [
            "alpha", "top_m",
            "pct_f1_s", "pct_acc_s", "pct_recall_s",
            "pct_train_time_sec_s", "pct_infer_time_sec_s",
        ]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
