# xconvgen_optuna_sensitivity.py
import os
import json
import random
import numpy as np
import pandas as pd
import optuna
import matplotlib.pyplot as plt
from datetime import datetime

# ==== your libs ====
from library.generators.XConvGeN import XConvGeN, GeneratorConfig
from fdc.fdc import FDC
# ====================

# ML stack
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

# ----------------------------
# Reproducibility
# ----------------------------
import tensorflow as tf
import keras
import torch

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    if tf.__version__.startswith('2'):
        tf.keras.backend.clear_session()
        tf.random.set_seed(seed)
        keras.utils.set_random_seed(seed)
    else:
        tf.compat.v1.keras.backend.clear_session()
        tf.compat.v1.set_random_seed(seed)
        keras.backend.set_random_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    gpu_devices = tf.config.list_physical_devices('GPU')
    if gpu_devices:
        tf.config.experimental.set_memory_growth(gpu_devices[0], True)

# ----------------------------
# Paths & constants
# ----------------------------
BASE_DIR = "PreparedData1"
DATASET_NAME = "Stroke"             # current code only runs this dataset
TASK = "supervised"
TRAIN_CSV = "training_data.csv"
HOLDOUT_CSV = "holdout_data.csv"
INFO_JSON = "additional_info.json"

# output dir for *plots and report only* (never synthetic data)
OUTDIR = "xconvgen_optuna_outputs"
os.makedirs(OUTDIR, exist_ok=True)

# optuna
N_TRIALS = 50
RANDOM_SEED = 42
DIRECTION = "minimize"  # minimize |F1(real_model) - F1(synth_model)|

# evaluation choice
F1_AVG = "macro"  # macro is robust if classes are imbalanced/multiclass

# ----------------------------
# Data utilities
# ----------------------------
def load_dataset():
    task_dir = os.path.join(BASE_DIR, DATASET_NAME, TASK)
    train_path = os.path.join(task_dir, TRAIN_CSV)
    holdout_path = os.path.join(task_dir, HOLDOUT_CSV)
    info_path = os.path.join(task_dir, INFO_JSON)

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Missing: {train_path}")
    if not os.path.exists(holdout_path):
        raise FileNotFoundError(f"Missing: {holdout_path}")
    if not os.path.exists(info_path):
        raise FileNotFoundError(f"Missing: {info_path}")

    train_df = pd.read_csv(train_path)
    holdout_df = pd.read_csv(holdout_path)
    with open(info_path, "r") as f:
        info = json.load(f)
    return train_df, holdout_df, info

def get_target_name(info):
    target = info.get("target")
    if target is None:
        raise ValueError("info['target'] is None; required for evaluation.")
    if isinstance(target, list):
        target = target[0]
    return target

def meta_indices(info, ordered_features):
    """Prepare FDC index lists."""
    ord_list = info.get("indices_ordinal_features", []) or []
    nom_list = info.get("indices_nominal_features", []) or []
    target = get_target_name(info)
    if target in ordered_features:
        nom_list = nom_list + [ordered_features.index(target)]
    cont_list = info.get("indices_continuous_features", []) or []
    return ord_list, nom_list, cont_list

def balance_syn_data(syn_data: pd.DataFrame, value_count: dict, label: str) -> pd.DataFrame:
    df_list = []
    for class_label in value_count:
        cl = class_label
        if isinstance(cl, str):
            try:
                cl = float(cl)
            except ValueError:
                pass
        if isinstance(cl, float):
            cl = int(cl)
        class_df = syn_data[syn_data[label] == cl].sample(
            n=value_count[str(cl)], axis=0, random_state=RANDOM_SEED
        )
        df_list.append(class_df)
    balanced_synthetic_data = pd.concat(df_list)
    return balanced_synthetic_data.sample(frac=1, random_state=RANDOM_SEED)

def one_hot_align(train_df: pd.DataFrame, test_df: pd.DataFrame, target: str):
    """One-hot encode categoricals (if any) and align columns between train & test."""
    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]
    X_test = test_df.drop(columns=[target])
    y_test = test_df[target]

    X_train_enc = pd.get_dummies(X_train, drop_first=False)
    X_test_enc = pd.get_dummies(X_test, drop_first=False)

    # align columns
    X_train_enc, X_test_enc = X_train_enc.align(X_test_enc, join="left", axis=1, fill_value=0)
    # If test has extra cols (shouldn't after align-left), align again other way:
    X_test_enc = X_test_enc.reindex(columns=X_train_enc.columns, fill_value=0)
    return X_train_enc, y_train, X_test_enc, y_test

def evaluate_f1_gap(real_train: pd.DataFrame, holdout: pd.DataFrame,
                    synth_balanced: pd.DataFrame, target: str) -> float:
    """Train two GradientBoostingClassifier models and compute |F1_real - F1_synth| on the same holdout."""

    # model trained on REAL
    Xtr_r, ytr_r, Xte_r, yte = one_hot_align(real_train, holdout, target)
    m_real = GradientBoostingClassifier(random_state=RANDOM_SEED)
    m_real.fit(Xtr_r, ytr_r)
    preds_r = m_real.predict(Xte_r)
    f1_real = f1_score(yte, preds_r, average=F1_AVG)

    # model trained on SYNTH (balanced)
    Xtr_s, ytr_s, Xte_s, yte2 = one_hot_align(synth_balanced, holdout, target)
    m_syn = GradientBoostingClassifier(random_state=RANDOM_SEED)
    m_syn.fit(Xtr_s, ytr_s)
    preds_s = m_syn.predict(Xte_s)
    f1_syn = f1_score(yte2, preds_s, average=F1_AVG)

    return abs(f1_real - f1_syn), f1_real, f1_syn

# ----------------------------
# Optuna objective
# ----------------------------
def make_objective(train_df: pd.DataFrame, holdout_df: pd.DataFrame, info: dict):
    ordered_features = info["ordered_features"].copy()
    target = get_target_name(info)
    if target not in ordered_features:
        ordered_features.append(target)

    # FDC setup
    fdc = FDC()
    ord_list, nom_list, cont_list = meta_indices(info, ordered_features)
    fdc.ord_list = ord_list
    fdc.nom_list = nom_list
    fdc.cont_list = cont_list

    # training data for generator
    train_arr = np.array(train_df[ordered_features].values, dtype=float)
    n_syn_samples = train_arr.shape[0] * 5

    target_value_counts = info["target_value_counts"]

    def objective(trial: optuna.trial.Trial):
        seed_everything(RANDOM_SEED + trial.number)

        neb = trial.suggest_int("neb", 2, 10)
        gen = trial.suggest_int("gen", neb, 20)  # ensure gen >= neb
        neb_epochs = trial.suggest_int("neb_epochs", 2, 50)
        alpha_clip = trial.suggest_float("alpha_clip", 0.0, 0.9)

        try:
            config = GeneratorConfig(
                n_feat=train_arr.shape[1],
                neb=neb,
                gen=gen,
                neb_epochs=neb_epochs,
                genAddNoise=False,
                alpha_clip=alpha_clip
            )
            model = XConvGeN(config=config, fdc=fdc, debug=False)
            model.reset(train_arr)
            model.train(train_arr)

            syn = model.generateData(n_syn_samples)
            syn_df = pd.DataFrame(syn, columns=ordered_features)

            # balance synthetic to match real target distribution
            syn_bal = balance_syn_data(syn_df, target_value_counts, target)

            gap, f1_real, f1_syn = evaluate_f1_gap(train_df[ordered_features], holdout_df[ordered_features], syn_bal, target)

            # Log some diagnostics in trial user attrs
            trial.set_user_attr("f1_real", float(f1_real))
            trial.set_user_attr("f1_syn", float(f1_syn))
            return float(gap)

        except Exception as e:
            # prune unpromising/failed trials
            raise optuna.exceptions.TrialPruned(f"Pruned due to error: {e}")

    return objective

# ----------------------------
# Run study
# ----------------------------
def run():
    seed_everything(RANDOM_SEED)
    train_df, holdout_df, info = load_dataset()

    # Ensure only columns we need (ordered_features + target)
    ordered_features = info["ordered_features"].copy()
    target = get_target_name(info)
    if target not in ordered_features:
        ordered_features.append(target)

    # make sure target is numeric if needed
    # (keep as-is; GradientBoosting handles ints/strings after one-hot. XConvGeN expects numeric arrays already.)

    study = optuna.create_study(
        study_name="xconvgen_sensitivity",
        direction=DIRECTION,
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED)
    )
    objective = make_objective(train_df, holdout_df, info)

    study.optimize(objective, n_trials=N_TRIALS, n_jobs=1, show_progress_bar=False)

    # ------------------------
    # Save artifacts
    # ------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # trials dataframe
    df_trials = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "state"))
    trials_csv = os.path.join(OUTDIR, f"trials_{timestamp}.csv")
    df_trials.to_csv(trials_csv, index=False)

    # best summary
    best_txt = os.path.join(OUTDIR, f"best_{timestamp}.txt")
    with open(best_txt, "w") as f:
        f.write("XConvGeN Sensitivity Analysis (Optuna)\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Direction: {DIRECTION}\n")
        f.write(f"Best value (|F1_real - F1_synth|): {study.best_value:.6f}\n")
        f.write("Best params:\n")
        for k, v in study.best_params.items():
            f.write(f"  - {k}: {v}\n")
        # add last trial's f1s if present
        tr = study.best_trial
        f1r = tr.user_attrs.get("f1_real")
        f1s = tr.user_attrs.get("f1_syn")
        if f1r is not None and f1s is not None:
            f.write(f"Best trial F1_real: {f1r:.6f}\n")
            f.write(f"Best trial F1_synth: {f1s:.6f}\n")

    # matplotlib-based visualizations (no browser, saved as PNG)
    from optuna.visualization.matplotlib import (
        plot_optimization_history,
        plot_param_importances,
        plot_slice,
        plot_contour,
    )

    plt.figure()
    plot_optimization_history(study)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, f"opt_history_{timestamp}.png"), dpi=180)
    plt.close()

    plt.figure()
    try:
        plot_param_importances(study)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, f"param_importances_{timestamp}.png"), dpi=180)
    except Exception:
        # importance may fail if not enough completed trials
        pass
    plt.close()

    plt.figure()
    plot_slice(study)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, f"slice_{timestamp}.png"), dpi=180)
    plt.close()

    plt.figure()
    plot_contour(study, params=["neb", "gen", "neb_epochs", "alpha_clip"])
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, f"contour_{timestamp}.png"), dpi=180)
    plt.close()

    # Markdown analysis report
    report_md = os.path.join(OUTDIR, f"analysis_report_{timestamp}.md")
    with open(report_md, "w") as f:
        f.write(f"# XConvGeN Sensitivity Analysis (Optuna)\n\n")
        f.write(f"- **Date**: {timestamp}\n")
        f.write(f"- **Objective**: Minimize |F1(real) - F1(synth)| on holdout\n")
        f.write(f"- **F1 averaging**: `{F1_AVG}`\n")
        f.write(f"- **Trials**: {len(study.trials)}\n\n")
        f.write("## Best Result\n")
        f.write(f"- **Gap**: `{study.best_value:.6f}`\n")
        f.write("- **Best Params**:\n")
        for k, v in study.best_params.items():
            f.write(f"  - `{k}`: `{v}`\n")
        if "f1_real" in study.best_trial.user_attrs and "f1_syn" in study.best_trial.user_attrs:
            f.write(f"- **Best Trial F1(real)**: `{study.best_trial.user_attrs['f1_real']:.6f}`\n")
            f.write(f"- **Best Trial F1(synth)**: `{study.best_trial.user_attrs['f1_syn']:.6f}`\n")
        f.write("\n## Artifacts\n")
        f.write(f"- Trials CSV: `{os.path.basename(trials_csv)}`\n")
        f.write(f"- Plots: optimization history, param importances, slice, contour (PNG)\n")

    print(f"Done. Artifacts saved in: {OUTDIR}")
    print(f"- {os.path.basename(trials_csv)}")
    print(f"- {os.path.basename(best_txt)}")
    print(f"- analysis_report_{timestamp}.md and PNG plots")

if __name__ == "__main__":
    run()
