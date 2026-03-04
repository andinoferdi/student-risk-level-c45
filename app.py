import argparse
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split


METRIC_COLUMNS = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]


def _entropy(y: list[str] | pd.Series) -> float:
    n = len(y)
    if n == 0:
        return 0.0
    counts = Counter(y)
    ent = 0.0
    for c in counts.values():
        p = c / n
        if p > 0:
            ent -= p * math.log2(p)
    return ent


def _split_info(sizes: list[int]) -> float:
    n = sum(sizes)
    if n == 0:
        return 0.0
    si = 0.0
    for s in sizes:
        if s == 0:
            continue
        p = s / n
        si -= p * math.log2(p)
    return si


def _info_gain(parent_y: list[str], child_ys: list[list[str]]) -> float:
    n = len(parent_y)
    if n == 0:
        return 0.0
    parent_ent = _entropy(parent_y)
    weighted_ent = 0.0
    for cy in child_ys:
        weighted_ent += (len(cy) / n) * _entropy(cy)
    return parent_ent - weighted_ent


def _gain_ratio(parent_y: list[str], child_ys: list[list[str]]) -> float:
    ig = _info_gain(parent_y, child_ys)
    si = _split_info([len(cy) for cy in child_ys])
    if si <= 1e-12:
        return 0.0
    return ig / si


@dataclass
class Node:
    is_leaf: bool
    prediction: str
    class_counts: dict[str, int]
    feature: str | None = None
    feature_type: str | None = None
    threshold: float | None = None
    children: dict[str, "Node"] = field(default_factory=dict)
    left: "Node | None" = None
    right: "Node | None" = None
    depth: int = 0

    def predict_row(self, row: dict[str, Any]) -> str:
        if self.is_leaf:
            return self.prediction

        if self.feature_type == "numeric":
            val = row.get(self.feature)
            if val is None or (isinstance(val, float) and math.isnan(val)):
                return self.prediction
            if float(val) <= self.threshold:
                return self.left.predict_row(row) if self.left else self.prediction
            return self.right.predict_row(row) if self.right else self.prediction

        val = row.get(self.feature)
        if val in self.children:
            return self.children[val].predict_row(row)
        return self.prediction

    def predict_df(self, X: pd.DataFrame) -> list[str]:
        rows = X.to_dict(orient="records")
        return [self.predict_row(r) for r in rows]

    def make_leaf(self) -> None:
        self.is_leaf = True
        self.feature = None
        self.feature_type = None
        self.threshold = None
        self.children = {}
        self.left = None
        self.right = None

    def to_rules(self, indent: str = "") -> str:
        if self.is_leaf:
            n = sum(self.class_counts.values())
            return f"{indent}THEN class = {self.prediction} (n={n})"

        if self.feature_type == "numeric":
            return "\n".join(
                [
                    f"{indent}IF {self.feature} <= {self.threshold:.6g}",
                    self.left.to_rules(indent + "  "),
                    f"{indent}ELSE  # {self.feature} > {self.threshold:.6g}",
                    self.right.to_rules(indent + "  "),
                ]
            )

        rules = [f"{indent}SPLIT {self.feature}"]
        for value, child in self.children.items():
            rules.append(f"{indent}IF {self.feature} == {repr(value)}")
            rules.append(child.to_rules(indent + "  "))
        rules.append(f"{indent}ELSE")
        rules.append(f"{indent}  THEN class = {self.prediction} (fallback)")
        return "\n".join(rules)


class C45Classifier:
    def __init__(self, max_depth: int = 12, min_samples_split: int = 10, min_gain_ratio: float = 1e-6):
        self.max_depth = int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.min_gain_ratio = float(min_gain_ratio)
        self.root: Node | None = None
        self.feature_types_: dict[str, str] = {}

    @staticmethod
    def _majority_class(y: pd.Series) -> str:
        counts = Counter(y)
        return sorted(counts.items(), key=lambda kv: (-kv[1], str(kv[0])))[0][0]

    def _best_split_numeric(self, x: pd.Series, y: pd.Series) -> tuple[float, float | None]:
        arr = x.to_numpy(dtype=float, copy=False)
        order = np.argsort(arr)
        arr_sorted = arr[order]
        y_sorted = y.to_numpy(copy=False)[order]

        if np.all(arr_sorted == arr_sorted[0]):
            return 0.0, None

        thresholds: list[float] = []
        for i in range(len(arr_sorted) - 1):
            if y_sorted[i] != y_sorted[i + 1] and arr_sorted[i] != arr_sorted[i + 1]:
                thresholds.append((arr_sorted[i] + arr_sorted[i + 1]) / 2.0)

        if not thresholds:
            uniq = np.unique(arr_sorted)
            if len(uniq) <= 1:
                return 0.0, None
            step = max(1, (len(uniq) - 1) // 50)
            thresholds = [(uniq[i] + uniq[i + 1]) / 2.0 for i in range(0, len(uniq) - 1, step)]

        parent_y = y.tolist()
        best_gr, best_t = 0.0, None
        for threshold in thresholds:
            left_mask = arr <= threshold
            right_mask = ~left_mask
            if left_mask.sum() == 0 or right_mask.sum() == 0:
                continue
            gr = _gain_ratio(parent_y, [y[left_mask].tolist(), y[right_mask].tolist()])
            if gr > best_gr:
                best_gr, best_t = gr, float(threshold)
        return best_gr, best_t

    def _best_split_categorical(self, x: pd.Series, y: pd.Series) -> float:
        groups = [yy.tolist() for _, yy in y.groupby(x)]
        if len(groups) <= 1:
            return 0.0
        return _gain_ratio(y.tolist(), groups)

    def _best_split(
        self, X: pd.DataFrame, y: pd.Series, available_features: list[str]
    ) -> dict[str, Any]:
        best = {"feature": None, "type": None, "threshold": None, "gain_ratio": 0.0}
        for feat in available_features:
            ftype = self.feature_types_[feat]
            if ftype == "numeric":
                gr, threshold = self._best_split_numeric(X[feat], y)
                if gr > best["gain_ratio"] and threshold is not None:
                    best = {
                        "feature": feat,
                        "type": "numeric",
                        "threshold": threshold,
                        "gain_ratio": gr,
                    }
            else:
                gr = self._best_split_categorical(X[feat], y)
                if gr > best["gain_ratio"]:
                    best = {
                        "feature": feat,
                        "type": "categorical",
                        "threshold": None,
                        "gain_ratio": gr,
                    }
        return best

    def _build(
        self, X: pd.DataFrame, y: pd.Series, depth: int, available_features: list[str]
    ) -> Node:
        node = Node(
            is_leaf=False,
            prediction=self._majority_class(y),
            class_counts=dict(Counter(y)),
            depth=depth,
        )

        if len(set(y)) == 1:
            node.is_leaf = True
            return node
        if depth >= self.max_depth:
            node.is_leaf = True
            return node
        if len(y) < self.min_samples_split:
            node.is_leaf = True
            return node
        if not available_features:
            node.is_leaf = True
            return node

        best = self._best_split(X, y, available_features)
        if best["feature"] is None or best["gain_ratio"] < self.min_gain_ratio:
            node.is_leaf = True
            return node

        feat = best["feature"]
        node.feature = feat
        node.feature_type = best["type"]

        if node.feature_type == "numeric":
            node.threshold = best["threshold"]
            left_idx = X[feat] <= node.threshold
            right_idx = ~left_idx
            node.left = self._build(X[left_idx], y[left_idx], depth + 1, available_features)
            node.right = self._build(X[right_idx], y[right_idx], depth + 1, available_features)
            return node

        new_features = [f for f in available_features if f != feat]
        for value in X[feat].dropna().unique().tolist():
            idx = X[feat] == value
            child_X = X.loc[idx].drop(columns=[feat])
            node.children[value] = self._build(child_X, y[idx], depth + 1, new_features)
        return node

    def fit(self, X: pd.DataFrame, y: pd.Series, feature_types: dict[str, str]) -> "C45Classifier":
        self.feature_types_ = feature_types.copy()
        self.root = self._build(X, y, depth=0, available_features=list(X.columns))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.root is None:
            raise RuntimeError("Model belum dilatih. Jalankan fit() terlebih dulu.")
        return np.array(self.root.predict_df(X))

    def prune(self, X_val: pd.DataFrame, y_val: pd.Series) -> "C45Classifier":
        if self.root is None or len(y_val) == 0:
            return self

        def _accuracy(y_true: pd.Series, y_pred: np.ndarray) -> float:
            return float((np.asarray(y_true) == np.asarray(y_pred)).mean()) if len(y_true) else 0.0

        def _prune(node: Node, Xv: pd.DataFrame, yv: pd.Series) -> None:
            if node.is_leaf or len(yv) == 0:
                return

            if node.feature_type == "numeric":
                left_mask = Xv[node.feature] <= node.threshold
                right_mask = ~left_mask
                _prune(node.left, Xv[left_mask], yv[left_mask])
                _prune(node.right, Xv[right_mask], yv[right_mask])
            else:
                for value, child in list(node.children.items()):
                    idx = Xv[node.feature] == value
                    _prune(child, Xv.loc[idx].drop(columns=[node.feature]), yv[idx])

            acc_subtree = _accuracy(yv, np.array(node.predict_df(Xv)))
            acc_leaf = _accuracy(yv, np.array([node.prediction] * len(yv)))
            if acc_leaf >= acc_subtree:
                node.make_leaf()

        _prune(self.root, X_val, y_val)
        return self


def clean_and_prepare(df: pd.DataFrame, target_col: str, drop_cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    df = df.drop_duplicates()

    if target_col not in df.columns:
        raise ValueError(f"Kolom target '{target_col}' tidak ditemukan. Kolom tersedia: {list(df.columns)}")

    for col in drop_cols:
        if col in df.columns:
            df = df.drop(columns=[col])

    df = df.dropna(subset=[target_col])

    for col in df.columns:
        if col == target_col:
            continue
        if df[col].dtype == "object":
            normalized = df[col].astype(str).str.strip().str.replace(",", ".", regex=False)
            as_num = pd.to_numeric(normalized, errors="coerce")
            if as_num.notna().mean() >= 0.95:
                df[col] = as_num
            else:
                df[col] = normalized

    for col in df.columns:
        if col == target_col:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna("Unknown").astype(str)

    df[target_col] = df[target_col].astype(str).str.strip()
    return df


def infer_feature_types(df: pd.DataFrame, target_col: str) -> dict[str, str]:
    feature_types: dict[str, str] = {}
    for col in df.columns:
        if col == target_col:
            continue
        feature_types[col] = "numeric" if pd.api.types.is_numeric_dtype(df[col]) else "categorical"
    return feature_types


def resolve_data_path(data_arg: str) -> Path:
    given = Path(data_arg)
    base_dir = Path(__file__).resolve().parent
    candidates = [given]
    if not given.is_absolute():
        candidates.append(base_dir / given)
        if len(given.parts) == 1:
            candidates.append(base_dir / "dataset" / given)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    tried = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Dataset tidak ditemukan. Lokasi yang dicoba: {tried}")


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("Nilai --save_plots harus true/false.")


def run_single_split_evaluation(
    X: pd.DataFrame,
    y: pd.Series,
    feature_types: dict[str, str],
    test_size: float,
    val_size: float,
    random_state: int,
    max_depth: int,
    min_samples_split: int,
    min_gain_ratio: float,
) -> dict[str, Any]:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        test_size=val_size,
        random_state=random_state,
        stratify=y_train,
    )

    model = C45Classifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_gain_ratio=min_gain_ratio,
    )
    model.fit(X_tr, y_tr, feature_types=feature_types)
    model.prune(X_val, y_val)

    y_pred = model.predict(X_test)
    labels = sorted(y.unique().tolist())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    report = classification_report(y_test, y_pred, labels=labels, zero_division=0)

    return {
        "random_state": random_state,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_train_internal": len(X_tr),
        "n_val_internal": len(X_val),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "labels": labels,
        "cm": cm,
        "classification_report": report,
        "model": model,
    }


def run_repeated_evaluation(
    X: pd.DataFrame,
    y: pd.Series,
    feature_types: dict[str, str],
    test_size: float,
    val_size: float,
    n_repeats: int,
    seed: int,
    max_depth: int,
    min_samples_split: int,
    min_gain_ratio: float,
) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for repeat in range(n_repeats):
        random_state = seed + repeat
        result = run_single_split_evaluation(
            X=X,
            y=y,
            feature_types=feature_types,
            test_size=test_size,
            val_size=val_size,
            random_state=random_state,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_gain_ratio=min_gain_ratio,
        )
        rows.append(
            {
                "run": repeat + 1,
                "random_state": random_state,
                "accuracy": result["accuracy"],
                "precision_macro": result["precision_macro"],
                "recall_macro": result["recall_macro"],
                "f1_macro": result["f1_macro"],
            }
        )
    return pd.DataFrame(rows)


def summarize_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        metrics_df[METRIC_COLUMNS]
        .agg(["mean", "std", "min", "max"])
        .transpose()
        .reset_index()
        .rename(columns={"index": "metric"})
    )
    return summary


def check_label_determinism(df: pd.DataFrame, target_col: str) -> dict[str, Any]:
    required_columns = {"ipk", "tingkat_kehadiran", target_col}
    if not required_columns.issubset(set(df.columns)):
        return {
            "status": "skipped",
            "message": "Kolom ipk atau tingkat_kehadiran tidak lengkap, diagnostik dilewati.",
        }

    attendance = df["tingkat_kehadiran"].astype(float)
    gpa = df["ipk"].astype(float)
    predicted = np.select(
        [
            attendance <= 0.745,
            (attendance > 0.745) & (gpa <= 2.495),
            (attendance > 0.745) & (gpa > 2.495) & (attendance <= 0.845),
            (attendance > 0.845) & (gpa <= 2.995),
        ],
        ["Tinggi", "Tinggi", "Sedang", "Sedang"],
        default="Rendah",
    )
    actual = df[target_col].astype(str).str.strip().to_numpy()
    matches = predicted == actual
    return {
        "status": "ok",
        "match_ratio": float(matches.mean()),
        "mismatch_count": int((~matches).sum()),
        "total_rows": int(len(df)),
        "is_perfect": bool(matches.all()),
    }


def save_metrics(metrics_df: pd.DataFrame, summary_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(output_dir / "metrics_per_run.csv", index=False)
    summary_df.to_csv(output_dir / "metrics_summary.csv", index=False)


def save_visualizations(
    df: pd.DataFrame,
    target_col: str,
    representative_result: dict[str, Any],
    metrics_df: pd.DataFrame,
    output_dir: Path,
    save_plots: bool,
) -> list[Path]:
    if not save_plots:
        return []

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_files: list[Path] = []
    sns.set_theme(style="whitegrid")

    target_counts = df[target_col].value_counts().rename_axis("kelas").reset_index(name="jumlah")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=target_counts, x="kelas", y="jumlah", hue="kelas", dodge=False, legend=False, ax=ax)
    ax.set_title("Distribusi Kelas Tingkat Risiko Kelulusan Mahasiswa")
    ax.set_xlabel("Kelas")
    ax.set_ylabel("Jumlah Data")
    target_dist_path = output_dir / "target_distribution.png"
    fig.tight_layout()
    fig.savefig(target_dist_path, dpi=150)
    plt.close(fig)
    saved_files.append(target_dist_path)

    cm_df = pd.DataFrame(
        representative_result["cm"],
        index=[f"aktual_{label}" for label in representative_result["labels"]],
        columns=[f"prediksi_{label}" for label in representative_result["labels"]],
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("Confusion Matrix Representatif Tingkat Risiko Kelulusan Mahasiswa")
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")
    cm_path = output_dir / "confusion_matrix_representative.png"
    fig.tight_layout()
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    saved_files.append(cm_path)

    metric_long = metrics_df.melt(id_vars=["run"], value_vars=METRIC_COLUMNS, var_name="metric", value_name="value")
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(data=metric_long, x="metric", y="value", hue="metric", dodge=False, legend=False, ax=ax)
    ax.set_title("Distribusi Metrik Repeated Evaluation")
    ax.set_xlabel("Metrik")
    ax.set_ylabel("Skor")
    ax.set_ylim(0, 1.05)
    boxplot_path = output_dir / "metrics_boxplot.png"
    fig.tight_layout()
    fig.savefig(boxplot_path, dpi=150)
    plt.close(fig)
    saved_files.append(boxplot_path)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=metric_long, x="run", y="value", hue="metric", marker="o", ax=ax)
    ax.set_title("Tren Metrik per Run pada Tingkat Risiko Kelulusan Mahasiswa")
    ax.set_xlabel("Run")
    ax.set_ylabel("Skor")
    ax.set_ylim(0, 1.05)
    trend_path = output_dir / "metrics_trend.png"
    fig.tight_layout()
    fig.savefig(trend_path, dpi=150)
    plt.close(fig)
    saved_files.append(trend_path)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x="ipk",
        y="tingkat_kehadiran",
        hue=target_col,
        alpha=0.75,
        ax=ax,
    )
    ax.set_title("Sebaran IPK dan Kehadiran untuk Tingkat Risiko Kelulusan Mahasiswa")
    ax.set_xlabel("IPK")
    ax.set_ylabel("Tingkat Kehadiran")
    scatter_path = output_dir / "feature_scatter_ipk_kehadiran.png"
    fig.tight_layout()
    fig.savefig(scatter_path, dpi=150)
    plt.close(fig)
    saved_files.append(scatter_path)

    return saved_files


def print_metric_summary(summary_df: pd.DataFrame) -> None:
    for _, row in summary_df.iterrows():
        print(
            f"{row['metric']:<16} mean={row['mean']:.4f}, std={row['std']:.4f}, "
            f"min={row['min']:.4f}, max={row['max']:.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data_manajemen_perguruan_tinggi.csv")
    parser.add_argument("--target", default="tingkat_risiko")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--val_size", type=float, default=0.25)
    parser.add_argument("--max_depth", type=int, default=12)
    parser.add_argument("--min_samples_split", type=int, default=10)
    parser.add_argument("--min_gain_ratio", type=float, default=1e-6)
    parser.add_argument("--n_repeats", type=int, default=30)
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--save_plots", type=parse_bool, nargs="?", const=True, default=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.n_repeats < 1:
        raise ValueError("Parameter --n_repeats harus >= 1.")

    print("--- ANALISIS TINGKAT RISIKO KELULUSAN MAHASISWA ---")
    print("\n1. Memuat dataset")
    data_path = resolve_data_path(args.data)
    df_raw = pd.read_csv(data_path)
    print(f"Lokasi data: {data_path}")
    print(f"Total data awal: {len(df_raw)} baris, {df_raw.shape[1]} kolom")

    print("\n2. Seleksi data dan pembersihan")
    df = clean_and_prepare(df_raw, target_col=args.target, drop_cols=["id_siswa"])
    print(f"Kolom setelah seleksi: {list(df.columns)}")
    print("Distribusi kelas target:")
    print(df[args.target].value_counts())

    print("\n3. Pemahaman data")
    print(df.info())
    print(df.describe(include="all").transpose().head(12))

    X = df.drop(columns=[args.target])
    y = df[args.target]
    feature_types = infer_feature_types(df, target_col=args.target)
    num_feats = [k for k, v in feature_types.items() if v == "numeric"]
    cat_feats = [k for k, v in feature_types.items() if v == "categorical"]
    print(f"Fitur numerik ({len(num_feats)}): {num_feats}")
    print(f"Fitur kategorikal ({len(cat_feats)}): {cat_feats}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n4. Single split evaluasi representatif")
    representative_result = run_single_split_evaluation(
        X=X,
        y=y,
        feature_types=feature_types,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.seed,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_gain_ratio=args.min_gain_ratio,
    )
    print(f"Random state representatif: {representative_result['random_state']}")
    print(f"Jumlah data latih: {representative_result['n_train']}")
    print(f"Jumlah data uji: {representative_result['n_test']}")
    print(f"Train internal: {representative_result['n_train_internal']}")
    print(f"Validation internal: {representative_result['n_val_internal']}")
    print(f"Akurasi : {representative_result['accuracy']:.4f}")
    print(f"Presisi : {representative_result['precision_macro']:.4f} (macro)")
    print(f"Recall  : {representative_result['recall_macro']:.4f} (macro)")
    print(f"F1-score: {representative_result['f1_macro']:.4f} (macro)")
    cm_df = pd.DataFrame(
        representative_result["cm"],
        index=[f"aktual_{label}" for label in representative_result["labels"]],
        columns=[f"prediksi_{label}" for label in representative_result["labels"]],
    )
    print("\nConfusion matrix representatif:")
    print(cm_df)
    print("\nLaporan klasifikasi representatif:")
    print(representative_result["classification_report"])

    print("\n5. Repeated stratified evaluation")
    metrics_df = run_repeated_evaluation(
        X=X,
        y=y,
        feature_types=feature_types,
        test_size=args.test_size,
        val_size=args.val_size,
        n_repeats=args.n_repeats,
        seed=args.seed,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_gain_ratio=args.min_gain_ratio,
    )
    summary_df = summarize_metrics(metrics_df)
    print("Ringkasan metrik utama:")
    print_metric_summary(summary_df)

    save_metrics(metrics_df, summary_df, output_dir)
    print(f"\nFile metrik tersimpan di: {output_dir}")
    print(f"- {output_dir / 'metrics_per_run.csv'}")
    print(f"- {output_dir / 'metrics_summary.csv'}")

    print("\n6. Diagnostik deterministik label")
    diagnostic = check_label_determinism(df, args.target)
    if diagnostic["status"] == "ok":
        print(
            "Kecocokan rule deterministik (ipk + tingkat_kehadiran) "
            f"terhadap label: {diagnostic['match_ratio']:.4f} "
            f"({diagnostic['total_rows'] - diagnostic['mismatch_count']}/{diagnostic['total_rows']})"
        )
        perfect_metrics = np.isclose(metrics_df[METRIC_COLUMNS], 1.0).all().all()
        if perfect_metrics:
            if diagnostic["is_perfect"]:
                print(
                    "Interpretasi: metrik sempurna konsisten dengan pola label yang deterministik, "
                    "bukan karena manipulasi skor."
                )
            else:
                print(
                    "Interpretasi: metrik sempurna perlu ditinjau ulang karena rule deterministik "
                    "tidak sepenuhnya cocok dengan label."
                )
    else:
        print(diagnostic["message"])

    print("\n7. Aturan pohon keputusan representatif")
    print(representative_result["model"].root.to_rules())

    print("\n8. Visualisasi")
    saved_plots = save_visualizations(
        df=df,
        target_col=args.target,
        representative_result=representative_result,
        metrics_df=metrics_df,
        output_dir=output_dir,
        save_plots=args.save_plots,
    )
    if saved_plots:
        print("Grafik tersimpan:")
        for plot_path in saved_plots:
            print(f"- {plot_path}")
    else:
        print("Penyimpanan grafik dinonaktifkan karena --save_plots=false.")


if __name__ == "__main__":
    main()
