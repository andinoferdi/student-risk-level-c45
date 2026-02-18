import argparse
import math
from pathlib import Path
from dataclasses import dataclass, field
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support


# 1. PROSES: fungsi dasar untuk C4.5 (entropy, gain ratio)

def _entropy(y):
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


def _split_info(sizes):
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


def _info_gain(parent_y, child_ys):
    n = len(parent_y)
    if n == 0:
        return 0.0
    parent_ent = _entropy(parent_y)
    weighted_ent = 0.0
    for cy in child_ys:
        weighted_ent += (len(cy) / n) * _entropy(cy)
    return parent_ent - weighted_ent


def _gain_ratio(parent_y, child_ys):
    ig = _info_gain(parent_y, child_ys)
    si = _split_info([len(cy) for cy in child_ys])
    if si <= 1e-12:
        return 0.0
    return ig / si



# 2. PROSES: struktur Node untuk pohon keputusan

@dataclass
class Node:
    is_leaf: bool
    prediction: str
    class_counts: dict
    feature: str = None
    feature_type: str = None  # "numeric" atau "categorical"
    threshold: float = None
    children: dict = field(default_factory=dict)  # khusus categorical
    left: "Node" = None  # khusus numeric
    right: "Node" = None  # khusus numeric
    depth: int = 0

    def predict_row(self, row: dict):
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

    def predict_df(self, X: pd.DataFrame):
        rows = X.to_dict(orient="records")
        return [self.predict_row(r) for r in rows]

    def make_leaf(self):
        self.is_leaf = True
        self.feature = None
        self.feature_type = None
        self.threshold = None
        self.children = {}
        self.left = None
        self.right = None

    def to_rules(self, indent=""):
        if self.is_leaf:
            n = sum(self.class_counts.values())
            return f"{indent}THEN class = {self.prediction} (n={n})"

        if self.feature_type == "numeric":
            s = []
            s.append(f"{indent}IF {self.feature} <= {self.threshold:.6g}")
            s.append(self.left.to_rules(indent + "  "))
            s.append(f"{indent}ELSE  # {self.feature} > {self.threshold:.6g}")
            s.append(self.right.to_rules(indent + "  "))
            return "\n".join(s)

        s = [f"{indent}SPLIT {self.feature}"]
        for v, child in self.children.items():
            s.append(f"{indent}IF {self.feature} == {repr(v)}")
            s.append(child.to_rules(indent + "  "))
        s.append(f"{indent}ELSE")
        s.append(f"{indent}  THEN class = {self.prediction} (fallback)")
        return "\n".join(s)



# 3. PROSES: implementasi C4.5 (gain ratio + threshold numerik)

class C45Classifier:
    def __init__(self, max_depth=12, min_samples_split=10, min_gain_ratio=1e-6):
        self.max_depth = int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.min_gain_ratio = float(min_gain_ratio)

        self.root = None
        self.feature_types_ = {}

    @staticmethod
    def _majority_class(y):
        counts = Counter(y)
        return sorted(counts.items(), key=lambda kv: (-kv[1], str(kv[0])))[0][0]

    def _best_split_numeric(self, x: pd.Series, y: pd.Series):
        arr = x.to_numpy(dtype=float, copy=False)
        order = np.argsort(arr)
        arr_sorted = arr[order]
        y_sorted = y.to_numpy(copy=False)[order]

        if np.all(arr_sorted == arr_sorted[0]):
            return 0.0, None

        thresholds = []
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

        for t in thresholds:
            left_mask = arr <= t
            right_mask = ~left_mask
            if left_mask.sum() == 0 or right_mask.sum() == 0:
                continue
            gr = _gain_ratio(parent_y, [y[left_mask].tolist(), y[right_mask].tolist()])
            if gr > best_gr:
                best_gr, best_t = gr, float(t)

        return best_gr, best_t

    def _best_split_categorical(self, x: pd.Series, y: pd.Series):
        groups = []
        for _, yy in y.groupby(x):
            groups.append(yy.tolist())
        if len(groups) <= 1:
            return 0.0
        return _gain_ratio(y.tolist(), groups)

    def _best_split(self, X: pd.DataFrame, y: pd.Series, available_features):
        best = {"feature": None, "type": None, "threshold": None, "gain_ratio": 0.0}

        for feat in available_features:
            ftype = self.feature_types_[feat]
            if ftype == "numeric":
                gr, t = self._best_split_numeric(X[feat], y)
                if gr > best["gain_ratio"] and t is not None:
                    best = {"feature": feat, "type": "numeric", "threshold": t, "gain_ratio": gr}
            else:
                gr = self._best_split_categorical(X[feat], y)
                if gr > best["gain_ratio"]:
                    best = {"feature": feat, "type": "categorical", "threshold": None, "gain_ratio": gr}

        return best

    def _build(self, X: pd.DataFrame, y: pd.Series, depth: int, available_features):
        counts = dict(Counter(y))
        pred = self._majority_class(y)

        node = Node(
            is_leaf=False,
            prediction=pred,
            class_counts=counts,
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
        node.children = {}
        for val in X[feat].dropna().unique().tolist():
            idx = X[feat] == val
            node.children[val] = self._build(X.loc[idx].drop(columns=[feat]), y[idx], depth + 1, new_features)
        return node

    def fit(self, X: pd.DataFrame, y: pd.Series, feature_types: dict):
        self.feature_types_ = feature_types.copy()
        features = list(X.columns)
        self.root = self._build(X, y, depth=0, available_features=features)
        return self

    def predict(self, X: pd.DataFrame):
        if self.root is None:
            raise RuntimeError("Model belum dilatih. Jalankan fit() terlebih dulu.")
        return np.array(self.root.predict_df(X))

    def prune(self, X_val: pd.DataFrame, y_val: pd.Series):
        if self.root is None or len(y_val) == 0:
            return self

        def _accuracy(y_true, y_pred):
            y_true = np.asarray(y_true)
            y_pred = np.asarray(y_pred)
            return float((y_true == y_pred).mean()) if len(y_true) else 0.0

        def _prune(node: Node, Xv: pd.DataFrame, yv: pd.Series):
            if node.is_leaf or len(yv) == 0:
                return

            if node.feature_type == "numeric":
                left_mask = Xv[node.feature] <= node.threshold
                right_mask = ~left_mask
                _prune(node.left, Xv[left_mask], yv[left_mask])
                _prune(node.right, Xv[right_mask], yv[right_mask])
            else:
                for val, child in list(node.children.items()):
                    idx = Xv[node.feature] == val
                    _prune(child, Xv.loc[idx].drop(columns=[node.feature]), yv[idx])

            y_pred_sub = np.array(node.predict_df(Xv))
            acc_sub = _accuracy(yv, y_pred_sub)
            acc_leaf = _accuracy(yv, np.array([node.prediction] * len(yv)))

            if acc_leaf >= acc_sub:
                node.make_leaf()

        _prune(self.root, X_val, y_val)
        return self



# 4. PROSES: util preprocessing

def clean_and_prepare(df: pd.DataFrame, target_col: str, drop_cols: list):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    df = df.drop_duplicates()

    if target_col not in df.columns:
        raise ValueError(f"Kolom target '{target_col}' tidak ditemukan. Kolom yang ada: {list(df.columns)}")

    for c in drop_cols:
        if c in df.columns:
            df = df.drop(columns=[c])

    df = df.dropna(subset=[target_col])

    for col in df.columns:
        if col == target_col:
            continue
        if df[col].dtype == "object":
            s = df[col].astype(str).str.strip().str.replace(",", ".", regex=False)
            num = pd.to_numeric(s, errors="coerce")
            if num.notna().mean() >= 0.95:
                df[col] = num
            else:
                df[col] = df[col].astype(str).str.strip()

    for col in df.columns:
        if col == target_col:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna("Unknown").astype(str)

    df[target_col] = df[target_col].astype(str).str.strip()
    return df


def infer_feature_types(df: pd.DataFrame, target_col: str):
    types = {}
    for col in df.columns:
        if col == target_col:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            types[col] = "numeric"
        else:
            types[col] = "categorical"
    return types


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



# 5. PROSES: main pipeline (KDD)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="college_student_management_data.csv")
    parser.add_argument("--target", default="risk_level")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--val_size", type=float, default=0.25)
    parser.add_argument("--max_depth", type=int, default=12)
    parser.add_argument("--min_samples_split", type=int, default=10)
    parser.add_argument("--min_gain_ratio", type=float, default=1e-6)
    args = parser.parse_args()

    print("--- 1. Memuat Dataset ---")
    data_path = resolve_data_path(args.data)
    df_raw = pd.read_csv(data_path)
    print(f"Total data awal: {len(df_raw)} baris, {df_raw.shape[1]} kolom")

    print("\n--- 2. Data Selection (Memilih Fitur dan Target) ---")
    drop_cols = ["student_id"]
    df = clean_and_prepare(df_raw, target_col=args.target, drop_cols=drop_cols)

    print(f"Kolom setelah selection: {list(df.columns)}")
    print("Distribusi target:")
    print(df[args.target].value_counts())

    print("\n--- 3. Data Understanding (Info & Statistik Singkat) ---")
    print(df.info())
    print(df.describe(include="all").transpose().head(12))

    print("\n--- 4. Data Splitting (Train/Test) ---")
    X = df.drop(columns=[args.target])
    y = df[args.target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=42,
        stratify=y
    )
    print(f"Jumlah data latih: {len(X_train)} baris")
    print(f"Jumlah data uji  : {len(X_test)} baris")

    print("\n--- 5. Split internal Train/Validation (untuk pruning sederhana) ---")
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train,
        test_size=args.val_size,
        random_state=42,
        stratify=y_train
    )
    print(f"Train untuk bangun pohon: {len(X_tr)} baris")
    print(f"Validation untuk pruning: {len(X_val)} baris")

    print("\n--- 6. Transformation (Tipe Fitur) ---")
    feature_types = infer_feature_types(df, target_col=args.target)
    num_feats = [k for k, v in feature_types.items() if v == "numeric"]
    cat_feats = [k for k, v in feature_types.items() if v == "categorical"]
    print(f"Numeric     ({len(num_feats)}): {num_feats}")
    print(f"Categorical ({len(cat_feats)}): {cat_feats}")

    print("\n--- 7. Data Mining (Training C4.5) ---")
    model = C45Classifier(
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_gain_ratio=args.min_gain_ratio
    )
    model.fit(X_tr, y_tr, feature_types=feature_types)

    print("\n--- 8. Evaluation (Pruning + Confusion Matrix) ---")
    model.prune(X_val, y_val)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    pr, rc, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)

    print(f"Akurasi : {acc:.4f}")
    print(f"Presisi : {pr:.4f} (macro)")
    print(f"Recall  : {rc:.4f} (macro)")
    print(f"F1-score: {f1:.4f} (macro)")

    labels = sorted(y.unique().tolist())
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    print("\nConfusion Matrix (baris=aktual, kolom=prediksi):")
    cm_df = pd.DataFrame(cm, index=[f"aktual_{l}" for l in labels], columns=[f"pred_{l}" for l in labels])
    print(cm_df)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, labels=labels, zero_division=0))

    print("\n--- 9. Aturan Pohon (Ringkas) ---")
    print(model.root.to_rules())


if __name__ == "__main__":
    main()
