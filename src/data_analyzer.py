# src/data_analyzer.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass
class DataAnalyzer:
    df: pd.DataFrame

    def calculate_basic_stats(self) -> dict:
        n_rows, n_cols = self.df.shape
        mem_mb = self.df.memory_usage(deep=True).sum() / (1024 ** 2)
        return {
            "n_rows": int(n_rows),
            "n_cols": int(n_cols),
            "memory_mb": round(mem_mb, 3),
            "column_names": list(self.df.columns),
            "index_name": self.df.index.name,
        }

    def analyze_missing_values(self) -> dict:
        n_rows, n_cols = self.df.shape
        per_col = self.df.isna().sum().rename("null_count").to_frame()
        per_col["null_pct"] = (per_col["null_count"] / max(n_rows, 1) * 100).round(2)
        per_col["non_null_count"] = n_rows - per_col["null_count"]
        per_col["dtype"] = self.df.dtypes.astype(str)
        per_col = per_col[["dtype","null_count","null_pct","non_null_count"]].sort_values(
            by=["null_count","null_pct"], ascending=[False,False]
        )
        total_cells = n_rows * n_cols
        total_nulls = int(per_col["null_count"].sum())
        rows_with_any_null = int(self.df.isna().any(axis=1).sum())
        return {
            "total_cells": int(total_cells),
            "total_missing": int(total_nulls),
            "missing_pct": round((total_nulls / total_cells * 100), 4) if total_cells else 0.0,
            "rows_with_any_null": rows_with_any_null,
            "rows_with_any_null_pct": round((rows_with_any_null / max(n_rows,1) * 100), 4) if n_rows else 0.0,
            "per_column": per_col.reset_index(names="column"),
        }

    def detect_data_types(self) -> pd.DataFrame:
        df = self.df.copy()
        types = df.dtypes.astype(str).rename("pandas_dtype").to_frame()
        inferred = {}
        for col in df.columns:
            s = df[col]
            if s.dtype == "object":
                try:
                    pd.to_numeric(s, errors="raise")
                    inferred[col] = "numeric_possible"; continue
                except Exception:
                    pass
                try:
                    pd.to_datetime(s, errors="raise", infer_datetime_format=True)
                    inferred[col] = "datetime_possible"
                except Exception:
                    inferred[col] = "object"
            else:
                inferred[col] = str(s.dtype)
        types["inferred"] = pd.Series(inferred)
        types["is_datetime"] = self._is_datetime_cols(df).reindex(df.columns, fill_value=False)
        types["is_numeric"] = df.dtypes.apply(lambda dt: np.issubdtype(dt, np.number))
        return types.reset_index(names="column")

    def _is_datetime_cols(self, df: pd.DataFrame) -> pd.Series:
        is_dt = {}
        for c in df.columns:
            s = df[c]
            if np.issubdtype(s.dtype, np.datetime64):
                is_dt[c] = True
            elif s.dtype == "object":
                try:
                    pd.to_datetime(s.dropna().sample(min(25, len(s.dropna()))), errors="raise")
                    is_dt[c] = True
                except Exception:
                    is_dt[c] = False
            else:
                is_dt[c] = False
        return pd.Series(is_dt)

    def calculate_cardinality(self) -> pd.DataFrame:
        nunique = self.df.nunique(dropna=True)
        return (
            pd.DataFrame({
                "column": nunique.index,
                "unique_values": nunique.values,
                "unique_ratio": (nunique / max(len(self.df),1)).round(4)
            }).sort_values("unique_values", ascending=False).reset_index(drop=True)
        )

    def detect_outliers_iqr(self) -> pd.DataFrame:
        num_df = self.df.select_dtypes(include=np.number)
        rows = []
        for col in num_df.columns:
            s = num_df[col].dropna()
            if s.empty: rows.append((col, 0, 0.0)); continue
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
            mask = (num_df[col] < lower) | (num_df[col] > upper)
            cnt = int(mask.sum()); pct = round(cnt / max(len(s),1) * 100, 4)
            rows.append((col, cnt, pct))
        return pd.DataFrame(rows, columns=["column","outliers_iqr","outliers_iqr_pct"])

    def detect_outliers_zscore(self, threshold: float = 3.0) -> pd.DataFrame:
        num_df = self.df.select_dtypes(include=np.number)
        rows = []
        for col in num_df.columns:
            s = num_df[col].dropna()
            if s.empty: rows.append((col, 0, 0.0)); continue
            mean, std = s.mean(), s.std(ddof=0)
            if std == 0 or np.isnan(std):
                rows.append((col, 0, 0.0)); continue
            z = (num_df[col] - mean) / std
            mask = z.abs() >= threshold
            cnt = int(mask.sum()); pct = round(cnt / max(len(s),1) * 100, 4)
            rows.append((col, cnt, pct))
        return pd.DataFrame(rows, columns=["column","outliers_z","outliers_z_pct"])

    def calculate_skewness_kurtosis(self) -> pd.DataFrame:
        num_df = self.df.select_dtypes(include=np.number)
        if num_df.empty:
            return pd.DataFrame(columns=["column","skewness","kurtosis"])
        skew = num_df.skew(numeric_only=True); kurt = num_df.kurtosis(numeric_only=True)
        return pd.DataFrame({
            "column": num_df.columns,
            "skewness": [round(skew.get(c, np.nan), 6) for c in num_df.columns],
            "kurtosis": [round(kurt.get(c, np.nan), 6) for c in num_df.columns],
        })

    def detect_duplicates(self) -> dict:
        dup_mask = self.df.duplicated(keep=False)
        dup_rows = self.df[dup_mask]
        n_dup_groups = int(self.df.duplicated().sum())
        return {
            "n_duplicate_rows": int(dup_rows.shape[0]),
            "n_duplicate_groups": n_dup_groups,
            "duplicate_rows": dup_rows,
        }

    def calculate_correlation_matrix(self) -> pd.DataFrame:
        num_df = self.df.select_dtypes(include=np.number)
        if num_df.shape[1] == 0:
            return pd.DataFrame()
        return num_df.corr(numeric_only=True).round(4)

    # in src/data_analyzer.py inside DataAnalyzer.generate_column_statistics
    def generate_column_statistics(self) -> pd.DataFrame:
        rows = []
        n = len(self.df)

        for col in self.df.columns:
            s = self.df[col]
            dtype = str(s.dtype)
            nulls = int(s.isna().sum())
            null_pct = round(nulls / max(n, 1) * 100, 4) if n else 0.0
            unique = int(s.nunique(dropna=True))

            # make example values a STRING (not list) to keep Arrow happy
            ex_vals = s.dropna().head(3).tolist()
            example_str = ", ".join(map(str, ex_vals)) if ex_vals else ""

            row = {
                "column": col,
                "dtype": dtype,
                "nulls": nulls,
                "null_pct": null_pct,
                "unique": unique,
                "min": None,
                "max": None,
                "mean": None,
                "std": None,
                "example_values": example_str,   # <- string, not list
            }

            if np.issubdtype(s.dtype, np.number):
                row["min"]  = float(np.nanmin(s)) if s.notna().any() else None
                row["max"]  = float(np.nanmax(s)) if s.notna().any() else None
                row["mean"] = float(np.nanmean(s)) if s.notna().any() else None
                row["std"]  = float(np.nanstd(s)) if s.notna().any() else None

            elif np.issubdtype(s.dtype, np.datetime64):
                # Convert datetimes to ISO strings so Arrow doesn’t choke on mixed None/ts
                row["min"] = s.min().isoformat() if s.notna().any() else None
                row["max"] = s.max().isoformat() if s.notna().any() else None

            rows.append(row)

        return pd.DataFrame(rows)


    def calculate_usability_score(self) -> dict:
        n_rows = len(self.df)
        if n_rows == 0:
            return {"usability_score": 0, "components": {}}
        missing = self.analyze_missing_values()
        dup = self.detect_duplicates()
        out_iqr = self.detect_outliers_iqr()
        miss_pct = missing["missing_pct"]
        dup_pct = round(dup["n_duplicate_rows"] / n_rows * 100, 4) if n_rows else 0.0
        out_pct = float(out_iqr["outliers_iqr_pct"].mean()) if not out_iqr.empty else 0.0
        score = 100.0
        score -= min(miss_pct * 1.2, 40)
        score -= min(dup_pct * 0.6, 30)
        score -= min(out_pct * 0.4, 20)
        score = int(max(0, round(score)))
        return {"usability_score": score,
                "components": {"missing_pct": miss_pct, "duplicate_pct": dup_pct, "avg_outlier_pct": out_pct}}

    def create_analysis_report(self) -> dict:
        return {
            "basic_stats": self.calculate_basic_stats(),
            "missing_values": self.analyze_missing_values(),
            "data_types": self.detect_data_types(),
            "cardinality": self.calculate_cardinality(),
            "outliers_iqr": self.detect_outliers_iqr(),
            "outliers_zscore": self.detect_outliers_zscore(),
            "skewness_kurtosis": self.calculate_skewness_kurtosis(),
            "duplicates": self.detect_duplicates(),
            "correlation_matrix": self.calculate_correlation_matrix(),
            "column_statistics": self.generate_column_statistics(),
            "usability": self.calculate_usability_score(),
        }

def create_analysis_report(df: pd.DataFrame) -> dict:
    return DataAnalyzer(df).create_analysis_report()

def calculate_usability_score(analysis_report: dict) -> dict:
    score = analysis_report["usability"]["usability_score"]
    return {"overall_score": int(score)}
