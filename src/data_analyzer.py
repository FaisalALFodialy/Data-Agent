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
            if s.empty: 
                rows.append((col, 0, 0.0, []))
                continue
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
            
            #Apply mask to cleaned data (s) instead of original data with NaN
            outliers = s[(s < lower) | (s > upper)]
            cnt = len(outliers)
            pct = round(cnt / max(len(s), 1) * 100, 4)
            
            # Store outlier values for verification (limit to first 100 for performance)
            outlier_values = outliers.head(100).tolist() if cnt > 0 else []
            
            rows.append((col, cnt, pct, outlier_values))
        return pd.DataFrame(rows, columns=["column","outliers_iqr","outliers_iqr_pct", "outlier_values"])

    def detect_outliers_zscore(self, threshold: float = 3.0) -> pd.DataFrame:
        num_df = self.df.select_dtypes(include=np.number)
        rows = []
        for col in num_df.columns:
            s = num_df[col].dropna()
            if s.empty: 
                rows.append((col, 0, 0.0, []))
                continue
            mean, std = s.mean(), s.std(ddof=0)
            if std == 0 or np.isnan(std):
                rows.append((col, 0, 0.0, []))
                continue
                
            # Apply Z-score calculation to cleaned data (s) instead of original data with NaN
            z_scores = (s - mean) / std
            outliers = s[z_scores.abs() >= threshold]
            cnt = len(outliers)
            pct = round(cnt / max(len(s), 1) * 100, 4)
            
            # Store outlier values for verification (limit to first 100 for performance)
            outlier_values = outliers.head(100).tolist() if cnt > 0 else []
            
            rows.append((col, cnt, pct, outlier_values))
        return pd.DataFrame(rows, columns=["column","outliers_z","outliers_z_pct", "outlier_values"])

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

    def calculate_standard_deviation(self) -> dict:
        numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) == 0:
            return {"per_column": {}}
        std_devs = self.df[numeric_cols].std().to_dict()
        return {"per_column": std_devs}

    def verify_outlier_detection(self, column: str, expected_outlier_value: float = None) -> dict:
        """
        Verify outlier detection results for a specific column.
        
        This function provides detailed verification of outlier detection
        to ensure the algorithm is working correctly.
        
        Args:
            column: Column name to verify
            expected_outlier_value: Optional specific value expected to be an outlier
            
        Returns:
            Dict containing verification results and statistics
        """
        if column not in self.df.columns:
            return {"error": f"Column '{column}' not found in dataset"}
        
        if not pd.api.types.is_numeric_dtype(self.df[column]):
            return {"error": f"Column '{column}' is not numeric"}
        
        # Get clean data (no NaN values)
        clean_data = self.df[column].dropna()
        if clean_data.empty:
            return {"error": f"Column '{column}' has no non-null values"}
        
        # Manual IQR calculation for verification
        q1 = clean_data.quantile(0.25)
        q3 = clean_data.quantile(0.75)
        iqr = q3 - q1
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr
        
        # Find outliers manually
        manual_outliers = clean_data[(clean_data < lower_fence) | (clean_data > upper_fence)]
        manual_count = len(manual_outliers)
        
        # Get results from our detection algorithm
        outlier_results = self.detect_outliers_iqr()
        algo_result = outlier_results[outlier_results['column'] == column]
        
        if algo_result.empty:
            return {"error": f"No outlier detection results found for column '{column}'"}
        
        algo_count = int(algo_result.iloc[0]['outliers_iqr'])
        algo_percentage = float(algo_result.iloc[0]['outliers_iqr_pct'])
        algo_values = algo_result.iloc[0]['outlier_values']
        
        # Verification results
        verification = {
            "column": column,
            "total_non_null_values": len(clean_data),
            "statistics": {
                "q1": float(q1),
                "q3": float(q3),
                "iqr": float(iqr),
                "lower_fence": float(lower_fence),
                "upper_fence": float(upper_fence)
            },
            "manual_calculation": {
                "outlier_count": manual_count,
                "outlier_percentage": round(manual_count / len(clean_data) * 100, 4),
                "sample_outliers": manual_outliers.head(10).tolist()
            },
            "algorithm_results": {
                "outlier_count": algo_count,
                "outlier_percentage": algo_percentage,
                "sample_outliers": algo_values[:10] if algo_values else []
            },
            "verification": {
                "counts_match": manual_count == algo_count,
                "percentage_match": abs(round(manual_count / len(clean_data) * 100, 4) - algo_percentage) < 0.01
            }
        }
        
        # Check specific value if provided
        if expected_outlier_value is not None:
            is_outlier_manual = (expected_outlier_value < lower_fence) or (expected_outlier_value > upper_fence)
            is_in_results = expected_outlier_value in manual_outliers.values
            verification["specific_value_check"] = {
                "value": expected_outlier_value,
                "should_be_outlier": is_outlier_manual,
                "found_in_results": is_in_results,
                "verification_passed": is_outlier_manual == is_in_results
            }
        
        return verification



    
    def create_analysis_report(self):
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
            "usability_score": self.calculate_usability_score(),
            "usability": self.calculate_usability_score(),
            "standard_deviation": self.calculate_standard_deviation()
        }


def create_analysis_report(df: pd.DataFrame) -> dict:
    return DataAnalyzer(df).create_analysis_report()

def calculate_usability_score(analysis_report: dict) -> dict:
    """Calculate usability score from analysis report."""
    try:
        if "usability" in analysis_report:
            usability_data = analysis_report["usability"]
            score = usability_data.get("usability_score", 0)
            components = usability_data.get("components", {})
            
            return {
                "overall_score": int(score),
                "components": components,
                "grade": "A" if score >= 90 else "B" if score >= 80 else "C" if score >= 70 else "D" if score >= 60 else "F",
                "grade_description": "Excellent" if score >= 90 else "Good" if score >= 80 else "Fair" if score >= 70 else "Poor" if score >= 60 else "Failing"
            }
        else:
            return {"overall_score": 0, "components": {}, "grade": "F", "grade_description": "No usability data available"}
    except Exception as e:
        return {"overall_score": 0, "components": {}, "grade": "F", "grade_description": f"Error: {str(e)}"}
