
import io, base64, csv
import pandas as pd

#CSV loader 
def load_csv_file(uploaded_file) -> pd.DataFrame | None:
    """
    Robust CSV reader:
    - Tries multiple encodings (UTF-8, UTF-8-SIG, Windows-1256, CP1252, Latin-1)
    - Sniffs delimiter (comma/semicolon/tab/pipe) using csv.Sniffer
    - Falls back to pandas inference (sep=None, engine='python')
    - Final fallback: try reading as Excel if it's actually an Excel file
    """
    if uploaded_file is None:
        return None

    # read bytes once
    uploaded_file.seek(0)
    content = uploaded_file.read()
    if not content:
        return None

    # encodings to try (Arabic-friendly first)
    encodings = ["utf-8", "utf-8-sig", "windows-1256", "cp1256", "cp1252", "latin-1"]

    # helper: try with a specific encoding and optional delimiter
    def try_read(encoding: str, sep=None):
        buf = io.BytesIO(content)
        try:
            if sep is None:
                # let pandas infer sep using the python engine (more tolerant)
                return pd.read_csv(buf, encoding=encoding, sep=None, engine="python")
            else:
                return pd.read_csv(buf, encoding=encoding, sep=sep)
        except Exception:
            return None

    # 1) Try to sniff delimiter with csv.Sniffer on a text sample
    for enc in encodings:
        try:
            sample_text = content[:50000].decode(enc, errors="strict")
        except Exception:
            continue

        # Normalize line endings
        sample_text = sample_text.replace("\r\n", "\n").replace("\r", "\n")
        try:
            dialect = csv.Sniffer().sniff(sample_text, delimiters=[",", ";", "\t", "|"])
            sep = dialect.delimiter
            df = try_read(enc, sep=sep)
            if df is not None and df.shape[1] > 0:
                return df
        except Exception:
            # Sniffer failed; we'll try inference paths below
            pass

        # 2) Let pandas infer separator
        df = try_read(enc, sep=None)
        if df is not None and df.shape[1] > 0:
            return df

        # 3) Try common explicit separators as a fallback
        for sep in [",", ";", "\t", "|"]:
            df = try_read(enc, sep=sep)
            if df is not None and df.shape[1] > 0:
                return df

    # 4) Final fallback: maybe it's actually an Excel file with .csv extension
    try:
        buf = io.BytesIO(content)
        df_xls = pd.read_excel(buf)
        if isinstance(df_xls, pd.DataFrame) and df_xls.shape[1] > 0:
            return df_xls
    except Exception:
        pass

    return None

def save_csv_file(df: pd.DataFrame, path: str, include_index: bool = False, encoding: str = "utf-8"):
    df.to_csv(path, index=include_index, encoding=encoding)

def validate_dataframe(df: pd.DataFrame) -> bool:
    return isinstance(df, pd.DataFrame) and df.shape[1] > 0

def create_download_link(df: pd.DataFrame, filename: str) -> str:
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    b64 = base64.b64encode(csv_bytes).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">⬇️ Download {filename}</a>'

def generate_cleaning_report(original_df: pd.DataFrame, cleaned_df: pd.DataFrame, pipeline: list) -> dict:
    return {
        "original_shape": getattr(original_df, "shape", None),
        "cleaned_shape": getattr(cleaned_df, "shape", None),
        "operations": pipeline,
        "rows_removed": (original_df.shape[0] - cleaned_df.shape[0]) if original_df is not None else None
    }

def format_bytes(num_bytes: int) -> str:
    for unit in ['B','KB','MB','GB','TB']:
        if num_bytes < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} PB"

def calculate_memory_usage(df: pd.DataFrame) -> dict:
    bytes_total = int(df.memory_usage(deep=True).sum())
    return {"bytes": bytes_total, "total": format_bytes(bytes_total)}
