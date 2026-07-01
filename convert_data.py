"""Convert data/raw/data.xlsx to data/raw/log_entries.csv."""

from pathlib import Path
import pandas as pd

XLSX_PATH = Path("data/raw/data.xlsx")
CSV_PATH = Path("data/raw/log_entries.csv")


def convert(xlsx_path: Path = XLSX_PATH, csv_path: Path = CSV_PATH) -> Path:
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Source file not found: {xlsx_path}")

    df = pd.read_excel(xlsx_path, engine="openpyxl")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"Converted {xlsx_path} -> {csv_path}  ({len(df)} rows)")
    return csv_path


if __name__ == "__main__":
    convert()
