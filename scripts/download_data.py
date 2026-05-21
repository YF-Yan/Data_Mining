"""
下载 UCI Online Retail 数据集到 data/online_retail.csv

用法：
    python scripts/download_data.py
"""

import urllib.request
from pathlib import Path

# UCI 提供的在线零售数据直链（若失效请手动从官网下载）
DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_CSV = DATA_DIR / "online_retail.csv"


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("正在下载 Online Retail 数据（xlsx）...")
    xlsx_path = DATA_DIR / "Online Retail.xlsx"
    urllib.request.urlretrieve(DATA_URL, xlsx_path)

    print("正在转换为 CSV...")
    import pandas as pd

    df = pd.read_excel(xlsx_path)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"已保存至: {OUTPUT_CSV}")
    print(f"共 {len(df):,} 行记录。")


if __name__ == "__main__":
    main()
