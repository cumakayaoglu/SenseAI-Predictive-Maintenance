import zipfile
import shutil
from pathlib import Path

import pandas as pd
import numpy as np

# ===========================
# KLASÖRLER
# ===========================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "cmapss"

ZIP_PATH = DATA_DIR / "CMAPSSData.zip"

print("Current Working Directory :", Path.cwd())
print("Script Location           :", BASE_DIR)
print("ZIP PATH                  :", ZIP_PATH)
print("ZIP Exists                :", ZIP_PATH.exists())


# ===========================
# ZIP ÇIKAR
# ===========================

def extract_data():

    if not ZIP_PATH.exists():
        raise FileNotFoundError(
            f"\nCMAPSSData.zip bulunamadı.\n"
            f"Beklenen konum:\n{ZIP_PATH}"
        )

    print("\nZip dosyası kontrol ediliyor...")

    if not zipfile.is_zipfile(ZIP_PATH):
        raise RuntimeError(
            "CMAPSSData.zip bozuk görünüyor.\n"
            "Dosyayı tekrar indir."
        )

    print("Zip açılıyor...")

    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(DATA_DIR)

    nested = DATA_DIR / "CMAPSSData"

    if nested.exists():

        for file in nested.iterdir():
            shutil.move(str(file), str(DATA_DIR / file.name))

        shutil.rmtree(nested)

    print("Zip başarıyla çıkarıldı.")


# ===========================
# VERİYİ HAZIRLA
# ===========================

def prepare_data():

    print("\nVeri hazırlanıyor...")

    columns = [
        "unit_number",
        "time_cycles",
        "setting_1",
        "setting_2",
        "setting_3",
    ]

    columns += [f"sensor_{i}" for i in range(1, 22)]

    train_file = DATA_DIR / "train_FD001.txt"
    test_file = DATA_DIR / "test_FD001.txt"

    if not train_file.exists():
        raise FileNotFoundError(train_file)

    train_df = pd.read_csv(
        train_file,
        sep=r"\s+",
        engine="python",
        header=None,
        names=columns,
    )

    test_df = pd.read_csv(
        test_file,
        sep=r"\s+",
        engine="python",
        header=None,
        names=columns,
    )

    rul = (
        train_df.groupby("unit_number")["time_cycles"]
        .max()
        .reset_index()
    )

    rul.columns = ["unit_number", "max_cycles"]

    train_df = train_df.merge(rul, on="unit_number")

    train_df["RUL"] = (
        train_df["max_cycles"] - train_df["time_cycles"]
    )

    train_df.drop(columns=["max_cycles"], inplace=True)

    train_df["label"] = np.where(train_df["RUL"] <= 30, 1, 0)

    train_df.to_csv(DATA_DIR / "processed_train.csv", index=False)
    test_df.to_csv(DATA_DIR / "processed_test.csv", index=False)

    print("\nHazırlandı.")
    print("Toplam satır :", len(train_df))
    print("Riskli satır :", (train_df["label"] == 1).sum())


# ===========================
# MAIN
# ===========================

if __name__ == "__main__":
    extract_data()
    prepare_data()