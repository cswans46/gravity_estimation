from pathlib import Path
import pyreadstat
import pandas as pd

path = Path(
    r"D:\Research\Esmaeil Research\cai_2023"
    r"\rta_individual_agreements_20260111.dta"
)

reader = pyreadstat.read_file_in_chunks(
    pyreadstat.read_dta,
    str(path),
    chunksize=500_000,
    usecols=["exporter", "importer", "year", "rta"],
)

total_rows = 0
duplicate_rows = 0
seen_pairs = set()

for chunk, _ in reader:
    total_rows += len(chunk)

    duplicates = chunk.duplicated(
        subset=["exporter", "importer", "year"],
        keep=False
    )
    duplicate_rows += int(duplicates.sum())

    # Check duplicates across chunk boundaries as well.
    keys = pd.MultiIndex.from_frame(
        chunk[["exporter", "importer", "year"]]
    )
    duplicate_rows += sum(key in seen_pairs for key in keys)
    seen_pairs.update(keys)

print(f"Rows processed: {total_rows:,}")
print(f"Duplicate rows detected: {duplicate_rows:,}")