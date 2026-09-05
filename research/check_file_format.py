from pathlib import Path
import sys

data_path = Path("d:/ML_June_2026/Data")

for f in sorted(data_path.glob("*.parquet")):
    with open(f, 'rb') as file:
        header = file.read(100)
    print(f"\n{f.name}:")
    print(f"  First 100 bytes (hex): {header.hex()}")
    print(f"  First 100 bytes (repr): {repr(header[:50])}")
    print(f"  File size: {f.stat().st_size} bytes")
