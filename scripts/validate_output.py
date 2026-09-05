"""
Validate v2 output against reference parquet.
Full row-by-row, column-by-column comparison.
Usage: python scripts/validate_output.py <reference.parquet> <v2_output.parquet>
"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

def compare_dfs(ref, v2, label="comparison"):
    print(f"\n{'='*70}")
    print(f"VALIDATION: {label}")
    print(f"{'='*70}")
    
    # Shape check
    if ref.shape != v2.shape:
        print(f"[FAIL] Shape mismatch: ref={ref.shape}, v2={v2.shape}")
    else:
        print(f"[OK] Shape matches: {ref.shape}")
    
    # Column check
    ref_cols = set(ref.columns)
    v2_cols = set(v2.columns)
    only_ref = ref_cols - v2_cols
    only_v2 = v2_cols - ref_cols
    if only_ref or only_v2:
        print(f"[FAIL] Column mismatch. Only in ref: {only_ref}, Only in v2: {only_v2}")
    else:
        print(f"[OK] All {len(ref.columns)} columns match")
    
    # Row count
    if len(ref) != len(v2):
        print(f"[FAIL] Row count: ref={len(ref)}, v2={len(v2)}")
        return False
    
    if len(ref) == 0:
        print("[SKIP] Empty DataFrames")
        return True
    
    # Column-by-column comparison
    mismatches = {}
    for col in ref.columns:
        if col not in v2.columns:
            mismatches[col] = "missing in v2"
            continue
        
        ref_vals = ref[col].values
        v2_vals = v2[col].values
        
        # Handle NaN comparison
        if ref_vals.dtype.kind in ('f', 'c'):
            match = np.isclose(ref_vals, v2_vals, equal_nan=True, rtol=1e-10, atol=1e-12)
        elif ref_vals.dtype.kind in ('O', 'U', 'S'):
            match = (ref_vals == v2_vals) | (pd.isna(ref_vals) & pd.isna(v2_vals))
        else:
            match = (ref_vals == v2_vals) | (pd.isna(ref_vals) & pd.isna(v2_vals))
        
        mismatch_count = int((~match).sum())
        if mismatch_count > 0:
            # Find first few mismatching indices
            bad_idx = np.where(~match)[0][:5]
            examples = [(int(i), str(ref_vals[i]), str(v2_vals[i])) for i in bad_idx]
            mismatches[col] = {
                "count": mismatch_count,
                "pct": mismatch_count / len(ref) * 100,
                "examples": examples,
            }
    
    if not mismatches:
        print(f"[PASS] All {len(ref)} rows × {len(ref.columns)} columns match exactly!")
        return True
    
    print(f"\n[FAIL] {len(mismatches)} column(s) with mismatches:")
    for col, info in mismatches.items():
        print(f"  {col}: {info['count']} mismatches ({info['pct']:.2f}%)")
        if info['examples']:
            print(f"    First mismatches (idx, ref, v2):")
            for idx, rv, vv in info['examples']:
                print(f"      row {idx}: ref={rv}, v2={vv}")
    
    return False


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python scripts/validate_output.py <reference.parquet> <v2_output.parquet>")
        sys.exit(1)
    
    ref_path = Path(sys.argv[1])
    v2_path = Path(sys.argv[2])
    
    if not ref_path.exists():
        print(f"Reference not found: {ref_path}")
        sys.exit(1)
    if not v2_path.exists():
        print(f"V2 output not found: {v2_path}")
        sys.exit(1)
    
    print(f"Loading reference: {ref_path}")
    ref = pd.read_parquet(ref_path)
    print(f"Loading v2 output: {v2_path}")
    v2 = pd.read_parquet(v2_path)
    
    success = compare_dfs(ref, v2, label=f"{ref_path.name} vs {v2_path.name}")
    sys.exit(0 if success else 1)
