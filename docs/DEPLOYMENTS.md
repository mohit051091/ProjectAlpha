# Deployment Registry

This registry tracks the offline and online releases of the pre-move detection system.

## Version 1.0.0 (2026-06-12)
- **Deployment Date**: 2026-06-12
- **Type**: Offline Validation Release
- **Changes**:
  - Implemented 7-fold walk-forward purged splits validation engine.
  - Migrated classifier baseline to LightGBM.
  - Implemented Daily Open Trend Filter.
- **Risk**: Low (Offline Validation)
- **Rollback Plan**: Restore legacy `50_baseline_model.py` script.
- **Result**: Successful. Achieved +81.81% cumulative P&L under daily open trend filter using 0.25 probability threshold.
