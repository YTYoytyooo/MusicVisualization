"""Standalone entry; all project data lives under data/v3.0."""
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
if __name__=='__main__':
    # The full editor is the default; frozen research tools remain explicitly accessible.
    if len(sys.argv) > 1 and sys.argv[1] == 'research':
        sys.argv.pop(1)
        from tools.train_next_v2.v3_studio.cli import main
    else:
        from tools.train_next_v2.v3_workbench.main import main
    raise SystemExit(main())
