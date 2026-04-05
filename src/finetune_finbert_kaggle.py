from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if sys.path and Path(sys.path[0]).resolve() == SCRIPT_DIR:
    sys.path.pop(0)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.finetune_finbert_kaggle import *  # type: ignore[F403]


if __name__ == "__main__":
    from src.models.finetune_finbert_kaggle import main

    main()
