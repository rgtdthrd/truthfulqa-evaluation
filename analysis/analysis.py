import sys
from pathlib import Path

# Ensure project root is on path so "utils" can be imported when run from analysis/
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
from utils import analyze_truthfulqa_results
from utils import SCORING_DIR, DATA_DIR

def main() -> None:
    analyze_truthfulqa_results(
        scoring_path=SCORING_DIR / "truthfulqa_sample_50_manual_label.csv",
        dataset_path=DATA_DIR / "truthfulqa_sample_50.jsonl",
    )

if __name__ == "__main__":
    main()
    print("Analysis complete")