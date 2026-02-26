import sys
from pathlib import Path

# Ensure project root is on path so "utils" can be imported when run from analysis/
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
from utils import run_manual_semantic_labeling
from utils import SCORING_DIR, DATA_DIR, RESPONSES_DIR

def main() -> None:
    run_manual_semantic_labeling(
        dataset_path=DATA_DIR / "truthfulqa_sample_50.jsonl",
        responses_path=RESPONSES_DIR / "truthfulqa_sample_50_stepfun_step_3.5_flash_free.jsonl",
        output_path=SCORING_DIR / "truthfulqa_sample_50_manual_label.csv",
    )

if __name__ == "__main__":
    main()
    print("Manual labeling complete")
    print(f"Manual labeling output written to {SCORING_DIR / 'truthfulqa_sample_50_manual_label.csv'}")