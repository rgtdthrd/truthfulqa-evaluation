import sys
from pathlib import Path

# Ensure project root is on path so "utils" can be imported when run from analysis/
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from utils import generate_responses_for_sample, build_scoring_csv


def main() -> None:
    # Use the default OpenRouter model configured in config.OPENROUTER_MODEL
    output_path: Path = generate_responses_for_sample()
    print(f"Saved model responses to {output_path}")
    output_path: Path = build_scoring_csv()
    print(f"Scoring CSV written to {output_path}")


if __name__ == "__main__":
    main()

