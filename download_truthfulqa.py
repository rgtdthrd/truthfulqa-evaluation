from utils import download_truthfulqa_generation


def main() -> None:
    """
    Download the TruthfulQA generation validation split and save it under ./data.
    """
    download_truthfulqa_generation(split="validation")


if __name__ == "__main__":
    main()
