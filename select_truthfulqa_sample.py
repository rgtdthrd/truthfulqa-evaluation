from utils import create_truthfulqa_sample, SAMPLE_PATH


def main() -> None:
    sampled_ds, counts = create_truthfulqa_sample()

    print(f"Saved stratified sample of {len(sampled_ds)} questions to {SAMPLE_PATH}")
    print("Questions per category in the sample:")
    for cat, count in sorted(counts.items(), key=lambda x: x[0]):
        print(f"  {cat}: {count}")
    print(f"Number of categories covered: {sum(1 for c in counts.values() if c > 0)}")


if __name__ == "__main__":
    main()

