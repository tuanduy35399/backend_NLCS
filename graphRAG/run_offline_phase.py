import argparse
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from indexing.build_index import BuildIndex


def parse_args():
    parser = argparse.ArgumentParser(description="Build GraphRAG offline index")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Tiep tuc tu checkpoint da luu gan nhat",
    )
    parser.add_argument(
        "--resume-from",
        type=int,
        metavar="N",
        help="Bo qua N node graph da luu (vi du: --resume-from 60)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("Bat dau chay")
    builder = BuildIndex()
    builder.run(resume=args.resume, resume_from=args.resume_from)
