import os
import sys

from common import parser


def check_exit(status: int):
    if status != 0:
        sys.exit(status)


if __name__ == "__main__":
    parser.parse_args()  # make --help work

    dirname = os.path.dirname(os.path.abspath(__file__))
    check_exit(os.system(f"python {os.path.join(dirname, "infer_stage1.py")} {" ".join(sys.argv[1:])}"))
    check_exit(os.system(f"python {os.path.join(dirname, "infer_stage2.py")} {" ".join(sys.argv[1:])}"))
    check_exit(os.system(f"python {os.path.join(dirname, "infer_postprocess.py")} {" ".join(sys.argv[1:])}"))
