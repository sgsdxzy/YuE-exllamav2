import os
import sys

from common import parser

if __name__ == "__main__":
    parser.parse_args() # make --help work

    dirname = os.path.dirname(os.path.abspath(__file__))
    os.system(f"python {os.path.join(dirname, "infer_stage1.py")} {" ".join(sys.argv[1:])}")
    os.system(f"python {os.path.join(dirname, "infer_stage2.py")} {" ".join(sys.argv[1:])}")
    os.system(f"python {os.path.join(dirname, "infer_postprocess.py")} {" ".join(sys.argv[1:])}")