# ts_benchmark/__main__.py

import sys
from ts_benchmark.cli import build_parser, run

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        sys.exit(run(args))
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if getattr(args, "verbose", False):
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
