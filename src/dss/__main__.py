"""Allow ``python -m dss``."""

import sys

from dss.cli import main

if __name__ == "__main__":
    sys.exit(main())
