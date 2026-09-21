"""Enable ``python -m opticore`` as an alias for the CLI."""

from __future__ import annotations

import sys

from opticore.cli.main import main

if __name__ == "__main__":
    sys.exit(main())
