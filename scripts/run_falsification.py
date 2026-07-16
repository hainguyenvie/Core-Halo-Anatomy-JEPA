#!/usr/bin/env python3
import sys

from core_halo_jepa.cli import main

if __name__ == "__main__":
    sys.argv.insert(1, "run-falsification")
    main()
