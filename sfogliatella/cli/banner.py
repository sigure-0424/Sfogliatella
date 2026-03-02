"""Startup ASCII-art banner for Sfogliatella."""

import sys

BANNER = r"""
 ____    __                    _  _          _         _  _
/ ___|  / _|  ___    __ _  | |(_)  __ _  | |_  ___ | || |  __ _
\___ \ | |_  / _ \  / _` | | || | / _` | | __|/ _ \| || | / _` |
 ___) ||  _|| (_) || (_| | | || || (_| | | |_|  __/| || || (_| |
|____/ |_|   \___/  \__, | |_||_| \__,_|  \__|\___||_||_| \__,_|
                     |___/
"""

def print_banner(version: str = "0.1.0", no_banner: bool = False) -> None:
    if no_banner:
        return
    print(BANNER, file=sys.stderr)
    print(f"  Sfogliatella v{version} — General-Purpose Time-Series ML Library", file=sys.stderr)
    print("  JAX backend · Column-index-independent · No PyTorch/TF", file=sys.stderr)
    print("", file=sys.stderr)
