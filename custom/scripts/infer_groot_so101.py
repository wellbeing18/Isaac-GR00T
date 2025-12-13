#!/usr/bin/env python3
"""
Compatibility shim for legacy scripts.

Historically this repo shipped `custom/scripts/infer_groot_so101.py` as the
single-threaded SO101 inference entrypoint. Several diagnostic utilities and
wrapper scripts still import `infer_groot_so101` for helpers like:
  - is_lora_checkpoint
  - load_groot_with_lora
  - SO101Robot / Gr00tLocalInference

Those implementations were later moved to `deprecated_infer_groot_so101.py`
and the preferred runtime path became `infer_groot_async.py`.

To avoid breaking existing workflows, this file re-exports the legacy
implementation and preserves the original CLI behavior.
"""

# Re-export legacy API surface for downstream scripts.
from deprecated_infer_groot_so101 import *  # noqa: F401,F403


def main() -> int:
    # Keep a stable entrypoint for `python custom/scripts/infer_groot_so101.py ...`
    from deprecated_infer_groot_so101 import main as _main

    return _main()


if __name__ == "__main__":
    raise SystemExit(main())


