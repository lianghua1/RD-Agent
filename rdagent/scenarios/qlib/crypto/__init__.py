"""
Cryptocurrency data layer for the Qlib factor/model scenarios.

Design goals
------------
- **Multi-coin cross-section**: the Qlib loop is cross-sectional, so crypto needs a
  coin universe rather than a single symbol.
- **User-selectable**: exchange / market / quote / frequency / universe / date range
  are all configuration, never hard-coded.
- **Two outputs**: the CoSTEER coder reads an H5 snapshot, the runner reads Qlib
  binary data. Both are produced from the same downloaded frames.
- **Non-destructive**: crypto datasets live in their own directory and never
  overwrite ``cn_data``. ``CRYPTO_MODE=false`` keeps the original A-share path.
"""

from rdagent.scenarios.qlib.crypto.conf import CryptoDataSettings, crypto_enabled

__all__ = ["CryptoDataSettings", "crypto_enabled"]
