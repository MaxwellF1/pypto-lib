# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Public limits of the physical-dynamic prefill path.

Only the prefill programs (and the prefill-side hc_head) bound their runtime
token count against this ceiling, so it lives beside them rather than in
``config.py``: a constant in ``config.py`` is imported by every decode kernel in
this directory as well, which needlessly widens what a prefill-only change
touches.
"""

# Largest runtime token count the prefill kernels accept (Issue #905 P4).
PREFILL_MAX_TOKENS = 8192
