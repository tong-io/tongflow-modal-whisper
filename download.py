"""Modal download entry for whisper.

Run:
  modal run download.py::download

Weights are cached on the Modal volume when the transcribe function runs;
this entrypoint is a no-op for consistency with other plugins.
"""

from __future__ import annotations

import modal

app = modal.App("whisper-download")


@app.local_entrypoint()
def download() -> None:
    print(
        "No separate weight download for whisper; "
        "models load into the volume on first transcribe."
    )
