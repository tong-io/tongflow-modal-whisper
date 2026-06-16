"""Modal deploy entry for whisper.

Deploy:
  modal deploy deploy.py
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import modal
from tongflow import deploy
from tongflow.models.transcribe import TranscribeInput, TranscribeOutput
from tongflow.models.transcribe_timestamp import (
    TranscribeTimestampInput,
    TranscribeTimestampOutput,
)
from tongflow.node_slots import NodeSlots
from tongflow.slots import node_slot


_cfg: dict[str, Any] = {}
_volume_name = str(_cfg.get("volumeName") or "whisper-models")
# Whisper model selection is a deploy-time knob; not part of the ABI input.
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "git",
        "build-essential",
        "cmake",
        "ffmpeg",
        "wget",
        "pkg-config",
        "libavcodec-dev",
        "libavformat-dev",
        "libavutil-dev",
    )
    .run_commands(
        "git clone --recursive https://github.com/abdeladim-s/pywhispercpp.git /root/pywhispercpp",
        "cd /root/pywhispercpp && WHISPER_FFMPEG=1 pip install .",
    )
    .pip_install(
        "tongflow==0.1.0",
        "requests",
    )
)

app = modal.App(Path(__file__).resolve().parent.name, image=image)
secrets = modal.Secret.from_dict({})

MODEL_DIR = "/root/models"
model_volume = modal.Volume.from_name(_volume_name, create_if_missing=True)

def _transcribe_asset(audio_asset, *, model_name: str, language: str) -> dict:
    from pywhispercpp.model import Model
    from tongflow.protocol import asset_as_path

    try:
        with asset_as_path(audio_asset, suffix=".bin") as media_path:
            model = Model(model_name, models_dir=MODEL_DIR, n_threads=4)
            segments = model.transcribe(str(media_path), language=language)
            full_text = "，".join([s.text for s in segments])
            return {
                "success": True,
                "text": full_text,
                "segments": [
                    {"start": s.t0, "end": s.t1, "text": s.text} for s in segments
                ],
                "language": language,
                "model": model_name,
            }
    except Exception as e:
        logger.error(f"transcription error: {e}", exc_info=True)
        return {"success": False, "error": f"transcription error: {e}"}


@deploy
@app.cls(
    scaledown_window=5,
    cpu=4.0,
    memory=4096,
    timeout=600,
    secrets=[secrets],
    volumes={MODEL_DIR: model_volume},
)
class Inference:
    @modal.method()
    @node_slot(NodeSlots.TRANSCRIBE)
    def transcribe_slot(self, input: TranscribeInput) -> TranscribeOutput:
        if input.audio is None:
            return TranscribeOutput(success=False, error="Missing `audio` Asset")
        out = _transcribe_asset(
            input.audio,
            model_name=WHISPER_MODEL,
            language=input.language or "auto",
        )
        if not out.get("success"):
            return TranscribeOutput(
                success=False,
                error=str(out.get("error") or "transcribe failed"),
            )
        return TranscribeOutput(
            success=True,
            text=str(out.get("text") or ""),
        )

    @modal.method()
    @node_slot(NodeSlots.TRANSCRIBE_TIMESTAMP)
    def transcribe_timestamp(
        self,
        input: TranscribeTimestampInput,
    ) -> TranscribeTimestampOutput:
        if input.audio is None:
            return TranscribeTimestampOutput(
                success=False, error="Missing `audio` Asset"
            )
        out = _transcribe_asset(
            input.audio,
            model_name=WHISPER_MODEL,
            language=input.language or "auto",
        )
        if not out.get("success"):
            return TranscribeTimestampOutput(
                success=False,
                error=str(out.get("error") or "transcribe failed"),
            )
        from tongflow.models.transcribe_timestamp import (
            TranscribeTimestampOutputRootTimeStampsItem,
        )

        segs = out.get("segments") or []
        stamps = [
            TranscribeTimestampOutputRootTimeStampsItem(
                start=float(s.get("start") or 0.0),
                end=float(s.get("end") or 0.0),
                text=str(s.get("text") or ""),
            )
            for s in segs
            if isinstance(s, dict)
        ]
        return TranscribeTimestampOutput(
            success=True,
            text=str(out.get("text") or ""),
            time_stamps=stamps,
        )
