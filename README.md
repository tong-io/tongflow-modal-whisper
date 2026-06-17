# tongflow-modal-whisper

Official [TongFlow](https://github.com/tong-io/tongflow) plugin. Speech recognition with **Whisper** (OpenAI Whisper, `base` model by default), running on a GPU via [Modal](https://modal.com). An alternative to `tongflow-modal-qwen3asr` on the same transcription slots.

## Capabilities

- **Speech recognition** (`transcribe`) — transcribe speech from audio or video.
- **Speech recognition with timestamps** (`transcribe-timestamp`) — transcribe with segment timing.

## Credentials

Add in TongFlow **Settings** (gear icon, top-right):

| Key | Required | Notes |
| --- | --- | --- |
| `MODAL_TOKEN_ID` | ✅ | Create at [modal.com/settings/tokens](https://modal.com/settings/tokens). |
| `MODAL_TOKEN_SECRET` | ✅ | Paired with `MODAL_TOKEN_ID`. |
| `WHISPER_MODEL` | optional | Whisper size — `tiny` / `base` (default) / `small` / `medium` / `large`. |

On first use the plugin deploys to your Modal account automatically and caches the build. Whisper weights download from OpenAI — no Hugging Face token required.
