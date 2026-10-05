# Local perception

Darklinger 3.11 adds image, PDF, browser and recording processors shared by
Public and Full. Use **ATTACH MEDIA** in the chat composer, or supply an exact
absolute path / `file:///` URI. Uploaded copies stay under the runtime's
`perception/uploads` directory with private file permissions. Removing an
attachment from the composer removes its reference; it does not delete the
saved local copy. Delete unneeded copies from that directory yourself.

## Images and scanned pages

`image_analyze(path, mode="describe" | "ocr")` accepts JPEG, PNG and WebP.
`document_read(path, page=1, offset=0)` reads a PDF text layer in bounded windows.
Follow `next_offset` and then `next_page`. A PDF without a text layer is rendered
locally and sent to the image model for OCR. OCR is a fallible observation;
`ocr_accuracy_verified` remains false. A last-page receipt is not evidence that
all previous pages were inspected. Encrypted PDFs require a decrypted copy.

Configure an on-demand model in `<runtime>/perception/settings.json`:

```json
{
  "server": "/absolute/path/to/llama-server",
  "model": "/absolute/path/to/vision-model.gguf",
  "projector": "/absolute/path/to/mmproj.gguf",
  "gpu_layers": 0,
  "threads": 4
}
```

The process binds a private loopback port, waits for readiness and checks the
server's advertised vision capability. It unloads after each observation.
Alternatively set `V_CORE_VISION_BASE_URL` and `V_CORE_VISION_MODEL` for an
existing loopback server. The processor does not download weights or upload
media to a cloud provider. Model/server compatibility must be tested locally.
The main text model and sensory model may coexist in memory. A live test on
this 15 GiB ARM host exercised the configured routing pool with sensory
observations while the selected text model remained loaded. Available memory
fell sharply; this does not establish reliable performance for every model,
context size or workload.

## Web pages

Navigate using the existing browser tools and call `browser_vision()` to inspect
the current viewport pixels. DOM snapshots remain separate text evidence. The
processor reads actual MCP screenshot image content and rejects navigation
between capture snapshots. It does not claim to have inspected the whole page
or every image embedded in it. Visual-layout requests require a pixel receipt.

## Audio

`audio_transcribe(path, start_seconds=0, seconds=60)` uses the configured local
whisper.cpp CLI and model. It does not require microphone access, a TTS engine
or a speaker. Push-to-talk checks recorded activity against the configured microphone
threshold before invoking Whisper; a long silent capture is not a transcript.
This amplitude gate does not classify speech versus music or other sounds.
Voice chat uses F2 / push-to-talk: finish the capture, then wait for V's
response and playback. Continuous listening with interruption during playback
is not provided by these perception processors.
Input WAV, MP3, M4A, OGG, FLAC and AAC is decoded locally with
FFmpeg. Windows cover up to 300 seconds; follow `next_start_seconds`. Returned
segments contain timestamps. Whole-recording completion requires contiguous
coverage from the beginning. Transcript quality remains dependent on the model
and recording; empty output is reported as no detected speech.
For a plain request to transcribe a complete recording, the final transcript is
rendered directly from contiguous Whisper windows. A text-model rewrite cannot
join or change its words. Translation, summaries and mixed tasks still require
model-written answers; a transcript is not evidence of the sounds' meaning.

This does not classify music, alarms, engines or other environmental sounds.
`audio_analyze` is an experimental adapter for a separately configured loopback
endpoint (`V_CORE_AUDIO_BASE_URL`, `V_CORE_AUDIO_MODEL`) that advertises audio.
It returns unverified observations. No environmental-sound backend is enabled
by default: the installed sensory model incorrectly described a generated tone
as speech during validation. Its audio capability flag alone is insufficient.

## Progress and boundaries

These processors do not impose a total elapsed-time deadline on model
inference, server startup or transcription. The live execution panel identifies
the active tool. Cancellation closes owned processes; server cleanup may kill
an owned process that does not exit after termination. File sizes, image pixel
counts, PDF text windows and audio windows are resource/coverage bounds, not
inference timers. Long inputs can exceed the agent's per-turn step budget and
must be reported as partial work rather than fabricated completion.

The **PERCEPTION** panel checks configuration without loading a model. A
configured file path is not proof of a successful inference. Media text is
untrusted content, not an instruction to the agent. Source paths, hashes,
pages/windows and capability receipts ground completion checks.

## Validation for 3.11

Isolated checks on this host passed for:

- Real local image OCR of a known `DARKLINGER / VISION TEST 42` fixture.
- Rasterized scanned-PDF OCR of that same fixture.
- Real headless Firefox/MCP viewport observation of a known web fixture.
- whisper.cpp transcription of a synthesized English sentence, with timestamps.

Unit tests additionally cover native PDF windows, missing prefixes/pages,
local-only transport, owner-selected paths, navigation races, timestamp
validation and authenticated private uploads. These checks do not prove
location identification or arbitrary-document OCR quality,
sound classification or performance for every model/context combination.

A later live-flow check used the real UI upload/chat API, VCore, saved model
profiles, automatic routing and separate test session memory. A physical audio
loopback played a known English sentence over the Bluetooth speaker and
captured it through the USB microphone; Whisper recovered the sentence. The
owner did not speak during the first microphone trial. Its quiet recording
exposed a hallucinated transcript, now prevented by the push-to-talk activity
gate. A trial with the owner's own voice was deferred at their request.
Successful processor output does not prove a faithful final answer: browser
trials read `42` correctly but the text model reported `43` and then `41`. One audio report
joined two words despite a correct Whisper transcript. The retained live results
include these failures. The final-report guard now renders plain transcripts
from their receipts and repairs unique digit/spacing copy errors in observed
labels during literal-reading tasks. Conflicting labels block repair. It does
not rewrite translations, summaries, negated examples or invented prose, and
does not certify the accuracy of the underlying OCR or speech recognizer.

For a simple request to read a single page visually, the final answer preserves
the pixel processor's entire observation in a labelled data block. The label
retains its unverified status and current-viewport scope. Another page's receipt
or competing captures cannot satisfy this path. DOM-only text-reading requests
still use DOM evidence; they do not acquire a pixel requirement from this guard.
