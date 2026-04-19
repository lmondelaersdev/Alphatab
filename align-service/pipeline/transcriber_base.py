"""Stub that documents how to plug in a transcriber (TabCNN / MT3 / YourMT3+).

Not wired into `/align` — alignment is score-informed and does not need
transcription. Ship this file so the extension point is discoverable.

Example::

    from pipeline.registry import register_transcriber
    from pipeline.base import Transcriber, NoteEvent

    @register_transcriber("tabcnn")
    class TabCNNTranscriber(Transcriber):
        def transcribe(self, audio_path: str) -> list[NoteEvent]:
            # run your model...
            return events

Then a future `/transcribe?model=tabcnn` endpoint can look the class up via
:func:`pipeline.registry.TRANSCRIBERS` without any change to `app.py`.
"""
