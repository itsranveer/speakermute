"""User-facing error types. The CLI prints these without a traceback."""


class SpeakermuteError(Exception):
    """Base class for errors that should be shown to the user as a message."""


class MissingTokenError(SpeakermuteError):
    """No Hugging Face token available for the gated pyannote models."""


class TranscriptError(SpeakermuteError):
    """transcript.json is missing or malformed."""


class SpeakerNotFoundError(SpeakermuteError):
    """The requested speaker label does not occur in the transcript."""
