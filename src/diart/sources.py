import asyncio
import base64
from pathlib import Path
from queue import SimpleQueue
from typing import Text, Optional, AnyStr

import numpy as np
import sounddevice as sd
import torch
import websockets
from einops import rearrange
from rx.subject import Subject
import signal

from .audio import FilePath, AudioLoader


class AudioSource:
    """Represents a source of audio that can start streaming via the `stream` property.

    Parameters
    ----------
    uri: Text
        Unique identifier of the audio source.
    sample_rate: int
        Sample rate of the audio source.
    """
    def __init__(self, uri: Text, sample_rate: int):
        self.uri = uri
        self.sample_rate = sample_rate
        self.stream = Subject()

    @property
    def duration(self) -> Optional[float]:
        """The duration of the stream if known. Defaults to None (unknown duration)."""
        return None

    def read(self):
        """Start reading the source and yielding samples through the stream."""
        raise NotImplementedError

    def close(self):
        """Stop reading the source and close all open streams."""
        raise NotImplementedError
