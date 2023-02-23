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
        
class FileAudioSource(AudioSource):
    """Represents an audio source tied to a file.
    Parameters
    ----------
    file: FilePath
        Path to the file to stream.
    sample_rate: int
        Sample rate of the chunks emitted.
    """
    def __init__(
        self,
        file: FilePath,
        sample_rate: int,
        padding_end: float = 0,
        block_size: int = 1000,
    ):
        super().__init__(Path(file).stem, sample_rate)
        self.loader = AudioLoader(self.sample_rate, mono=True)
        self._duration = self.loader.get_duration(file)
        self.file = file
        self.resolution = 1 / self.sample_rate
        self.block_size = block_size
        self.padding_end = padding_end
        self.is_closed = False

    @property
    def duration(self) -> Optional[float]:
        # The duration of a file is known
        return self._duration + self.padding_end

    def read(self):
        """Send each chunk of samples through the stream"""
        waveform = self.loader.load(self.file)

        # Add zero padding at the end if required
        if self.padding_end > 0:
            num_pad_samples = int(np.rint(self.padding_end * self.sample_rate))
            zero_padding = torch.zeros(waveform.shape[0], num_pad_samples)
            waveform = torch.cat([waveform, zero_padding], dim=1)

        # Split into blocks
        _, num_samples = waveform.shape
        chunks = rearrange(
            waveform.unfold(1, self.block_size, self.block_size),
            "channel chunk sample -> chunk channel sample",
        ).numpy()

        # Add last incomplete chunk with padding
        if num_samples % self.block_size != 0:
            last_chunk = waveform[:, chunks.shape[0] * self.block_size:].unsqueeze(0).numpy()
            diff_samples = self.block_size - last_chunk.shape[-1]
            last_chunk = np.concatenate([last_chunk, np.zeros((1, 1, diff_samples))], axis=-1)
            chunks = np.vstack([chunks, last_chunk])

        # Stream blocks
        for i, waveform in enumerate(chunks):
            try:
                if self.is_closed:
                    break
                self.stream.on_next(waveform)
            except BaseException as e:
                self.stream.on_error(e)
                break
        self.stream.on_completed()
        self.close()

    def close(self):
        self.is_closed = True


class MicrophoneAudioSource(AudioSource):
    """Represents an audio source tied to the default microphone available"""

    def __init__(self, sample_rate: int, block_size: int = 1000):
        super().__init__("live_recording", sample_rate)
        self.block_size = block_size
        self._mic_stream = sd.InputStream(
            channels=1,
            samplerate=sample_rate,
            latency=0,
            blocksize=self.block_size,
            callback=self._read_callback
        )
        self._queue = SimpleQueue()

    def _read_callback(self, samples, *args):
        self._queue.put_nowait(samples[:, [0]].T)

    def read(self):
        self._mic_stream.start()
        while self._mic_stream:
            try:
                while self._queue.empty():
                    if self._mic_stream.closed:
                        break
                self.stream.on_next(self._queue.get_nowait())
            except BaseException as e:
                self.stream.on_error(e)
                break
        self.stream.on_completed()
        self.close()

    def close(self):
        self._mic_stream.stop()
        self._mic_stream.close()
