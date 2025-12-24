from dataclasses import dataclass
from pathlib import Path


@dataclass
class AudioProcessingConfig:
    CHECKPOINT: str = "pyannote/voice-activity-detection"
    ACCESS_TOKEN: str = None
    SAMPLE_RATE: int = 16000
    HOP_LENGTH: int = 512
    FRAME_LENGTH: int = 1024
    MIN_SEGMENT_DURATION: float = 0.2
    MIN_SPLIT_DURATION: float = 0.3
    MAX_SPLIT_DURATION: float = 3.0
    ENERGY_PERCENTILE: int = 20
    ENERGY_MULTIPLIER: float = 3.0
    LOW_ENERGY_PERCENTILE: int = 10
    LOW_ENERGY_MULTIPLIER: float = 1.5
    MIN_GAP_DURATION: float = 0.5


@dataclass
class ProcessingPaths:
    audio_dir: str
    label_dir: str
    output_dir: str = "output"

    def __post_init__(self):
        self.audio_path = Path(self.audio_dir)
        self.label_path = Path(self.label_dir)
        self.output_path = Path(self.output_dir)
