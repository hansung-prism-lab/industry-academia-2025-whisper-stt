import torch
import logging
from typing import List, Tuple
from pyannote.audio import Pipeline

from .config import AudioProcessingConfig


class VoiceActivityDetector:
    def __init__(self, config: AudioProcessingConfig):
        self.config = config
        self.pipeline = self._initialize_pipeline()

    def _initialize_pipeline(self) -> Pipeline:
        try:
            pipeline = Pipeline.from_pretrained(
                self.config.CHECKPOINT,
                use_auth_token=self.config.ACCESS_TOKEN
            )
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            pipeline.to(device)
            return pipeline
        except Exception as e:
            logging.error(f"VAD 파이프라인 초기화 실패: {e}")
            raise

    def extract_raw_segments(self, audio_file_path: str) -> List[Tuple[float, float, float]]:
        try:
            vad_result = self.pipeline(audio_file_path)
            segments = []

            for segment in vad_result.get_timeline().support():
                duration = segment.end - segment.start
                if duration >= 0.1:
                    segments.append((segment.start, segment.end, duration))

            return segments
        except Exception as e:
            return []
