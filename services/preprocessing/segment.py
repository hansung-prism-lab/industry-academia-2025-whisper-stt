import numpy as np
import librosa
import logging
from typing import List, Tuple

from .config import AudioProcessingConfig


class AudioSegmentProcessor:
    def __init__(self, config: AudioProcessingConfig):
        self.config = config

    def calculate_energy_threshold(self, audio: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray, float]:
        rms = librosa.feature.rms(
            y=audio,
            frame_length=self.config.FRAME_LENGTH,
            hop_length=self.config.HOP_LENGTH
        )[0]

        times = librosa.frames_to_time(
            np.arange(len(rms)),
            sr=sr,
            hop_length=self.config.HOP_LENGTH
        )

        background_energy = np.percentile(rms, self.config.ENERGY_PERCENTILE)
        threshold = background_energy * self.config.ENERGY_MULTIPLIER

        return rms, times, threshold

    def refine_segment_boundaries(self, audio: np.ndarray, sr: int,
                                  start_time: float, end_time: float,
                                  threshold: float) -> Tuple[float, float]:
        hop_length = self.config.HOP_LENGTH
        start_frame = int(start_time * sr / hop_length)
        end_frame = int(end_time * sr / hop_length)

        rms = librosa.feature.rms(y=audio, frame_length=self.config.FRAME_LENGTH, hop_length=hop_length)[0]

        if start_frame >= len(rms) or end_frame > len(rms):
            return start_time, end_time

        segment_energy = rms[start_frame:end_frame]
        above_threshold = segment_energy > threshold

        if not np.any(above_threshold):
            return start_time, end_time

        active_indices = np.where(above_threshold)[0]
        first_active = active_indices[0]
        last_active = active_indices[-1]

        duration = end_time - start_time
        time_per_frame = duration / len(segment_energy)

        refined_start = start_time + (first_active * time_per_frame)
        refined_end = start_time + ((last_active + 1) * time_per_frame)

        return refined_start, refined_end

    def refine_segments_with_energy(self, audio_file_path: str,
                                    raw_segments: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
        audio, sr = librosa.load(audio_file_path, sr=self.config.SAMPLE_RATE)
        rms, times, threshold = self.calculate_energy_threshold(audio, sr)

        refined_segments = []

        for start_time, end_time, duration in raw_segments:
            hop_length = self.config.HOP_LENGTH
            start_frame = int(start_time * sr / hop_length)
            end_frame = int(end_time * sr / hop_length)

            if start_frame < len(rms) and end_frame <= len(rms):
                segment_energy = rms[start_frame:end_frame]

                if len(segment_energy) > 0 and np.max(segment_energy) > threshold:
                    refined_start, refined_end = self.refine_segment_boundaries(
                        audio, sr, start_time, end_time, threshold
                    )
                    refined_duration = refined_end - refined_start

                    if refined_duration >= self.config.MIN_SEGMENT_DURATION:
                        refined_segments.append((refined_start, refined_end, refined_duration))

        return refined_segments


class SegmentAdjuster:
    def __init__(self, config: AudioProcessingConfig):
        self.config = config

    def adjust_to_target_count(self, segments: List[Tuple[float, float, float]],
                               target_count: int, audio_file_path: str) -> List[Tuple[float, float, float]]:
        if len(segments) == target_count:
            return segments

        logging.info(f"구간 조정: {len(segments)}개 → {target_count}개")

        if len(segments) > target_count:
            return self._merge_excessive_segments(segments, target_count)
        else:
            return self._split_or_detect_missing(segments, target_count, audio_file_path)

    def _merge_excessive_segments(self, segments: List[Tuple[float, float, float]],
                                  target_count: int) -> List[Tuple[float, float, float]]:
        merged = segments.copy()

        while len(merged) > target_count:
            min_gap = float('inf')
            merge_idx = -1

            for i in range(len(merged) - 1):
                gap = merged[i + 1][0] - merged[i][1]
                if gap < min_gap:
                    min_gap = gap
                    merge_idx = i

            if merge_idx >= 0:
                start1, _, _ = merged[merge_idx]
                _, end2, _ = merged[merge_idx + 1]
                new_duration = end2 - start1

                merged[merge_idx] = (start1, end2, new_duration)
                merged.pop(merge_idx + 1)

        return merged

    def _split_or_detect_missing(self, segments: List[Tuple[float, float, float]],
                                 target_count: int, audio_file_path: str) -> List[Tuple[float, float, float]]:
        extended_segments = []

        for start, end, duration in segments:
            if (duration > self.config.MAX_SPLIT_DURATION and
                    len(extended_segments) + len(segments) - len(extended_segments) < target_count):
                mid_point = start + duration / 2
                extended_segments.append((start, mid_point, mid_point - start))
                extended_segments.append((mid_point, end, end - mid_point))
            else:
                extended_segments.append((start, end, duration))

        if len(extended_segments) < target_count:
            try:
                additional_segments = self._detect_additional_segments(audio_file_path, extended_segments)
                extended_segments.extend(additional_segments)
                extended_segments.sort()
            except Exception as e:
                logging.warning(f"추가 구간 탐지 실패: {e}")

        return extended_segments[:target_count]

    def _detect_additional_segments(self, audio_file_path: str,
                                    existing_segments: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
        additional = []

        try:
            audio, sr = librosa.load(audio_file_path, sr=self.config.SAMPLE_RATE)
            hop_length = self.config.HOP_LENGTH
            rms = librosa.feature.rms(y=audio, frame_length=self.config.FRAME_LENGTH, hop_length=hop_length)[0]

            background_energy = np.percentile(rms, self.config.LOW_ENERGY_PERCENTILE)
            low_threshold = background_energy * self.config.LOW_ENERGY_MULTIPLIER

            for i in range(len(existing_segments) - 1):
                gap_start = existing_segments[i][1]
                gap_end = existing_segments[i + 1][0]

                if gap_end - gap_start > self.config.MIN_GAP_DURATION:
                    gap_segments = self._find_segments_in_gap(
                        rms, gap_start, gap_end, low_threshold, sr, hop_length
                    )
                    additional.extend(gap_segments)

        except Exception as e:
            logging.error(f"추가 구간 탐지 중 오류: {e}")

        return additional

    def _find_segments_in_gap(self, rms: np.ndarray, gap_start: float, gap_end: float,
                              threshold: float, sr: int, hop_length: int) -> List[Tuple[float, float, float]]:
        start_frame = int(gap_start * sr / hop_length)
        end_frame = int(gap_end * sr / hop_length)

        if start_frame >= len(rms) or end_frame > len(rms):
            return []

        gap_energy = rms[start_frame:end_frame]
        above_threshold = gap_energy > threshold

        if not np.any(above_threshold):
            return []

        diff = np.diff(np.concatenate(([False], above_threshold, [False])).astype(int))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]

        segments = []
        for start_idx, end_idx in zip(starts, ends):
            seg_start = gap_start + (start_idx * hop_length / sr)
            seg_end = gap_start + (end_idx * hop_length / sr)
            duration = seg_end - seg_start

            if duration >= self.config.MIN_SPLIT_DURATION:
                segments.append((seg_start, seg_end, duration))

        return segments


class SegmentMatcher:

    @staticmethod
    def handle_count_mismatch(voice_segments: List[Tuple[float, float, float]],
                              sentences: List[str]) -> Tuple[List[Tuple[float, float, float]], List[str]]:
        voice_count = len(voice_segments)
        text_count = len(sentences)

        if voice_count > text_count:
            sentences.extend([""] * (voice_count - text_count))
            logging.info(f"빈 텍스트 {voice_count - text_count}개 추가")

        elif text_count > voice_count:
            if voice_count > 0:
                excess = text_count - voice_count
                combined_text = " ".join(sentences[-excess - 1:])
                sentences = sentences[:-excess - 1] + [combined_text]
            else:
                sentences = sentences[:voice_count]

        return voice_segments, sentences
