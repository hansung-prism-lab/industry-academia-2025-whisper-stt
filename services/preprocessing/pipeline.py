import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from .config import AudioProcessingConfig
from .vad import VoiceActivityDetector
from .segment import AudioSegmentProcessor, SegmentAdjuster, SegmentMatcher
from .audio_io import AudioFileSaver

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """파이프라인 처리 결과"""
    segments: List[Tuple[float, float, float]]
    saved_files: List[Dict]
    matched_texts: Optional[List[str]] = None
    success: bool = True
    error_message: Optional[str] = None


class AudioPreprocessingPipeline:

    def __init__(self, config: Optional[AudioProcessingConfig] = None):
        self.config = config or AudioProcessingConfig()
        self._vad: Optional[VoiceActivityDetector] = None
        self._processor: Optional[AudioSegmentProcessor] = None
        self._adjuster: Optional[SegmentAdjuster] = None
        self._saver = AudioFileSaver()

        logger.info("AudioPreprocessingPipeline 초기화 완료")

    @property
    def vad(self) -> VoiceActivityDetector:
        """VAD 인스턴스 (지연 로딩)"""
        if self._vad is None:
            logger.info("VAD 모델 로딩 중...")
            self._vad = VoiceActivityDetector(self.config)
            logger.info("VAD 모델 로딩 완료")
        return self._vad

    @property
    def processor(self) -> AudioSegmentProcessor:
        """세그먼트 프로세서 인스턴스"""
        if self._processor is None:
            self._processor = AudioSegmentProcessor(self.config)
        return self._processor

    @property
    def adjuster(self) -> SegmentAdjuster:
        """세그먼트 조정기 인스턴스"""
        if self._adjuster is None:
            self._adjuster = SegmentAdjuster(self.config)
        return self._adjuster

    def extract_segments(
        self,
        audio_path: str,
        refine_with_energy: bool = True
    ) -> List[Tuple[float, float, float]]:

        logger.info(f"세그먼트 추출 시작: {audio_path}")

        # 1단계: VAD로 원시 세그먼트 추출
        raw_segments = self.vad.extract_raw_segments(audio_path)
        logger.info(f"VAD 완료: {len(raw_segments)}개 세그먼트 감지")

        if not raw_segments:
            logger.warning("감지된 세그먼트가 없습니다")
            return []

        # 2단계: 에너지 기반 정제
        if refine_with_energy:
            refined_segments = self.processor.refine_segments_with_energy(
                audio_path, raw_segments
            )
            logger.info(f"에너지 정제 완료: {len(refined_segments)}개 세그먼트")
            return refined_segments

        return raw_segments

    def adjust_segments(
        self,
        segments: List[Tuple[float, float, float]],
        target_count: int,
        audio_path: str
    ) -> List[Tuple[float, float, float]]:

        if len(segments) == target_count:
            return segments

        logger.info(f"세그먼트 조정: {len(segments)}개 → {target_count}개")
        return self.adjuster.adjust_to_target_count(
            segments, target_count, audio_path
        )

    def match_with_texts(
        self,
        segments: List[Tuple[float, float, float]],
        texts: List[str]
    ) -> Tuple[List[Tuple[float, float, float]], List[str]]:

        return SegmentMatcher.handle_count_mismatch(segments, texts)

    def save_segments(
        self,
        audio_path: str,
        segments: List[Tuple[float, float, float]],
        output_dir: str
    ) -> List[Dict]:

        logger.info(f"세그먼트 저장 중: {len(segments)}개 → {output_dir}")
        saved_files = self._saver.save_audio_segments(
            audio_path, segments, output_dir
        )
        logger.info(f"저장 완료: {len(saved_files)}개 파일")
        return saved_files

    def process(
        self,
        audio_path: str,
        output_dir: str,
        target_segments: Optional[int] = None,
        texts: Optional[List[str]] = None,
        refine_with_energy: bool = True,
        save_files: bool = True
    ) -> PipelineResult:

        try:
            logger.info(f"파이프라인 시작: {audio_path}")

            # 1단계: 세그먼트 추출
            segments = self.extract_segments(audio_path, refine_with_energy)

            if not segments:
                return PipelineResult(
                    segments=[],
                    saved_files=[],
                    success=False,
                    error_message="세그먼트를 추출할 수 없습니다"
                )

            if target_segments is not None:
                segments = self.adjust_segments(segments, target_segments, audio_path)

            matched_texts = None
            if texts is not None:
                segments, matched_texts = self.match_with_texts(segments, texts.copy())


            saved_files = []
            if save_files:
                saved_files = self.save_segments(audio_path, segments, output_dir)

            logger.info(f"파이프라인 완료: {len(segments)}개 세그먼트")

            return PipelineResult(
                segments=segments,
                saved_files=saved_files,
                matched_texts=matched_texts,
                success=True
            )

        except Exception as e:
            logger.error(f"파이프라인 실행 중 오류: {e}")
            return PipelineResult(
                segments=[],
                saved_files=[],
                success=False,
                error_message=str(e)
            )

    def process_batch(
        self,
        audio_paths: List[str],
        output_base_dir: str,
        target_segments: Optional[int] = None
    ) -> List[PipelineResult]:

        import os
        from pathlib import Path

        results = []
        total = len(audio_paths)

        for i, audio_path in enumerate(audio_paths, 1):
            logger.info(f"처리 중 [{i}/{total}]: {audio_path}")


            stem = Path(audio_path).stem
            output_dir = os.path.join(output_base_dir, stem)

            result = self.process(
                audio_path=audio_path,
                output_dir=output_dir,
                target_segments=target_segments
            )
            results.append(result)

        success_count = sum(1 for r in results if r.success)
        logger.info(f"일괄 처리 완료: {success_count}/{total} 성공")

        return results
