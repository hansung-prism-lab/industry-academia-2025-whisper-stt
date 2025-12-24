import warnings
warnings.filterwarnings('ignore')

import os
import logging
import pandas as pd
from typing import Optional
from dotenv import load_dotenv

from .preprocessing import (
    AudioProcessingConfig,
    ProcessingPaths,
    TranscriptProcessor,
    VoiceActivityDetector,
    AudioSegmentProcessor,
    SegmentAdjuster,
    SegmentMatcher,
    AudioFileSaver,
    DatasetBuilder,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()
config = AudioProcessingConfig()

paths = ProcessingPaths(
    audio_dir="data/raw_data/audio",
    label_dir="data/raw_data/label",
    output_dir="data/final_data"
)


class AudioTextMatchingPipeline:

    def __init__(self, config: AudioProcessingConfig):
        self.config = config
        self.vad = VoiceActivityDetector(config)
        self.segment_processor = AudioSegmentProcessor(config)
        self.segment_adjuster = SegmentAdjuster(config)

    def process_single_file(self, audio_file_path: str, label_file_path: str,
                            output_dir: str) -> Optional[pd.DataFrame]:
        logging.info(f"파일 처리 시작: {os.path.basename(audio_file_path)}")

        sentences = TranscriptProcessor.process_transcript_file(label_file_path)
        if not sentences:
            logging.error("텍스트 분할 실패")
            return None

        target_count = len(sentences)

        raw_segments = self.vad.extract_raw_segments(audio_file_path)
        if not raw_segments:
            logging.error("VAD 구간 추출 실패")
            return None

        refined_segments = self.segment_processor.refine_segments_with_energy(
            audio_file_path, raw_segments
        )

        adjusted_segments = self.segment_adjuster.adjust_to_target_count(
            refined_segments, target_count, audio_file_path
        )

        if len(adjusted_segments) != target_count:
            adjusted_segments, sentences = SegmentMatcher.handle_count_mismatch(
                adjusted_segments, sentences
            )

        saved_files = AudioFileSaver.save_audio_segments(
            audio_file_path, adjusted_segments, output_dir
        )

        if not saved_files:
            return None

        df = DatasetBuilder.create_dataframe(
            saved_files, sentences, os.path.basename(audio_file_path)
        )

        DatasetBuilder.validate_and_summarize(df)

        return df

    def process_batch(self, paths: ProcessingPaths) -> Optional[pd.DataFrame]:
        audio_files = [f for f in os.listdir(paths.audio_dir) if f.endswith('.wav')]
        audio_files.sort()

        results = {}
        successful = 0
        failed = 0

        for i, audio_file in enumerate(audio_files, 1):
            json_file = audio_file.replace('.wav', '.json')
            audio_path = os.path.join(paths.audio_dir, audio_file)
            label_path = os.path.join(paths.label_dir, json_file)

            if not os.path.exists(label_path):
                failed += 1
                continue

            file_output_dir = os.path.join(paths.output_dir, audio_file.replace('.wav', ''))

            df = self.process_single_file(audio_path, label_path, file_output_dir)

            if df is not None and len(df) > 0:
                results[audio_file] = df
                successful += 1
                logging.info(f"{audio_file} 완료 ({len(df)}개 샘플)")
            else:
                failed += 1

        if results:
            combined_df = pd.concat(results.values(), ignore_index=True)

            output_csv = os.path.join(paths.output_dir, "training_dataset.csv")
            combined_df.to_csv(output_csv, index=False, encoding='utf-8-sig')

            return combined_df
        else:
            return None


def main():
    pipeline = AudioTextMatchingPipeline(config)
    result_df = pipeline.process_batch(paths)
    return result_df


if __name__ == "__main__":
    final_dataset = main()
