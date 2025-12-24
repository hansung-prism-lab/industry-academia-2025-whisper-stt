import pandas as pd
import logging
from typing import List, Dict


class DatasetBuilder:
    @staticmethod
    def create_dataframe(saved_files: List[Dict], sentences: List[str],
                         original_filename: str) -> pd.DataFrame:
        data = []
        for i, (file_info, sentence) in enumerate(zip(saved_files, sentences)):
            data.append({
                'original_audio': original_filename,
                'segment_id': i + 1,
                'audio_file_path': file_info['file_path'],
                'start_time': file_info['start_time'],
                'end_time': file_info['end_time'],
                'text': sentence,
            })

        return pd.DataFrame(data)

    @staticmethod
    def validate_and_summarize(df: pd.DataFrame) -> None:
        if len(df) == 0:
            return

        for i in range(min(3, len(df))):
            row = df.iloc[i]
            text_preview = row['text'][:50] + ('...' if len(row['text']) > 50 else '')
            logging.info(f"  샘플 {i + 1}: {row['start_time']:.1f}-{row['end_time']:.1f}s: '{text_preview}'")
