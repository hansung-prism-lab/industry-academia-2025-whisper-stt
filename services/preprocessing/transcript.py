import json
import os
import re
import logging
from typing import List


class TranscriptProcessor:

    @staticmethod
    def load_transcript(json_file_path: str) -> str:
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data.get('Transcript', '')
        except Exception as e:
            return ""

    @staticmethod
    def split_into_sentences(transcript: str) -> List[str]:
        if not transcript:
            return []

        sentences = re.split(r'[.!?]+\s*', transcript)
        sentences = [s.strip() for s in sentences if s.strip()]

        logging.info(f"분할된 문장 수: {len(sentences)}")
        return sentences

    @classmethod
    def process_transcript_file(cls, json_file_path: str) -> List[str]:
        logging.info(f"transcript 처리 시작: {os.path.basename(json_file_path)}")

        transcript = cls.load_transcript(json_file_path)
        if not transcript:
            logging.warning("빈 transcript")
            return []

        sentences = cls.split_into_sentences(transcript)

        for i, sentence in enumerate(sentences[:3], 1):
            logging.info(f"문장 {i}: {sentence}")
        if len(sentences) > 3:
            logging.info(f"... 총 {len(sentences)}개 문장")

        return sentences
