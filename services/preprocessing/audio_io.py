import librosa
import soundfile as sf
from pathlib import Path
from typing import List, Tuple, Dict


class AudioFileSaver:

    @staticmethod
    def save_audio_segments(audio_file_path: str, segments: List[Tuple[float, float, float]],
                            output_dir: str) -> List[Dict]:
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            audio, sr = librosa.load(audio_file_path, sr=None)
            base_name = Path(audio_file_path).stem

            saved_files = []
            for i, (start_time, end_time, duration) in enumerate(segments):
                start_sample = int(start_time * sr)
                end_sample = int(end_time * sr)
                segment_audio = audio[start_sample:end_sample]

                output_filename = f"{base_name}_seg_{i + 1:03d}.wav"
                output_file_path = output_path / output_filename

                sf.write(output_file_path, segment_audio, sr)

                saved_files.append({
                    'file_path': str(output_file_path),
                    'segment_id': i + 1,
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': duration
                })
            return saved_files

        except Exception as e:
            return []
