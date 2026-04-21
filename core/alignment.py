import os
from typing import Iterable


def _clean_text(text):
    return " ".join(str(text or "").replace("\n", " ").split()).strip()


def parse_srt_file(path):
    if not path or not os.path.exists(path):
        return []

    try:
        import pysrt
    except ImportError:
        return []

    subtitles = pysrt.open(path, encoding="utf-8")
    segments = []
    for item in subtitles:
        text = _clean_text(item.text)
        if not text:
            continue
        segments.append(
            {
                "start": item.start.ordinal / 1000.0,
                "end": item.end.ordinal / 1000.0,
                "text": text,
            }
        )
    return segments


def write_srt_file(segments: Iterable[dict], path):
    try:
        import pysrt
    except ImportError:
        return None

    subtitle_file = pysrt.SubRipFile()
    for index, segment in enumerate(segments, start=1):
        text = _clean_text(segment.get("text"))
        if not text:
            continue
        subtitle_file.append(
            pysrt.SubRipItem(
                index=index,
                start=pysrt.SubRipTime(
                    milliseconds=int(float(segment.get("start", 0.0)) * 1000)
                ),
                end=pysrt.SubRipTime(
                    milliseconds=int(float(segment.get("end", 0.0)) * 1000)
                ),
                text=text,
            )
        )

    os.makedirs(os.path.dirname(path), exist_ok=True)
    subtitle_file.save(path, encoding="utf-8")
    return path


def align_text_to_timestamp(segments, timestamp, tolerance=1.5):
    if not segments:
        return ""

    matches = [
        segment["text"]
        for segment in segments
        if float(segment["start"]) <= timestamp <= float(segment["end"])
    ]
    if matches:
        unique_matches = []
        seen = set()
        for text in matches:
            if text not in seen:
                unique_matches.append(text)
                seen.add(text)
        return " ".join(unique_matches)

    closest_segment = None
    closest_distance = None
    for segment in segments:
        if timestamp < float(segment["start"]):
            distance = float(segment["start"]) - timestamp
        else:
            distance = timestamp - float(segment["end"])

        if closest_distance is None or distance < closest_distance:
            closest_distance = distance
            closest_segment = segment

    if closest_segment and closest_distance is not None and closest_distance <= tolerance:
        return closest_segment["text"]
    return ""
