import argparse
import os
from typing import List

from yt_dlp import YoutubeDL

from lib.io_utils import ensure_dir


DEFAULT_QUERIES: List[str] = [
    'ytsearch25:"handstand hold floor"',
    'ytsearch25:"gymnast handstand hold floor"',
    'ytsearch25:"calisthenics handstand hold"',
    'ytsearch25:"yoga handstand hold still"',
]


def build_ydl_opts(out_dir: str, max_downloads: int) -> dict:
    return {
        "outtmpl": os.path.join(out_dir, "%(id)s.%(ext)s"),
        "noplaylist": True,
        "ignoreerrors": True,
        "quiet": False,
        "concurrent_fragment_downloads": 4,
        "retries": 3,
        "format": "mp4/best/bestvideo+bestaudio",
        "max_downloads": max_downloads,
    }


def main():
    parser = argparse.ArgumentParser(description="Download ~100 pro handstand videos via yt-dlp.")
    parser.add_argument("--out-dir", default="data/pro/raw", help="Output directory for raw videos")
    parser.add_argument("--max-downloads", type=int, default=120, help="Cap total downloads across all queries")
    parser.add_argument("--query", action="append", help="Custom yt-dlp query; can repeat. If set, overrides defaults.")
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    queries = args.query if args.query else DEFAULT_QUERIES

    remaining = args.max_downloads
    with YoutubeDL(build_ydl_opts(args.out_dir, args.max_downloads)) as ydl:
        for q in queries:
            if remaining <= 0:
                break
            # Limit per query to avoid blowing the cap
            per_query_cap = min(remaining, 40)
            ydl.params["max_downloads"] = per_query_cap
            print(f"Downloading: {q} (limit {per_query_cap})")
            try:
                ydl.download([q])
            except Exception as e:
                print(f"Warning: download error for query '{q}': {e}")
            # We cannot precisely count successes here; conservatively decrement by per_query_cap
            remaining -= per_query_cap


if __name__ == "__main__":
    main()


