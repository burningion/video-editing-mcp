import json
import sys
from collections import defaultdict

import osxphotos
from thefuzz import fuzz


def load_keywords(keyword_dict):
    # Convert string dict to actual dict if needed
    if isinstance(keyword_dict, str):
        keyword_dict = json.loads(keyword_dict)
    return {k.lower(): v for k, v in keyword_dict.items()}


def match_description(description, keyword_dict, threshold=60):
    keywords = load_keywords(keyword_dict)

    matches = defaultdict(int)
    words = description.lower().split()

    for word in words:
        for keyword in keywords:
            ratio = fuzz.ratio(word, keyword)
            if ratio > threshold:
                matches[keyword] = max(matches[keyword], ratio)

    # Return keywords sorted by match ratio
    return sorted(matches.items(), key=lambda x: x[1], reverse=True)


def get_videos_by_keyword(photosdb, keyword):
    # Use only_movies=True instead of is_video=True
    videos = photosdb.query(
        osxphotos.QueryOptions(
            label=[keyword], photos=False, movies=True, incloud=True, ignore_case=True
        )
    )

    # Convert to list of dictionaries if needed
    video_data = [video.asdict() for video in videos]

    return video_data


def find_and_export_videos(photosdb, keyword, export_path):
    videos = photosdb.query(
        osxphotos.QueryOptions(
            label=[keyword], photos=False, movies=True, incloud=True, ignore_case=True
        )
    )

    exported_files = []
    for video in videos:
        try:
            exported = video.export(export_path)
            exported_files.extend(exported)
            print(f"Exported {video.filename} to {exported}")
        except Exception as e:
            print(f"Error exporting {video.filename}: {e}")

    return exported_files


# Example usage
if __name__ == "__main__":
    """
    Usage: python search_local_videos.py <keyword>
    """
    if len(sys.argv) < 2:
        print("Usage: python search_local_videos.py <keyword>")
        sys.exit(1)
    photosdb = osxphotos.PhotosDB()
    video_dict = photosdb.labels_as_dict
    videos = get_videos_by_keyword(photosdb, sys.argv[1])
    for video in videos:
        print(
            f"Found video: {video.get('filename', 'Unknown')}, {video.get('labels', '')}"
        )
        print(f"number of items returned: {len(videos)}")
    # Example
    keywords = video_dict
    matches = match_description("me skateboarding", keywords)
    print(matches)
