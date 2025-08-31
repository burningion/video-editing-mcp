import logging
import os
import subprocess
import sys
import threading
import time
from typing import List, Optional, Union, Any, Dict
import json
import webbrowser
import uuid

import mcp.server.stdio
import mcp.types as types
import osxphotos
import requests
from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
from pydantic import AnyUrl

from transformers import AutoModel
from videojungle import ApiClient

from .search_local_videos import get_videos_by_keyword

import numpy as np


if os.environ.get("VJ_API_KEY"):
    VJ_API_KEY = os.environ.get("VJ_API_KEY")
else:
    try:
        VJ_API_KEY = sys.argv[1]
    except Exception:
        VJ_API_KEY = None

BROWSER_OPEN = False
# Configure the logging
logging.basicConfig(
    filename="app.log",  # Name of the log file
    level=logging.INFO,  # Log level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
)

if not VJ_API_KEY:
    try:
        with open(".env", "r") as f:
            for line in f:
                if "VJ_API_KEY" in line:
                    VJ_API_KEY = line.split("=")[1].strip()
    except Exception:
        raise Exception(
            "VJ_API_KEY environment variable is required or a .env file with the key is required"
        )

    if not VJ_API_KEY:
        raise Exception("VJ_API_KEY environment variable is required")

vj = ApiClient(VJ_API_KEY)


class PhotosDBLoader:
    def __init__(self):
        self._db: Optional[osxphotos.PhotosDB] = None
        self.start_loading()

    def start_loading(self):
        def load():
            self._db = osxphotos.PhotosDB()
            logging.info("PhotosDB loaded")

        thread = threading.Thread(target=load)
        thread.daemon = True  # Make thread exit when main program exits
        thread.start()

    @property
    def db(self) -> osxphotos.PhotosDB:
        if self._db is None:
            raise Exception("PhotosDB still loading")
        return self._db


class EmbeddingModelLoader:
    def __init__(self, model_name: str = "jinaai/jina-clip-v1"):
        self._model: Optional[AutoModel] = None
        self.model_name = model_name
        self.start_loading()

    def start_loading(self):
        def load():
            self._model = AutoModel.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            logging.info(f"Model {self.model_name} loaded")

        thread = threading.Thread(target=load)
        thread.daemon = True
        thread.start()

    @property
    def model(self) -> AutoModel:
        if self._model is None:
            raise Exception(f"Model {self.model_name} still loading")
        return self._model

    def encode_text(
        self,
        texts: Union[str, List[str]],
        truncate_dim: Optional[int] = None,
        task: Optional[str] = None,
    ) -> dict:
        """
        Encode text and format the embeddings in the expected JSON structure
        """
        embeddings = self.model.encode_text(texts, truncate_dim=truncate_dim, task=task)

        # Format the response in the expected structure
        return {"embeddings": embeddings.tolist(), "embedding_type": "text_embeddings"}

    def encode_image(
        self, images: Union[str, List[str]], truncate_dim: Optional[int] = None
    ) -> dict:
        """
        Encode images and format the embeddings in the expected JSON structure
        """
        embeddings = self.model.encode_image(images, truncate_dim=truncate_dim)

        return {"embeddings": embeddings.tolist(), "embedding_type": "image_embeddings"}

    def post_embeddings(
        self, embeddings: dict, endpoint_url: str, headers: Optional[dict] = None
    ) -> requests.Response:
        """
        Post embeddings to the specified endpoint
        """
        if headers is None:
            headers = {"Content-Type": "application/json"}

        response = requests.post(endpoint_url, json=embeddings, headers=headers)
        response.raise_for_status()
        return response


# Create global loader instance, (requires access to host computer!)
if sys.platform == "darwin" and os.environ.get("LOAD_PHOTOS_DB"):
    photos_loader = PhotosDBLoader()

model_loader = EmbeddingModelLoader()

server = Server("video-jungle-mcp")

try:
    # videos_at_start = vj.video_files.list()
    projects_at_start = vj.projects.list()
except Exception as e:
    logging.error(f"Error getting projects at start: {e}")
    videos_at_start = []

counter = 10

# Cache for pagination with timestamps for cleanup
_search_result_cache: Dict[str, Dict] = {}
_project_assets_cache: Dict[str, Dict] = {}
_CACHE_TTL = 60 * 4  # 4 minute cache TTL


# Function to clean old cache entries
def cleanup_cache():
    """Remove cache entries older than TTL."""
    current_time = time.time()
    search_keys_to_remove = []
    project_keys_to_remove = []

    # Clean search cache
    for key, cache_entry in _search_result_cache.items():
        if current_time - cache_entry["timestamp"] > _CACHE_TTL:
            search_keys_to_remove.append(key)

    for key in search_keys_to_remove:
        del _search_result_cache[key]

    # Clean project assets cache
    for key, cache_entry in _project_assets_cache.items():
        if current_time - cache_entry["timestamp"] > _CACHE_TTL:
            project_keys_to_remove.append(key)

    for key in project_keys_to_remove:
        del _project_assets_cache[key]

    total_removed = len(search_keys_to_remove) + len(project_keys_to_remove)
    if total_removed > 0:
        logging.info(
            f"Cleaned up {len(search_keys_to_remove)} expired search caches and {len(project_keys_to_remove)} project asset caches"
        )


tools = [
    "add-video",
    "search-local-videos",
    "search-remote-videos",
    "generate-edit-from-videos",
    "get-project-assets",
    "create-videojungle-project",
    "create-video-bar-chart-from-two-axis-data",
    "create-video-line-chart-from-two-axis-data",
    "edit-locally",
    "generate-edit-from-single-video",
    "update-video-edit",
]


def validate_y_values(y_values: Any) -> bool:
    """
    Validates that y_values is a single-dimensional array/list of numbers.

    Args:
        y_values: The input to validate

    Returns:
        bool: True if validation passes

    Raises:
        ValueError: If validation fails with a descriptive message
    """
    # Check if input is a list or numpy array
    if not isinstance(y_values, (list, np.ndarray)):
        raise ValueError("y_values must be a list")

    # Convert to numpy array for easier handling
    y_array = np.array(y_values)

    # Check if it's multi-dimensional
    if len(y_array.shape) > 1:
        raise ValueError("y_values must be a single-dimensional array")

    # Check if all elements are numeric
    if not np.issubdtype(y_array.dtype, np.number):
        raise ValueError("all elements in y_values must be numbers")

    # Check for NaN or infinite values
    if np.any(np.isnan(y_array)) or np.any(np.isinf(y_array)):
        raise ValueError("y_values cannot contain NaN or infinite values")

    return True


@server.list_resources()
async def handle_list_resources() -> list[types.Resource]:
    """
    List available video files.
    Each video files is available at a specific url
    """
    global counter, projects_at_start
    counter += 1
    # check to see if DaVinci Resolve is open

    # We do this counter because otherwise Claude is very aggressive
    # about requests
    if counter % 100 == 0:
        projects = vj.projects.list()
        projects_at_start = projects
        counter = 0
    """
    videos = [
        types.Resource(
            uri=AnyUrl(f"vj://video-file/{video.id}"),
            name=f"Video Jungle Video: {video.name}",
            description=f"User provided description: {video.description}",
            mimeType="video/mp4",
        )
        for video in videos_at_start
    ]"""

    projects = [
        types.Resource(
            uri=AnyUrl(f"vj://projects/{project.id}"),
            name=f"Video Jungle Project: {project.name}",
            description=f"Project description: {project.description}",
            mimeType="application/json",
        )
        for project in projects_at_start
    ]

    return projects  # videos  # + projects


@server.read_resource()
async def handle_read_resource(uri: AnyUrl) -> str:
    """
    Read a video's content by its URI.
    The video id is extracted from the URI host component.
    """
    if uri.scheme != "vj":
        raise ValueError(f"Unsupported URI scheme: {uri.scheme}")

    id = uri.path
    if id is not None:
        id = id.lstrip("/projects/")
        proj = vj.projects.get(id)
        logging.info(f"project is: {proj}")
        return proj.model_dump_json()
    raise ValueError(f"Project not found: {id}")


@server.list_prompts()
async def handle_list_prompts() -> list[types.Prompt]:
    """
    List available prompts.
    Each prompt can have optional arguments to customize its behavior.
    """
    return [
        types.Prompt(
            name="generate-local-search",
            description="Generate a local search for videos using appropriate label names from the Photos app.",
            arguments=[
                types.PromptArgument(
                    name="search_query",
                    description="Natural language query to be translated into Photos app label names.",
                    required=False,
                )
            ],
        )
    ]


@server.get_prompt()
async def handle_get_prompt(
    name: str, arguments: dict[str, str] | None
) -> types.GetPromptResult:
    """
    Generate a prompt by combining arguments with server state.
    The prompt includes all current notes and can be customized via arguments.
    """
    if name != "generate-local-search":
        raise ValueError(f"Unknown prompt: {name}")

    if not arguments:
        raise ValueError("Missing arguments")

    search_query = arguments.get("search_query")
    if not search_query:
        raise ValueError("Missing search_query")

    return types.GetPromptResult(
        description="Generate a local search for videos using appropriate label names from the Photos app.",
        messages=[
            types.PromptMessage(
                role="user",
                content=types.TextContent(
                    type="text",
                    text=f"Here are the exact label names you need to match in your query:\n\n For the specific query: {search_query}, you should use the following labels: {photos_loader.db.labels_as_dict} for the search-local-videos tool",
                ),
            )
        ],
    )


@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """
    List available tools.
    Each tool specifies its arguments using JSON Schema validation.
    """
    if os.environ.get("LOAD_PHOTOS_DB"):
        return [
            types.Tool(
                name="create-videojungle-project",
                description="Create a new Video Jungle project to create video edits, add videos, assets, and more.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Name of the project",
                        },
                        "description": {
                            "type": "string",
                            "description": "Description of the project",
                        },
                    },
                },
            ),
            types.Tool(
                name="edit-locally",
                description="Create an OpenTimelineIO file for local editing with the user's desktop video editing suite.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "edit_id": {
                            "type": "string",
                            "description": "UUID of the edit to download",
                        },
                        "project_id": {
                            "type": "string",
                            "description": "UUID of the project the video edit lives within",
                        },
                    },
                    "required": ["edit_id", "project_id"],
                },
            ),
            types.Tool(
                name="add-video",
                description="Upload video from URL. Begins analysis of video to allow for later information retrieval for automatic video editing an search.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "url": {"type": "string"},
                    },
                    "required": ["name", "url"],
                },
            ),
            types.Tool(
                name="search-remote-videos",
                description="Default method to search videos. Will return videos including video_ids, which allow for information retrieval and building video edits. For large result sets, you can paginate through chunks using search_id and page parameters.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Text search query"},
                        "limit": {
                            "type": "integer",
                            "default": 10,
                            "minimum": 1,
                            "description": "Maximum number of results to return per page",
                        },
                        "project_id": {
                            "type": "string",
                            "format": "uuid",
                            "description": "Project ID to scope the search",
                        },
                        "duration_min": {
                            "type": "number",
                            "minimum": 0,
                            "description": "Minimum video duration in seconds",
                        },
                        "duration_max": {
                            "type": "number",
                            "minimum": 0,
                            "description": "Maximum video duration in seconds",
                        },
                        "search_id": {
                            "type": "string",
                            "description": "ID of a previous search to continue pagination. If provided, returns the next chunk of results",
                        },
                        "page": {
                            "type": "integer",
                            "default": 1,
                            "minimum": 1,
                            "description": "Page number to retrieve when paginating through results",
                        },
                        "items_per_page": {
                            "type": "integer",
                            "default": 5,
                            "minimum": 1,
                            "maximum": 20,
                            "description": "Number of items to show per page when paginating",
                        },
                        "created_after": {
                            "type": "string",
                            "format": "date-time",
                            "description": "Filter videos created after this datetime",
                        },
                        "created_before": {
                            "type": "string",
                            "format": "date-time",
                            "description": "Filter videos created before this datetime",
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Set of tags to filter by",
                        },
                        "include_segments": {
                            "type": "boolean",
                            "default": True,
                            "description": "Whether to include video segments in results",
                        },
                        "include_related": {
                            "type": "boolean",
                            "default": False,
                            "description": "Whether to include related videos",
                        },
                        "query_audio": {
                            "type": "string",
                            "description": "Audio search query",
                        },
                        "query_img": {
                            "type": "string",
                            "description": "Image search query",
                        },
                    },
                },
            ),
            types.Tool(
                name="search-local-videos",
                description="Search user's local videos in Photos app by keyword",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "keyword": {"type": "string"},
                        "start_date": {
                            "type": "string",
                            "description": "ISO 8601 formatted datetime string (e.g. 2024-01-21T15:30:00Z)",
                        },
                        "end_date": {
                            "type": "string",
                            "description": "ISO 8601 formatted datetime string (e.g. 2024-01-21T15:30:00Z)",
                        },
                    },
                    "required": ["keyword"],
                },
            ),
            types.Tool(
                name="generate-edit-from-videos",
                description="Generate an edit from videos, from within a specific project. Creates a new project to work within no existing project ID (UUID) is passed ",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "project_id": {
                            "type": "string",
                            "description": "Either an existing Project UUID or String. A UUID puts the edit in an existing project, and a string creates a new project with that name.",
                        },
                        "name": {"type": "string", "description": "Video Edit name"},
                        "open_editor": {
                            "type": "boolean",
                            "description": "Open a live editor with the project's edit",
                        },
                        "resolution": {
                            "type": "string",
                            "description": "Video resolution. Examples include '1920x1080', '1280x720'",
                        },
                        "render_subtiles": {
                            "type": "boolean",
                            "description": "Whether to render subtitiles in the video edit",
                            "default": True,
                        },
                        "vertical_crop": {
                            "type": "string",
                            "description": "ML-powered automatic vertical crop mode. Pass 'standard' to enable automatic vertical video cropping",
                        },
                        "edit": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "video_id": {
                                        "type": "string",
                                        "description": "Video UUID",
                                    },
                                    "video_start_time": {
                                        "type": "string",
                                        "description": "Clip start time in HH:MM:SS.mmm format (e.g., '00:01:30.500' or '01:05:22.123'). Hours, minutes, seconds, and 3-digit milliseconds are required.",
                                    },
                                    "video_end_time": {
                                        "type": "string",
                                        "description": "Clip end time in HH:MM:SS.mmm format (e.g., '00:01:30.500' or '01:05:22.123'). Hours, minutes, seconds, and 3-digit milliseconds are required.",
                                    },
                                    "type": {
                                        "type": "string",
                                        "description": "Type of asset ('videofile' for video files, or 'user' for project specific assets)",
                                    },
                                    "audio_levels": {
                                        "type": "array",
                                        "description": "Optional audio level adjustments for this clip",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "audio_level": {
                                                    "type": "string",
                                                    "description": "Audio level (0.0 to 1.0)",
                                                }
                                            },
                                        },
                                    },
                                    "crop": {
                                        "type": "object",
                                        "description": "Optional crop/zoom settings for this video segment",
                                        "properties": {
                                            "zoom": {
                                                "type": "number",
                                                "minimum": 0.1,
                                                "maximum": 10.0,
                                                "default": 1.0,
                                                "description": "Zoom factor (1.0 = 100%, 1.5 = 150%, etc.)",
                                            },
                                            "position_x": {
                                                "type": "number",
                                                "minimum": -1.0,
                                                "maximum": 1.0,
                                                "default": 0.0,
                                                "description": "Horizontal offset from center (-1.0 to 1.0)",
                                            },
                                            "position_y": {
                                                "type": "number",
                                                "minimum": -1.0,
                                                "maximum": 1.0,
                                                "default": 0.0,
                                                "description": "Vertical offset from center (-1.0 to 1.0)",
                                            },
                                        },
                                    },
                                },
                            },
                            "description": "Array of video clips to include in the edit",
                        },
                        "audio_asset": {
                            "type": "object",
                            "properties": {
                                "audio_id": {
                                    "type": "string",
                                    "description": "Audio asset UUID",
                                },
                                "type": {
                                    "type": "string",
                                    "description": "Audio file type (e.g., 'mp3', 'wav')",
                                },
                                "filename": {
                                    "type": "string",
                                    "description": "Audio file name",
                                },
                                "audio_start_time": {
                                    "type": "string",
                                    "description": "Audio start time in HH:MM:SS.mmm format (e.g., '00:01:30.500' or '01:05:22.123'). Hours, minutes, seconds, and 3-digit milliseconds are required.",
                                },
                                "audio_end_time": {
                                    "type": "string",
                                    "description": "Audio end time in HH:MM:SS.mmm format (e.g., '00:01:30.500' or '01:05:22.123'). Hours, minutes, seconds, and 3-digit milliseconds are required.",
                                },
                                "url": {
                                    "type": "string",
                                    "description": "Optional URL for the audio file",
                                },
                                "audio_levels": {
                                    "type": "array",
                                    "description": "Optional audio level adjustments",
                                    "items": {"type": "object"},
                                },
                            },
                            "description": "Optional audio overlay for the video edit",
                        },
                    },
                    "required": ["edit", "name", "project_id"],
                },
            ),
            types.Tool(
                name="generate-edit-from-single-video",
                description="Generate a compressed video edit from a single video.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "project_id": {"type": "string"},
                        "resolution": {"type": "string"},
                        "video_id": {"type": "string"},
                        "render_subtiles": {
                            "type": "boolean",
                            "description": "Whether to render subtitiles in the video edit",
                            "default": True,
                        },
                        "vertical_crop": {
                            "type": "string",
                            "description": "ML-powered automatic vertical crop mode. Pass 'standard' to enable automatic vertical video cropping",
                        },
                        "edit": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "video_start_time": {
                                        "type": "string",
                                        "description": "Clip start time in HH:MM:SS.mmm format (e.g., '00:01:30.500' or '01:05:22.123'). Hours, minutes, seconds, and 3-digit milliseconds are required.",
                                    },
                                    "video_end_time": {
                                        "type": "string",
                                        "description": "Clip end time in HH:MM:SS.mmm format (e.g., '00:01:30.500' or '01:05:22.123'). Hours, minutes, seconds, and 3-digit milliseconds are required.",
                                    },
                                },
                            },
                            "description": "Array of time segments to extract from the video",
                        },
                    },
                    "required": ["edit", "project_id", "video_id"],
                },
            ),
            types.Tool(
                name="update-video-edit",
                description="Update an existing video edit within a specific project.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "project_id": {
                            "type": "string",
                            "description": "UUID of the project containing the edit",
                        },
                        "edit_id": {
                            "type": "string",
                            "description": "UUID of the video edit to update",
                        },
                        "name": {"type": "string", "description": "Video Edit name"},
                        "description": {
                            "type": "string",
                            "description": "Description of the video edit",
                        },
                        "video_output_format": {
                            "type": "string",
                            "description": "Output format for the video (e.g., 'mp4', 'webm')",
                        },
                        "video_output_resolution": {
                            "type": "string",
                            "description": "Video resolution. Examples include '1920x1080', '1280x720'",
                        },
                        "video_output_fps": {
                            "type": "number",
                            "description": "Frames per second for the output video",
                        },
                        "video_series_sequential": {
                            "type": "array",
                            "description": "Array of video clips in sequential order",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "video_id": {
                                        "type": "string",
                                        "description": "Video UUID",
                                    },
                                    "video_start_time": {
                                        "type": "string",
                                        "description": "Clip start time in HH:MM:SS.mmm format (e.g., '00:01:30.500' or '01:05:22.123'). Hours, minutes, seconds, and 3-digit milliseconds are required.",
                                    },
                                    "video_end_time": {
                                        "type": "string",
                                        "description": "Clip end time in HH:MM:SS.mmm format (e.g., '00:01:30.500' or '01:05:22.123'). Hours, minutes, seconds, and 3-digit milliseconds are required.",
                                    },
                                    "audio_levels": {
                                        "type": "array",
                                        "description": "Optional audio level adjustments for this clip",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "audio_level": {
                                                    "type": "string",
                                                    "description": "Audio level (0.0 to 1.0)",
                                                }
                                            },
                                        },
                                    },
                                    "crop": {
                                        "type": "object",
                                        "description": "Optional crop/zoom settings for this video segment",
                                        "properties": {
                                            "zoom": {
                                                "type": "number",
                                                "minimum": 0.1,
                                                "maximum": 10.0,
                                                "default": 1.0,
                                                "description": "Zoom factor (1.0 = 100%, 1.5 = 150%, etc.)",
                                            },
                                            "position_x": {
                                                "type": "number",
                                                "minimum": -1.0,
                                                "maximum": 1.0,
                                                "default": 0.0,
                                                "description": "Horizontal offset from center (-1.0 to 1.0)",
                                            },
                                            "position_y": {
                                                "type": "number",
                                                "minimum": -1.0,
                                                "maximum": 1.0,
                                                "default": 0.0,
                                                "description": "Vertical offset from center (-1.0 to 1.0)",
                                            },
                                        },
                                    },
                                },
                            },
                        },
                        "audio_overlay": {
                            "type": "object",
                            "description": "Audio overlay settings and assets",
                        },
                        "rendered": {
                            "type": "boolean",
                            "description": "Whether the edit has been rendered",
                        },
                        "vertical_crop": {
                            "type": "string",
                            "description": "ML-powered automatic vertical crop mode. Pass 'standard' to enable automatic vertical video cropping",
                        },
                    },
                    "required": ["project_id", "edit_id"],
                },
            ),
            types.Tool(
                name="create-video-bar-chart-from-two-axis-data",
                description="Create a video bar chart from two-axis data",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "x_values": {"type": "array", "items": {"type": "string"}},
                        "y_values": {"type": "array", "items": {"type": "number"}},
                        "x_label": {"type": "string"},
                        "y_label": {"type": "string"},
                        "title": {"type": "string"},
                        "filename": {"type": "string"},
                    },
                    "required": ["x_values", "y_values", "x_label", "y_label", "title"],
                },
            ),
            types.Tool(
                name="create-video-line-chart-from-two-axis-data",
                description="Create a video line chart from two-axis data",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "x_values": {"type": "array", "items": {"type": "string"}},
                        "y_values": {"type": "array", "items": {"type": "number"}},
                        "x_label": {"type": "string"},
                        "y_label": {"type": "string"},
                        "title": {"type": "string"},
                        "filename": {"type": "string"},
                    },
                    "required": ["x_values", "y_values", "x_label", "y_label", "title"],
                },
            ),
            types.Tool(
                name="get-project-assets",
                description="Get all assets and details for a specific project, with pagination support for large projects",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "project_id": {
                            "type": "string",
                            "description": "UUID of the project to retrieve assets for",
                        },
                        "asset_types": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of asset types to filter by (e.g. 'user', 'video', 'image', 'audio', 'generated_video', 'video_edit'). Video assets in a project are labeled 'user' for user uploaded, so prefer 'user' when building a video edit from project assets.",
                            "default": ["user", "video", "image", "audio"],
                        },
                        "page": {
                            "type": "integer",
                            "default": 1,
                            "minimum": 1,
                            "description": "Page number to retrieve when paginating through assets",
                        },
                        "items_per_page": {
                            "type": "integer",
                            "default": 10,
                            "minimum": 1,
                            "maximum": 50,
                            "description": "Number of items to show per page when paginating",
                        },
                        "asset_cache_id": {
                            "type": "string",
                            "description": "ID of a previous asset cache to continue pagination. If provided, returns the next chunk of results",
                        },
                    },
                    "required": ["project_id"],
                },
            ),
        ]
    return [
        types.Tool(
            name="create-videojungle-project",
            description="Create a new Video Jungle project to create video edits, add videos, assets, and more.",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Name of the project"},
                    "description": {
                        "type": "string",
                        "description": "Description of the project",
                    },
                },
            },
        ),
        types.Tool(
            name="edit-locally",
            description="Create an OpenTimelineIO file for local editing with the user's desktop video editing suite.",
            inputSchema={
                "type": "object",
                "properties": {
                    "edit_id": {
                        "type": "string",
                        "description": "UUID of the edit to download",
                    },
                    "project_id": {
                        "type": "string",
                        "description": "UUID of the project the video edit lives within",
                    },
                },
                "required": ["edit_id", "project_id"],
            },
        ),
        types.Tool(
            name="add-video",
            description="Upload video from URL. Begins analysis of video to allow for later information retrieval for automatic video editing an search.",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "url": {"type": "string"},
                },
                "required": ["name", "url"],
            },
        ),
        types.Tool(
            name="search-remote-videos",
            description="Default method to search videos. Will return videos including video_ids, which allow for information retrieval and building video edits. For large result sets, you can paginate through chunks using search_id and page parameters.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Text search query"},
                    "limit": {
                        "type": "integer",
                        "default": 50,
                        "minimum": 1,
                        "maximum": 100,
                        "description": "Maximum number of results to return per page",
                    },
                    "project_id": {
                        "type": "string",
                        "format": "uuid",
                        "description": "Project ID to scope the search",
                    },
                    "duration_min": {
                        "type": "number",
                        "minimum": 0,
                        "description": "Minimum video duration in seconds",
                    },
                    "duration_max": {
                        "type": "number",
                        "minimum": 0,
                        "description": "Maximum video duration in seconds",
                    },
                    "search_id": {
                        "type": "string",
                        "description": "ID of a previous search to continue pagination. If provided, returns the next chunk of results",
                    },
                    "page": {
                        "type": "integer",
                        "default": 1,
                        "minimum": 1,
                        "description": "Page number to retrieve when paginating through results",
                    },
                    "items_per_page": {
                        "type": "integer",
                        "default": 20,
                        "minimum": 1,
                        "maximum": 50,
                        "description": "Number of items to show per page when paginating",
                    },
                    "created_after": {
                        "type": "string",
                        "format": "date-time",
                        "description": "Filter videos created after this datetime",
                    },
                    "created_before": {
                        "type": "string",
                        "format": "date-time",
                        "description": "Filter videos created before this datetime",
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Set of tags to filter by",
                    },
                    "include_segments": {
                        "type": "boolean",
                        "default": True,
                        "description": "Whether to include video segments in results",
                    },
                    "include_related": {
                        "type": "boolean",
                        "default": False,
                        "description": "Whether to include related videos",
                    },
                    "query_audio": {
                        "type": "string",
                        "description": "Audio search query",
                    },
                    "query_img": {
                        "type": "string",
                        "description": "Image search query",
                    },
                },
            },
        ),
        types.Tool(
            name="generate-edit-from-videos",
            description="Generate an edit from videos, from within a specific project. Creates a new project to work within no existing project ID (UUID) is passed ",
            inputSchema={
                "type": "object",
                "properties": {
                    "project_id": {
                        "type": "string",
                        "description": "Either an existing Project UUID or String. A UUID puts the edit in an existing project, and a string creates a new project with that name.",
                    },
                    "name": {"type": "string", "description": "Video Edit name"},
                    "open_editor": {
                        "type": "boolean",
                        "description": "Open a live editor with the project's edit",
                    },
                    "resolution": {
                        "type": "string",
                        "description": "Video resolution. Examples include '1920x1080', '1280x720'",
                    },
                    "vertical_crop": {
                        "type": "string",
                        "description": "ML-powered automatic vertical crop mode. Pass 'standard' to enable automatic vertical video cropping",
                    },
                    "edit": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "video_id": {
                                    "type": "string",
                                    "description": "Video UUID",
                                },
                                "video_start_time": {
                                    "type": "string",
                                    "description": "Clip start time in 00:00:00.000 format",
                                },
                                "video_end_time": {
                                    "type": "string",
                                    "description": "Clip end time in 00:00:00.000 format",
                                },
                                "type": {
                                    "type": "string",
                                    "description": "Type of asset ('videofile' for video files, or 'user' for project specific assets)",
                                },
                                "audio_levels": {
                                    "type": "array",
                                    "description": "Optional audio level adjustments for this clip",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "audio_level": {
                                                "type": "string",
                                                "description": "Audio level (0.0 to 1.0)",
                                            }
                                        },
                                    },
                                },
                                "crop": {
                                    "type": "object",
                                    "description": "Optional crop/zoom settings for this video segment",
                                    "properties": {
                                        "zoom": {
                                            "type": "number",
                                            "minimum": 0.1,
                                            "maximum": 10.0,
                                            "default": 1.0,
                                            "description": "Zoom factor (1.0 = 100%, 1.5 = 150%, etc.)",
                                        },
                                        "position_x": {
                                            "type": "number",
                                            "minimum": -1.0,
                                            "maximum": 1.0,
                                            "default": 0.0,
                                            "description": "Horizontal offset from center (-1.0 to 1.0)",
                                        },
                                        "position_y": {
                                            "type": "number",
                                            "minimum": -1.0,
                                            "maximum": 1.0,
                                            "default": 0.0,
                                            "description": "Vertical offset from center (-1.0 to 1.0)",
                                        },
                                    },
                                },
                            },
                        },
                        "description": "Array of video clips to include in the edit",
                    },
                    "audio_asset": {
                        "type": "object",
                        "properties": {
                            "audio_id": {
                                "type": "string",
                                "description": "Audio asset UUID",
                            },
                            "type": {
                                "type": "string",
                                "description": "Audio file type (e.g., 'mp3', 'wav')",
                            },
                            "filename": {
                                "type": "string",
                                "description": "Audio file name",
                            },
                            "audio_start_time": {
                                "type": "string",
                                "description": "Audio start time in 00:00:00.000 format",
                            },
                            "audio_end_time": {
                                "type": "string",
                                "description": "Audio end time in 00:00:00.000 format",
                            },
                            "url": {
                                "type": "string",
                                "description": "Optional URL for the audio file",
                            },
                            "audio_levels": {
                                "type": "array",
                                "description": "Optional audio level adjustments",
                                "items": {"type": "object"},
                            },
                        },
                        "description": "Optional audio overlay for the video edit",
                    },
                },
                "required": ["edit", "name", "project_id"],
            },
        ),
        types.Tool(
            name="generate-edit-from-single-video",
            description="Generate a compressed video edit from a single video.",
            inputSchema={
                "type": "object",
                "properties": {
                    "project_id": {"type": "string"},
                    "resolution": {"type": "string"},
                    "video_id": {"type": "string"},
                    "vertical_crop": {
                        "type": "string",
                        "description": "ML-powered automatic vertical crop mode. Pass 'standard' to enable automatic vertical video cropping",
                    },
                    "edit": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "video_start_time": {
                                    "type": "string",
                                    "description": "Clip start time in 00:00:00.000 format",
                                },
                                "video_end_time": {
                                    "type": "string",
                                    "description": "Clip end time in 00:00:00.000 format",
                                },
                            },
                        },
                        "description": "Array of time segments to extract from the video",
                    },
                },
                "required": ["edit", "project_id", "video_id"],
            },
        ),
        types.Tool(
            name="update-video-edit",
            description="Update an existing video edit within a specific project.",
            inputSchema={
                "type": "object",
                "properties": {
                    "project_id": {
                        "type": "string",
                        "description": "UUID of the project containing the edit",
                    },
                    "edit_id": {
                        "type": "string",
                        "description": "UUID of the video edit to update",
                    },
                    "name": {"type": "string", "description": "Video Edit name"},
                    "description": {
                        "type": "string",
                        "description": "Description of the video edit",
                    },
                    "video_output_format": {
                        "type": "string",
                        "description": "Output format for the video (e.g., 'mp4', 'webm')",
                    },
                    "video_output_resolution": {
                        "type": "string",
                        "description": "Video resolution. Examples include '1920x1080', '1280x720'",
                    },
                    "video_output_fps": {
                        "type": "number",
                        "description": "Frames per second for the output video",
                    },
                    "video_series_sequential": {
                        "type": "array",
                        "description": "Array of video clips in sequential order",
                        "items": {
                            "type": "object",
                            "properties": {
                                "video_id": {
                                    "type": "string",
                                    "description": "Video UUID",
                                },
                                "video_start_time": {
                                    "type": "string",
                                    "description": "Clip start time in 00:00:00.000 format",
                                },
                                "video_end_time": {
                                    "type": "string",
                                    "description": "Clip end time in 00:00:00.000 format",
                                },
                                "type": {
                                    "type": "string",
                                    "description": "Type of asset ('videofile' for video files, or 'user' for project specific assets)",
                                },
                                "audio_levels": {
                                    "type": "array",
                                    "description": "Optional audio level adjustments for this clip",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "audio_level": {
                                                "type": "string",
                                                "description": "Audio level (0.0 to 1.0)",
                                            }
                                        },
                                    },
                                },
                                "crop": {
                                    "type": "object",
                                    "description": "Optional crop/zoom settings for this video segment",
                                    "properties": {
                                        "zoom": {
                                            "type": "number",
                                            "minimum": 0.1,
                                            "maximum": 10.0,
                                            "default": 1.0,
                                            "description": "Zoom factor (1.0 = 100%, 1.5 = 150%, etc.)",
                                        },
                                        "position_x": {
                                            "type": "number",
                                            "minimum": -1.0,
                                            "maximum": 1.0,
                                            "default": 0.0,
                                            "description": "Horizontal offset from center (-1.0 to 1.0)",
                                        },
                                        "position_y": {
                                            "type": "number",
                                            "minimum": -1.0,
                                            "maximum": 1.0,
                                            "default": 0.0,
                                            "description": "Vertical offset from center (-1.0 to 1.0)",
                                        },
                                    },
                                },
                            },
                        },
                    },
                    "audio_overlay": {
                        "type": "object",
                        "description": "Audio overlay settings and assets",
                    },
                    "rendered": {
                        "type": "boolean",
                        "description": "Whether the edit has been rendered",
                    },
                    "vertical_crop": {
                        "type": "string",
                        "description": "ML-powered automatic vertical crop mode. Pass 'standard' to enable automatic vertical video cropping",
                    },
                },
                "required": ["project_id", "edit_id"],
            },
        ),
        types.Tool(
            name="create-video-bar-chart-from-two-axis-data",
            description="Create a video bar chart from two-axis data",
            inputSchema={
                "type": "object",
                "properties": {
                    "x_values": {"type": "array", "items": {"type": "string"}},
                    "y_values": {"type": "array", "items": {"type": "number"}},
                    "x_label": {"type": "string"},
                    "y_label": {"type": "string"},
                    "title": {"type": "string"},
                    "filename": {"type": "string"},
                },
                "required": ["x_values", "y_values", "x_label", "y_label", "title"],
            },
        ),
        types.Tool(
            name="create-video-line-chart-from-two-axis-data",
            description="Create a video line chart from two-axis data",
            inputSchema={
                "type": "object",
                "properties": {
                    "x_values": {"type": "array", "items": {"type": "string"}},
                    "y_values": {"type": "array", "items": {"type": "number"}},
                    "x_label": {"type": "string"},
                    "y_label": {"type": "string"},
                    "title": {"type": "string"},
                    "filename": {"type": "string"},
                },
                "required": ["x_values", "y_values", "x_label", "y_label", "title"],
            },
        ),
        types.Tool(
            name="get-project-assets",
            description="Get all assets and details for a specific project, with pagination support for large projects",
            inputSchema={
                "type": "object",
                "properties": {
                    "project_id": {
                        "type": "string",
                        "description": "UUID of the project to retrieve assets for",
                    },
                    "asset_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of asset types to filter by (e.g. 'user', 'video', 'image', 'audio', 'generated_video', 'generated_audio', 'video_edit')",
                        "default": [
                            "user",
                            "video",
                            "image",
                            "audio",
                            "generated_audio",
                        ],
                    },
                    "page": {
                        "type": "integer",
                        "default": 1,
                        "minimum": 1,
                        "description": "Page number to retrieve when paginating through assets",
                    },
                    "items_per_page": {
                        "type": "integer",
                        "default": 50,
                        "minimum": 1,
                        "maximum": 100,
                        "description": "Number of items to show per page when paginating",
                    },
                    "asset_cache_id": {
                        "type": "string",
                        "description": "ID of a previous asset cache to continue pagination. If provided, returns the next chunk of results",
                    },
                },
                "required": ["project_id"],
            },
        ),
    ]


def format_single_video(video):
    """
    Format a single video metadata tuple (metadata_dict, confidence_score)
    Returns a formatted string and a Python code string representation
    """
    try:
        # Create human-readable format
        readable_format = f"""
            Video Embedding Result:
            -------------
            Video ID: {video['video_id']}
            Description: {video['description']}
            Timestamp: {video['timepoint']}
            Detected Items: {', '.join(video['detected_items']) if video['detected_items'] else 'None'}
        """
    except Exception as e:
        raise ValueError(f"Error formatting video: {str(e)}")

    return readable_format


def filter_unique_videos_keep_first(json_results):
    seen = set()
    return [
        item
        for item in json_results
        if item["video_id"] not in seen and not seen.add(item["video_id"])
    ]


def format_video_info(video):
    try:
        if video.get("script") is not None:
            if len(video.get("script")) > 200:
                script = video.get("script")[:200] + "..."
            else:
                script = video.get("script")
        else:
            script = "N/A"
        segments = []
        for segment in video.get("matching_segments", []):
            segments.append(
                f"- Time: {segment.get('start_seconds', 'N/A')} to {segment.get('end_seconds', 'N/A')}"
            )
        joined_segments = "\n".join(segments)
        return (
            f"- Video Id: {video.get('video_id', 'N/A')}\n"
            f"  Video name: {video.get('video', {}).get('name', 'N/A')}\n"
            f"  URL to view video: {video.get('video', {}).get('url', 'N/A')}\n"
            f"  Video manuscript: {script}"
            f"  Matching scenes: {joined_segments}"
            f"  Generated description: {video.get('video', 'N/A').get('generated_description', 'N/A')}"
        )
    except Exception as e:
        return f"Error formatting video: {str(e)}"


def format_video_info_long(video):
    try:
        if video.get("script") is not None:
            if len(video.get("script")) > 200:
                script = video.get("script")[:200] + "..."
            else:
                script = video.get("script")
        else:
            script = "N/A"
        return (
            f"- Video Id: {video.get('video_id', 'N/A')}\n"
            f"  Video name: {video.get('video', {}).get('name', 'N/A')}\n"
            f"  URL to view video: {video.get('video', {}).get('url', 'N/A')}\n"
            f"  Generated description: {video.get('video', 'N/A').get('generated_description', 'N/A')}"
            f"  Video manuscript: {script}"
            f"  Matching times: {video.get('scene_changes', 'N/A')}"
        )
    except Exception as e:
        return f"Error formatting video: {str(e)}"


def format_asset_info(asset):
    """Format asset information for display based on the example structure you showed"""
    try:
        # Support both type and asset_type fields
        asset_type = asset.get("type", asset.get("asset_type", "unknown"))
        asset_id = asset.get("id", "N/A")
        # Support both name and keyname fields
        asset_name = asset.get("name", asset.get("keyname", "N/A"))

        # Common fields first
        formatted = [f"- Asset ID: {asset_id}"]
        formatted.append(f"  Type: {asset_type}")
        formatted.append(f"  Name: {asset_name}")

        # Get URL (try different possible fields)
        url = asset.get("url", "N/A")
        download_url = asset.get("download_url", "N/A")

        if url and url != "N/A":
            # Truncate very long URLs
            if len(url) > 80:
                formatted.append(f"  URL: {url[:77]}...")
            else:
                formatted.append(f"  URL: {url}")

        if download_url and download_url != "N/A" and download_url != url:
            if len(download_url) > 80:
                formatted.append(f"  Download URL: {download_url[:77]}...")
            else:
                formatted.append(f"  Download URL: {download_url}")

        # Description
        description = asset.get("description", "N/A")
        if description and description != "N/A":
            if len(description) > 120:
                formatted.append(f"  Description: {description[:117]}...")
            else:
                formatted.append(f"  Description: {description}")

        # Creation time
        created_at = asset.get("created_at", "N/A")
        if created_at and created_at != "N/A":
            formatted.append(f"  Created: {created_at}")

        # Handle video assets and user-uploaded content
        if asset_type in ["user", "video"]:
            # Look for generated description (from your example)
            gen_desc = asset.get("generated_description", "N/A")
            if gen_desc and gen_desc != "N/A":
                formatted.append(f"  Generated description: {gen_desc}")

            # Check for create_parameters.analysis (from your example)
            create_params = asset.get("create_parameters", {})
            if create_params and isinstance(create_params, dict):
                analysis = create_params.get("analysis", {})
                if analysis and isinstance(analysis, dict):
                    formatted.append(f" analysis: {str(analysis)}")

            # Status field (if available)
            status = asset.get("status", "N/A")
            if status and status != "N/A":
                formatted.append(f"  Status: {status}")

            # Asset path field (if available)
            asset_path = asset.get("asset_path", "N/A")
            if asset_path and asset_path != "N/A":
                formatted.append(f"  Asset path: {asset_path}")

        # Handle video_edit assets
        elif asset_type == "video_edit":
            description = asset.get("description", "N/A")
            if description and description != "N/A":
                formatted.append(f"  Description: {description}")

            # Add edit-specific details
            resolution = asset.get("video_output_resolution", "N/A")
            fps = asset.get("video_output_fps", "N/A")
            format = asset.get("video_output_format", "N/A")

            if resolution != "N/A":
                formatted.append(f"  Resolution: {resolution}")
            if fps != "N/A":
                formatted.append(f"  FPS: {fps}")
            if format != "N/A":
                formatted.append(f"  Format: {format}")

            # Show clips in the edit
            clips = asset.get("video_series_sequential", [])
            if clips and len(clips) > 0:
                formatted.append(f"  Clips: {len(clips)} total")
                # Show first 3 clips as examples
                for i, clip in enumerate(clips[:3]):
                    clip_id = clip.get("video_id", "N/A")
                    start = clip.get("video_start_time", "N/A")
                    end = clip.get("video_end_time", "N/A")
                    asset_type = clip.get("type", "N/A")
                    formatted.append(
                        f"    Clip {i+1}: {clip_id} of type {asset_type} from {start} to {end}"
                    )
                if len(clips) > 3:
                    formatted.append(f"    ... and {len(clips)-3} more clips")

        # Add any other important fields we might have missed
        important_fields = ["filetype", "duration", "width", "height", "uploaded"]
        for field in important_fields:
            if field in asset and asset[field] is not None and asset[field] != "N/A":
                formatted.append(f"  {field}: {asset[field]}")

        return "\n".join(formatted)
    except Exception as e:
        return f"Error formatting asset {asset.get('id', 'unknown')}: {str(e)}"


@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict | None
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """
    Handle tool execution requests.
    Tools can modify server state and notify clients of changes.
    """
    if name not in tools:
        raise ValueError(f"Unknown tool: {name}")

    if not arguments:
        raise ValueError("Missing arguments")

    # Store some tool results in server state for pagination
    global _search_result_cache

    if name == "create-videojungle-project" and arguments:
        namez = arguments.get("name")
        description = arguments.get("description")

        if not namez or not description:
            raise ValueError("Missing project name")

        # Create a new project
        project = vj.projects.create(name=namez, description=description)

        # Notify clients that resources have changed
        await server.request_context.session.send_resource_list_changed()

        return [
            types.TextContent(
                type="text",
                text=f"Created new project '{project.name}' with id '{project.id}'",
            )
        ]

    if name == "edit-locally" and arguments:
        project_id = arguments.get("project_id")
        edit_id = arguments.get("edit_id")

        if not project_id or not edit_id:
            raise ValueError("Missing edit and / or  project id")
        env_vars = {"VJ_API_KEY": VJ_API_KEY, "PATH": os.environ["PATH"]}
        edit_data = vj.projects.get_edit(project_id, edit_id)
        formatted_name = edit_data["name"].replace(" ", "-")
        with open(f"{formatted_name}.json", "w") as f:
            json.dump(edit_data, f, indent=4)
        logging.info(f"edit data is: {edit_data}")
        logging.info(f"current directory is: {os.getcwd()}")
        subprocess.Popen(
            [
                "uv",
                "run",
                "python",
                "./src/video_editor_mcp/generate_opentimeline.py",
                "--file",
                f"{formatted_name}.json",
                "--output",
                f"{formatted_name}.otio",
            ],
            env=env_vars,
        )

        return [
            types.TextContent(
                type="text",
                text=f"Edit {edit_data['name']} is being downloaded and converted to OpenTimelineIO format. You can find it in the current directory.",
            )
        ]

    if name == "add-video" and arguments:
        name = arguments.get("name")  # type: ignore
        url = arguments.get("url")

        if not name or not url:
            raise ValueError("Missing name or content")

        # Update server state
        vj.video_files.create(name=name, filename=str(url), upload_method="url")

        # Notify clients that resources have changed
        await server.request_context.session.send_resource_list_changed()
        return [
            types.TextContent(
                type="text",
                text=f"Added video '{name}' with url: {url}",
            )
        ]
    if name == "search-remote-videos" and arguments:
        # Check if this is a pagination request
        search_id = arguments.get("search_id")
        page = arguments.get("page", 1)
        items_per_page = arguments.get("items_per_page", 5)

        # Run cache cleanup
        cleanup_cache()

        # If we have a search_id, we're doing pagination
        if search_id and search_id in _search_result_cache:
            cache_entry = _search_result_cache[search_id]
            cached_results = cache_entry["results"]
            total_items = len(cached_results)
            total_pages = (total_items + items_per_page - 1) // items_per_page

            # Update timestamp on access
            _search_result_cache[search_id]["timestamp"] = time.time()

            start_idx = (page - 1) * items_per_page
            end_idx = min(start_idx + items_per_page, total_items)

            # Get current page items
            current_page_items = cached_results[start_idx:end_idx]

            # Format the paginated results
            query_info = cache_entry.get("query", "unknown")
            response_text = []
            response_text.append(
                f"Search Results for '{query_info}' (Page {page}/{total_pages}, showing items {start_idx+1}-{end_idx} of {total_items})"
            )

            # Add embedding note if it exists in the cache
            embedding_note = cache_entry.get("embedding_note")
            if embedding_note:
                response_text.append(embedding_note)

            # Format each item based on whether it's a regular result or an embedding result
            if len(current_page_items) > 0:
                if (
                    isinstance(current_page_items[0], dict)
                    and "video_id" in current_page_items[0]
                ):
                    response_text.extend(
                        format_video_info(video) for video in current_page_items
                    )
                else:
                    response_text.extend(current_page_items)
            else:
                response_text.append("No items to display on this page.")

            # Add pagination info with navigation options
            pagination_info = []
            if page > 1:
                pagination_info.append(
                    f"Previous page: call search-remote-videos with search_id='{search_id}' and page={page-1}"
                )

            has_more = page < total_pages
            if has_more:
                pagination_info.append(
                    f"Next page: call search-remote-videos with search_id='{search_id}' and page={page+1}"
                )

            if pagination_info:
                response_text.append("\nNavigation options:")
                response_text.extend(pagination_info)

            if not has_more:
                response_text.append("\nEnd of results.")

            return [
                types.TextContent(
                    type="text",
                    text="\n".join(response_text),
                )
            ]

        # This is a new search request
        logging.info(f"search-remote-videos received arguments: {arguments}")
        query = arguments.get("query")
        limit = arguments.get("limit", 10)
        project_id = arguments.get("project_id")
        tags = arguments.get("tags", None)
        duration_min = arguments.get("duration_min", None)
        duration_max = arguments.get("duration_max", None)
        created_after = arguments.get("created_after", None)
        created_before = arguments.get("created_before", None)
        include_segments = arguments.get("include_segments", True)
        include_related = arguments.get("include_related", False)

        # Validate that at least one query type is provided
        if not query and not tags:
            raise ValueError("At least one query or tag must be provided")

        # Perform the main search with all parameters
        if tags:
            search_params = {
                "limit": limit,
                "include_segments": include_segments,
                "include_related": include_related,
                "tags": json.loads(tags),
                "duration_min": duration_min,
                "duration_max": duration_max,
                "created_after": created_after,
                "created_before": created_before,
            }
        else:
            search_params = {
                "limit": limit,
                "include_segments": include_segments,
                "include_related": include_related,
                "duration_min": duration_min,
                "duration_max": duration_max,
                "created_after": created_after,
                "created_before": created_before,
            }

        # Add optional parameters
        if query:
            search_params["query"] = query
        if project_id:
            # Convert UUID to string if it's not already a string
            search_params["project_id"] = str(project_id)

        embedding_results = []
        embedding_search_formatted = []
        embedding_note = None

        # If we have a text query, try embedding search but fallback to regular search if model is still loading
        if query:
            try:
                embeddings = model_loader.encode_text(query)
                response = model_loader.post_embeddings(
                    embeddings,
                    "https://api.video-jungle.com/video-file/embedding-search",
                    headers={
                        "Content-Type": "application/json",
                        "X-API-KEY": VJ_API_KEY,
                    },
                )

                logging.info(f"Response is: {response.json()}")
                if response.status_code != 200:
                    raise RuntimeError(f"Error searching for videos: {response.text}")

                embedding_results = response.json()
                embedding_search_formatted = [
                    format_single_video(video) for video in embedding_results
                ]
            except Exception as e:
                if "still loading" in str(e):
                    logging.warning(
                        "Embedding model still loading, falling back to text-only search"
                    )
                    embedding_results = []
                    embedding_search_formatted = []
                    # Add note that will be displayed to the user
                    embedding_note = "Note: Embedding-based semantic search is still initializing. Only text-based search results are shown. Please try again later for more accurate semantic search results."
                else:
                    # For other errors, log and continue with regular search
                    logging.error(f"Error in embedding search: {e}")
                    embedding_results = []
                    embedding_search_formatted = []

        # Get regular search results
        logging.info(
            f"Search params being passed to vj.video_files.search: {search_params}"
        )
        logging.info(f"VJ client: {vj}, API key present: {bool(VJ_API_KEY)}")
        try:
            videos = vj.video_files.search(**search_params)
            logging.info(f"Search returned {len(videos)} videos")
            if videos:
                logging.info(f"First video: {videos[0]}")
        except Exception as e:
            logging.error(f"Error in vj.video_files.search: {e}")
            videos = []
        logging.info(f"num videos are: {len(videos)}")

        # If no results found, return a helpful message
        if len(videos) == 0 and not embedding_results:
            return [
                types.TextContent(
                    type="text",
                    text=f"No videos found matching query '{query}' with the specified filters. Try broadening your search criteria.",
                )
            ]

        # If only a few results, return them directly without pagination
        if len(videos) <= 3 and len(videos) >= 1 and not embedding_results:
            return [
                types.TextContent(
                    type="text",
                    text=format_video_info_long(video),
                )
                for video in videos
            ]

        # For larger result sets, set up pagination
        formatted_videos = [format_video_info(video) for video in videos]

        # Store the results in the cache for pagination
        new_search_id = str(uuid.uuid4())

        all_results = []
        if query and embedding_results:
            # Store both types of results
            all_results = formatted_videos + embedding_search_formatted
        else:
            all_results = formatted_videos

        # Store results with timestamp and embedding note if present
        _search_result_cache[new_search_id] = {
            "results": all_results,
            "timestamp": time.time(),
            "query": query or "tag-search",
            "embedding_note": embedding_note,
        }

        # Calculate pagination info
        total_items = len(all_results)
        total_pages = (total_items + items_per_page - 1) // items_per_page

        # Format the first page results
        response_text = []
        query_display = query or "tag search"
        response_text.append(
            f"Search Results for '{query_display}' (Page 1/{total_pages}, showing items 1-{min(items_per_page, total_items)} of {total_items})"
        )

        # Add note about embedding search if it was skipped due to model loading
        if embedding_note:
            response_text.append(embedding_note)

        # Show first page items
        first_page_items = all_results[:items_per_page]
        if first_page_items:
            response_text.extend(first_page_items)
        else:
            response_text.append("No results found matching your query.")

        # Add pagination info
        has_more = total_pages > 1
        if has_more:
            response_text.append("\nNavigation options:")
            response_text.append(
                f"Next page: call search-remote-videos with search_id='{new_search_id}' and page=2"
            )
            response_text.append(
                "\nTip: You can control items per page with the items_per_page parameter (default: 5, max: 20)"
            )
        else:
            response_text.append("\nEnd of results.")

        return [
            types.TextContent(
                type="text",
                text="\n".join(response_text),
            )
        ]

    if name == "search-local-videos" and arguments:
        if not os.environ.get("LOAD_PHOTOS_DB"):
            raise ValueError(
                "You must set the LOAD_PHOTOS_DB environment variable to True to use this tool"
            )

        keyword = arguments.get("keyword")
        if not keyword:
            raise ValueError("Missing keyword")
        start_date = None
        end_date = None

        if arguments.get("start_date") and arguments.get("end_date"):
            start_date = arguments.get("start_date")
            end_date = arguments.get("end_date")

        try:
            db = photos_loader.db
            videos = get_videos_by_keyword(db, keyword, start_date, end_date)
            return [
                types.TextContent(
                    type="text",
                    text=(
                        f"Number of Videos Returned: {len(videos)}. Here are the first 100 results: \n{videos[:100]}"
                    ),
                )
            ]
        except Exception:
            raise RuntimeError("Local Photos database not yet initialized")

    if name == "generate-edit-from-videos" and arguments:
        edit = arguments.get("edit")
        project = arguments.get("project_id")
        name = arguments.get("name")  # type: ignore
        open_editor = arguments.get("open_editor")
        resolution = arguments.get("resolution")
        audio_asset = arguments.get("audio_asset")
        vertical_crop = arguments.get("vertical_crop")
        subtitles = arguments.get("subtitles", True)
        created = False

        logging.info(f"edit is: {edit} and the type is: {type(edit)}")
        if open_editor is None:
            open_editor = True

        if not edit:
            raise ValueError("Missing edit")
        if not project:
            raise ValueError("Missing project")
        if not resolution:
            resolution = "1080x1920"
        if not name:
            raise ValueError("Missing name for edit")
        if resolution == "1080p":
            resolution = "1920x1080"
        elif resolution == "720p":
            resolution = "1280x720"

        try:
            w, h = resolution.split("x")
            _ = f"{int(w)}x{int(h)}"
        except Exception as e:
            raise ValueError(
                f"Resolution must be in the format 'widthxheight' where width and height are integers: {e}"
            )

        updated_edit = []
        for cut in edit:
            # Get the audio level for this clip (default to 0.5)
            audio_level_value = "0.5"
            if "audio_levels" in cut and len(cut["audio_levels"]) > 0:
                audio_level_value = cut["audio_levels"][0].get("audio_level", "0.5")

            clip_data = {
                "video_id": cut["video_id"],
                "video_start_time": cut["video_start_time"],
                "video_end_time": cut["video_end_time"],
                "type": cut["type"],
                "audio_levels": [
                    {
                        "audio_level": audio_level_value,
                        "start_time": cut["video_start_time"],
                        "end_time": cut["video_end_time"],
                    }
                ],
            }

            # Add crop settings if provided
            if "crop" in cut and cut["crop"]:
                clip_data["crop"] = cut["crop"]

            updated_edit.append(clip_data)

        logging.info(f"updated edit is: {updated_edit}")

        # Process audio asset if provided
        audio_overlay = []
        if audio_asset:
            audio_overlay_item = {
                "audio_id": audio_asset.get("audio_id", ""),
                "type": audio_asset.get("type", "mp3"),
                "filename": audio_asset.get("filename", ""),
                "audio_start_time": audio_asset.get("audio_start_time", "00:00:00.000"),
                "audio_end_time": audio_asset.get("audio_end_time", "00:00:00.000"),
                "url": audio_asset.get("url", ""),
                "audio_levels": audio_asset.get("audio_levels", []),
            }
            audio_overlay.append(audio_overlay_item)
            logging.info(f"Audio overlay configured: {audio_overlay_item}")
        else:
            subtitles = False
        json_edit = {
            "video_edit_version": "1.0",
            "video_output_format": "mp4",
            "video_output_resolution": resolution,
            "video_output_fps": 60.0,
            "name": name,
            "video_output_filename": "output_video.mp4",
            "audio_overlay": audio_overlay,
            "video_series_sequential": updated_edit,
            "skip_rendering": True,
            "subtitle_from_audio_overlay": subtitles,
        }

        # Add vertical_crop if provided
        if vertical_crop:
            json_edit["vertical_crop"] = vertical_crop

        try:
            proj = vj.projects.get(project)
        except Exception as e:
            logging.info(f"project not found, creating new project because {e}")
            proj = vj.projects.create(
                name=project, description="Claude generated project"
            )
            project = proj.id
            created = True

        logging.info(f"video edit is: {json_edit}")

        edit = vj.projects.render_edit(project, json_edit)

        webbrowser.open(
            f"https://app.video-jungle.com/projects/{proj.id}/edits/{edit['edit_id']}"
        )
        global BROWSER_OPEN
        BROWSER_OPEN = True
        if created:
            # we created a new project so let the user / LLM know
            return [
                types.TextContent(
                    type="text",
                    text=f"Created new project {proj.name} with id '{proj.id}' with the new edit id: {edit['edit_id']} viewable at this url: https://app.video-jungle.com/projects/{proj.id}/edits/{edit['edit_id']}",
                )
            ]

        return [
            types.TextContent(
                type="text",
                text=f"Generated edit in existing project {proj.name} with id '{proj.id}' with the new edit id: {edit['edit_id']} viewable at this url: https://app.video-jungle.com/projects/{proj.id}/edits/{edit['edit_id']}",
            )
        ]

    if name == "generate-edit-from-single-video" and arguments:
        edit = arguments.get("edit")
        project = arguments.get("project_id")
        video_id = arguments.get("video_id")

        resolution = arguments.get("resolution")
        vertical_crop = arguments.get("vertical_crop")
        created = False

        logging.info(f"edit is: {edit} and the type is: {type(edit)}")

        if not edit:
            raise ValueError("Missing edit")
        if not project:
            raise ValueError("Missing project")
        if not video_id:
            raise ValueError("Missing video_id")
        if not resolution:
            resolution = "1080x1920"

        try:
            w, h = resolution.split("x")
            _ = f"{int(w)}x{int(h)}"
        except Exception as e:
            raise ValueError(
                f"Resolution must be in the format 'widthxheight' where width and height are integers: {e}"
            )

        try:
            updated_edit = [
                {
                    "video_id": video_id,
                    "video_start_time": cut["video_start_time"],
                    "video_end_time": cut["video_end_time"],
                    "type": "video-file",
                    "audio_levels": [],
                }
                for cut in edit
            ]
        except Exception as e:
            raise ValueError(f"Error updating edit: {e}")

        logging.info(f"updated edit is: {updated_edit}")

        json_edit = {
            "video_edit_version": "1.0",
            "video_output_format": "mp4",
            "video_output_resolution": resolution,
            "video_output_fps": 60.0,
            "video_output_filename": "output_video.mp4",
            "audio_overlay": [],  # TODO: add this back in
            "video_series_sequential": updated_edit,
        }

        # Add vertical_crop if provided
        if vertical_crop:
            json_edit["vertical_crop"] = vertical_crop

        try:
            proj = vj.projects.get(project)
        except Exception:
            proj = vj.projects.create(
                name=project, description="Claude generated project"
            )
            project = proj.id
            created = True

        logging.info(f"video edit is: {json_edit}")
        try:
            edit = vj.projects.render_edit(project, json_edit)
        except Exception as e:
            logging.error(f"Error rendering edit: {e}")
        logging.info(f"edit is: {edit}")
        if created:
            # we created a new project so let the user / LLM know
            logging.info(f"created new project {proj.name} and created edit {edit}")
            return [
                types.TextContent(
                    type="text",
                    text=f"Created new project {proj.name} with project id '{proj.id}' viewable at this url: https://app.video-jungle.com/projects/{proj.id}/edits/{edit['edit_id']}",
                )
            ]

        return [
            types.TextContent(
                type="text",
                text=f"Generated edit with id '{edit['edit_id']}' in project {proj.name} with project id '{proj.id}' viewable at this url: https://app.video-jungle.com/projects/{proj.id}/edits/{edit['edit_id']}",
            )
        ]

    if name == "update-video-edit" and arguments:
        project_id = arguments.get("project_id")
        edit_id = arguments.get("edit_id")
        edit_name = arguments.get("name")
        description = arguments.get("description")
        video_output_format = arguments.get("video_output_format")
        video_output_resolution = arguments.get("video_output_resolution")
        video_output_fps = arguments.get("video_output_fps")
        video_series_sequential = arguments.get("video_series_sequential")
        audio_overlay = arguments.get("audio_overlay")
        rendered = arguments.get("rendered")
        vertical_crop = arguments.get("vertical_crop")

        # Validate required parameters
        if not project_id:
            raise ValueError("Missing project_id")
        if not edit_id:
            raise ValueError("Missing edit_id")

        # Process resolution format like in create function
        if video_output_resolution:
            if video_output_resolution == "1080p":
                video_output_resolution = "1920x1080"
            elif video_output_resolution == "720p":
                video_output_resolution = "1280x720"

            # Validate resolution format
            try:
                w, h = video_output_resolution.split("x")
                _ = f"{int(w)}x{int(h)}"
            except Exception as e:
                raise ValueError(
                    f"Resolution must be in the format 'widthxheight' where width and height are integers: {e}"
                )

        # Try to get the existing project
        try:
            proj = vj.projects.get(project_id)
        except Exception as e:
            raise ValueError(f"Project with ID {project_id} not found: {e}")

        # Try to get the existing edit
        try:
            existing_edit = vj.projects.get_edit(project_id, edit_id)
        except Exception as e:
            raise ValueError(
                f"Edit with ID {edit_id} not found in project {project_id}: {e}"
            )

        # Process video clips if provided
        updated_video_series = None
        if video_series_sequential:
            updated_video_series = []
            for clip in video_series_sequential:
                # Get the audio level for this clip (default to 0.5)
                audio_level_value = "0.5"
                if "audio_levels" in clip and len(clip["audio_levels"]) > 0:
                    audio_level_value = clip["audio_levels"][0].get(
                        "audio_level", "0.5"
                    )

                clip_data = {
                    "video_id": clip["video_id"],
                    "video_start_time": clip["video_start_time"],
                    "video_end_time": clip["video_end_time"],
                    "type": clip["type"],
                    "audio_levels": [
                        {
                            "audio_level": audio_level_value,
                            "start_time": clip["video_start_time"],
                            "end_time": clip["video_end_time"],
                        }
                    ],
                }

                # Add crop settings if provided
                if "crop" in clip and clip["crop"]:
                    clip_data["crop"] = clip["crop"]

                updated_video_series.append(clip_data)

        # Create an empty dictionary without type annotations
        update_json = dict()

        # Add fields one by one with explicit type handling
        update_json["video_edit_version"] = "1.0"

        if edit_name:
            update_json["name"] = edit_name
        if description:
            update_json["description"] = description
        if video_output_format:
            update_json["video_output_format"] = video_output_format
        if video_output_resolution:
            update_json["video_output_resolution"] = video_output_resolution
        if video_output_fps is not None:
            update_json["video_output_fps"] = float(video_output_fps)
        if updated_video_series is not None:
            # Cast to a list to ensure proper typing
            update_json["video_series_sequential"] = list(updated_video_series)
        if audio_overlay is not None:
            # Cast to a list to ensure proper typing
            update_json["audio_overlay"] = list(audio_overlay) if audio_overlay else []

        # Skip rendering by default like in create function
        update_json["skip_rendering"] = bool(True)

        # If rendering is explicitly requested
        if rendered is True:
            update_json["skip_rendering"] = bool(False)

        # Add vertical_crop if provided
        if vertical_crop:
            update_json["vertical_crop"] = vertical_crop

        logging.info(f"Updating edit {edit_id} with: {update_json}")

        # Call the API to update the edit
        updated_edit = vj.projects.update_edit(project_id, edit_id, update_json)

        # Optionally open the browser to the updated edit
        if not BROWSER_OPEN:
            webbrowser.open(
                f"https://app.video-jungle.com/projects/{project_id}/edits/{edit_id}"
            )

        return [
            types.TextContent(
                type="text",
                text=f"Updated edit {edit_id} in project {proj.name} at url https://app.video-jungle.com/projects/{project_id}/edits/{edit_id} with changes: {update_json}",
            )
        ]

    if name == "get-project-assets" and arguments:
        # Extract arguments
        project_id = arguments.get("project_id")
        page = arguments.get("page", 1)
        items_per_page = arguments.get("items_per_page", 10)
        asset_cache_id = arguments.get("asset_cache_id")
        asset_types = arguments.get("asset_types", ["user", "video", "image", "audio"])

        # Validate required arguments
        if not project_id:
            raise ValueError("Missing project_id parameter")

        # Run cache cleanup
        cleanup_cache()

        # Check if this is a pagination request using an existing cache
        if asset_cache_id and asset_cache_id in _project_assets_cache:
            cache_entry = _project_assets_cache[asset_cache_id]
            cached_assets = cache_entry["assets"]
            project_info = cache_entry.get("project_info", {})

            # Update timestamp on access
            _project_assets_cache[asset_cache_id]["timestamp"] = time.time()

            # Calculate pagination
            total_items = len(cached_assets)
            total_pages = (total_items + items_per_page - 1) // items_per_page

            # Calculate current page items
            start_idx = (page - 1) * items_per_page
            end_idx = min(start_idx + items_per_page, total_items)
            current_page_items = cached_assets[start_idx:end_idx]

            # Format the response
            response_text = []

            # Add project info header
            project_name = project_info.get("name", "Project")
            project_description = project_info.get("description", "")

            response_text.append(f"Project: {project_name}")
            if project_description:
                response_text.append(f"Description: {project_description}")

            response_text.append(
                f"\nAssets (Page {page}/{total_pages}, showing items {start_idx+1}-{end_idx} of {total_items}):"
            )

            # Format each asset
            if current_page_items:
                formatted_assets = [
                    format_asset_info(asset) for asset in current_page_items
                ]
                response_text.extend(formatted_assets)
            else:
                response_text.append("No assets to display on this page.")

            # Add pagination info
            navigation_options = []
            if page > 1:
                navigation_options.append(
                    f"Previous page: call get-project-assets with asset_cache_id='{asset_cache_id}' and page={page-1}"
                )

            has_more = page < total_pages
            if has_more:
                navigation_options.append(
                    f"Next page: call get-project-assets with asset_cache_id='{asset_cache_id}' and page={page+1}"
                )

            if navigation_options:
                response_text.append("\nNavigation options:")
                response_text.extend(navigation_options)

            if not has_more:
                response_text.append("\nEnd of results.")

            return [types.TextContent(type="text", text="\n".join(response_text))]

        # This is a new request - get the project and its assets
        try:
            # Fetch project data
            project = vj.projects.get(project_id)
            logging.info(f"Retrieved project: {project.name} (ID: {project_id})")

            # Get project data as a dictionary so we can extract assets
            project_data = project.model_dump()
            logging.info(f"Project data: {project_data}")

            # Direct assignment - based on the data structure you showed
            all_assets = project_data.get("assets", [])
            logging.info(f"Found {len(all_assets)} assets in project")

            # Filter assets by asset_type if specified
            project_assets = []
            for asset in all_assets:
                if not asset_types or asset.get("asset_type") in asset_types:
                    project_assets.append(asset)

            logging.info(
                f"After filtering by types {asset_types}: {len(project_assets)} assets remaining"
            )
            # If no assets found, provide a helpful message
            if not project_assets:
                return [
                    types.TextContent(
                        type="text",
                        text=f"Project {project.name} (ID: {project_id}) contains no assets of types: {', '.join(asset_types)}.",
                    )
                ]

            # Store results in cache for pagination
            new_cache_id = str(uuid.uuid4())
            _project_assets_cache[new_cache_id] = {
                "assets": project_assets,
                "project_info": {
                    "id": project_id,
                    "name": project.name,
                    "description": project.description,
                },
                "timestamp": time.time(),
            }

            # Calculate pagination
            total_items = len(project_assets)
            total_pages = (total_items + items_per_page - 1) // items_per_page

            # Get first page
            first_page_items = project_assets[:items_per_page]

            # Format the response
            response_text = []

            # Add project info header
            response_text.append(f"Project: {project.name}")
            if project.description:
                response_text.append(f"Description: {project.description}")

            response_text.append(
                f"\nAssets (Page 1/{total_pages}, showing items 1-{min(items_per_page, total_items)} of {total_items}):"
            )

            # Format assets
            if first_page_items:
                formatted_assets = [
                    format_asset_info(asset) for asset in first_page_items
                ]
                response_text.extend(formatted_assets)
            else:
                response_text.append("No assets to display.")

            # Add pagination info
            has_more = total_pages > 1
            if has_more:
                response_text.append("\nNavigation options:")
                response_text.append(
                    f"Next page: call get-project-assets with asset_cache_id='{new_cache_id}' and page=2"
                )
                response_text.append(
                    "\nTip: You can control items per page with the items_per_page parameter (default: 10, max: 50)"
                )
            else:
                response_text.append("\nEnd of results.")

            return [types.TextContent(type="text", text="\n".join(response_text))]

        except Exception as e:
            logging.error(f"Error fetching project assets: {e}")
            raise ValueError(f"Error retrieving project assets: {str(e)}")

    if (
        name
        in [
            "create-video-bar-chart-from-two-axis-data",
            "create-video-line-chart-from-two-axis-data",
        ]
        and arguments
    ):
        x_values = arguments.get("x_values")
        y_values = arguments.get("y_values")
        x_label = arguments.get("x_label")
        y_label = arguments.get("y_label")
        title = arguments.get("title")
        filename = arguments.get("filename")

        if not x_values or not y_values or not x_label or not y_label or not title:
            raise ValueError("Missing required arguments")
        if not filename:
            if name == "create-video-bar-chart-from-two-axis-data":
                filename = "bar_chart.mp4"
            elif name == "create-video-line-chart-from-two-axis-data":
                filename = "line_chart.mp4"
            else:
                raise ValueError("Invalid tool name")

        y_axis_safe = validate_y_values(y_values)
        if not y_axis_safe:
            raise ValueError("Y values are not valid")

        # Validate data and prepare for chart generation
        try:
            # Ensure output directory exists
            output_dir = os.path.join(os.getcwd(), "media", "videos", "720p30")
            os.makedirs(output_dir, exist_ok=True)

            # Prepare data for chart generation
            data = {
                "x_values": x_values,
                "y_values": y_values,
                "x_label": x_label,
                "y_label": y_label,
                "title": title,
                "filename": filename,
            }

            # Write data to temporary file
            chart_data_path = os.path.join(os.getcwd(), "chart_data.json")
            with open(chart_data_path, "w") as f:
                json.dump(data, f, indent=4)

            file_path = os.path.join(output_dir, filename)

            # Determine chart type
            chart_type = (
                "bar" if name == "create-video-bar-chart-from-two-axis-data" else "line"
            )

            # Get the script path
            script_path = os.path.join(os.path.dirname(__file__), "generate_charts.py")

            # Run the chart generation script
            env = os.environ.copy()
            env["PYTHONPATH"] = os.getcwd()

            # Use subprocess.run with proper error handling instead of Popen
            result = subprocess.run(
                ["uv", "run", "python", script_path, chart_data_path, chart_type],
                capture_output=True,
                text=True,
                env=env,
                timeout=60,  # 60 second timeout
            )

            if result.returncode != 0:
                error_msg = f"Chart generation failed: {result.stderr}"
                logging.error(error_msg)
                raise RuntimeError(error_msg)

            # Clean up temporary file
            try:
                os.remove(chart_data_path)
            except:
                pass

            chart_type_display = "Bar chart" if chart_type == "bar" else "Line chart"
            return [
                types.TextContent(
                    type="text",
                    text=f"{chart_type_display} video generation started.\nOutput will be saved to {file_path}",
                )
            ]

        except subprocess.TimeoutExpired:
            logging.error("Chart generation timed out")
            raise RuntimeError("Chart generation timed out after 60 seconds")
        except Exception as e:
            logging.error(f"Error generating chart: {str(e)}")
            raise RuntimeError(f"Failed to generate chart: {str(e)}")


async def main():
    # Run the server using stdin/stdout streams
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="video-jungle-mcp",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )
