import asyncio
from datetime import timedelta

from videojungle import ApiClient
from mcp.server.models import InitializationOptions
import mcp.types as types
from mcp.server import NotificationOptions, Server
from pydantic import AnyUrl
from . search_local_videos import get_videos_by_keyword
import threading
import osxphotos

import mcp.server.stdio
import sys
import os
import subprocess
from typing import Optional
import logging

if os.environ.get("VJ_API_KEY"):
    VJ_API_KEY = os.environ.get("VJ_API_KEY")
else:
    try:
        VJ_API_KEY = sys.argv[1]
    except Exception:
        VJ_API_KEY = None

# Configure the logging
logging.basicConfig(
    filename='app.log',           # Name of the log file
    level=logging.INFO,           # Log level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(levelname)s - %(message)s'  # Log format
)

if not VJ_API_KEY:
    try: 
        with open(".env", "r") as f:
            for line in f:
                if "VJ_API_KEY" in line:
                    VJ_API_KEY = line.split("=")[1]
    except Exception as e:
        raise Exception("VJ_API_KEY environment variable is required or a .env file with the key is required")
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

# Create global loader instance, (requires access to host computer!)
if sys.platform == "darwin" and os.environ.get("LOAD_PHOTOS_DB"):
    photos_loader = PhotosDBLoader()

server = Server("video-jungle-mcp")

videos_at_start = vj.video_files.list()
counter = 10

tools = ["add-video", "search-local-videos", "search-remote-videos", "generate-edit-from-videos", "generate-edit-from-single-video"]

@server.list_resources()
async def handle_list_resources() -> list[types.Resource]:
    """
    List available video files.
    Each video files is available at a specific url
    """
    global counter, videos_at_start
    counter += 1
    if counter % 10 == 0:
        videos = vj.video_files.list()
        videos_at_start = videos
    videos = [
        types.Resource(
            uri=AnyUrl(f"vj://video-file/{video.id}"),
            name=f"Video Jungle Video: {video.name}",
            description=f"User provided description: {video.description}",
            mimeType="video/mp4",
        )
        for video in videos_at_start
    ]

    '''
    projects = [
        types.Resource(
            uri=AnyUrl(f"vj://project/{project.id}"),
            name=f"Video Jungle Project: {project.name}",
            description=f"Project description: {project.description}",
            mimeType="application/json",
        )
        for project in projects
    ]'''

    return videos # + projects


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
        id = id.lstrip("/video-file/")
        video = vj.video_files.get(id)
        return video.model_dump_json()
    raise ValueError(f"Video not found: {id}")

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
                    text=f"Here are the exact label names you need to match in your query:\n\n For the specific query: {search_query}, you should use the following labels: {photos_loader.db.labels_as_dict} for the search-local-videos tool"
                    )
            )
        ])


@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """
    List available tools.
    Each tool specifies its arguments using JSON Schema validation.
    """
    return [
        types.Tool(
            name="add-video",
            description="Upload video from URL",
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
            description="Search remote videos hosted on Video Jungle by query",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="search-local-videos",
            description="Search local videos in Photos app by keyword",
            inputSchema={
                "type": "object",
                "properties": {
                    "keyword": {"type": "string"},
                },
                "required": ["keyword"],
            },
        ),
        types.Tool(
            name="generate-edit-from-videos",
            description="Generate an edit from videos",
            inputSchema={
                "type": "object",
                "properties": {
                    "project_id": {"type": "string"},
                    "resolution": {"type": "string"},
                    "edit": {"type": "array", "cuts": {"video_id": "string",
                                                       "video_start_time": "time",
                                                       "video_end_time": "time",}},
                },
                "required": ["edit", "project_id", "cuts"],
            },
        ),
        types.Tool(
            name="generate-edit-from-single-video",
            description="Generate a video edit from a single video",
            inputSchema={
                "type": "object",
                "properties": {
                    "project_id": {"type": "string"},
                    "resolution": {"type": "string"},
                    "video_id": {"type": "string"},
                    "edit": {"type": "array", "cuts": {
                                                       "video_start_time": "time",
                                                       "video_end_time": "time",}
                                                       },
                },
                "required": ["edit", "project_id", "video_id", "cuts"],
            },
        ),
    ]

def format_video_info(video):
    try:
        return (
            f"- Video Id: {video.get('video_id', 'N/A')}\n"
            f"  Video name: {video.get('video', {}).get('name', 'N/A')}\n"
            f"  URL to view video: {video.get('video', {}).get('url', 'N/A')}\n"
            f"  Video manuscript: {video.get('script', 'N/A')}"
            f"  Generated description: {video.get('generated_descriptioon', 'N/A')}"
        )
    except Exception as e:
        return f"Error formatting video: {str(e)}"

def format_video_info_long(video):
    try:
        return (
            f"- Video Id: {video.get('video_id', 'N/A')}\n"
            f"  Video name: {video.get('video', {}).get('name', 'N/A')}\n"
            f"  URL to view video: {video.get('video', {}).get('url', 'N/A')}\n"
            f"  Video manuscript: {video.get('script', 'N/A')}"
            f"  Video scenes: {video.get('scene_changes', 'N/A')}"
        )
    except Exception as e:
        return f"Error formatting video: {str(e)}"
    
def format_local_searched_video_details(video):
    # Get video duration in readable format
    duration = str(timedelta(seconds=video.get('exif_info', {}).get('duration', 0)))

    # Format date with timezone
    video_date = video.get('date')
    date_str = video_date.strftime("%B %d, %Y at %I:%M %p %Z") if video_date else "Date unknown"

    # Get location details
    place = video.get('place', {})
    location = f"{place.get('name', 'Location unknown')}" if place else "Location unknown"

    # Format detected faces
    face_count = len(video.get('face_info', []))
    faces = f"Detected {face_count} people" if face_count > 0 else "No people detected"

    return (
        f"Video: {video.get('filename', 'Untitled')}\n"
        f"  📅 Recorded: {date_str}\n"
        f"  ⏱️ Duration: {duration}\n"
        f"  📍 Location: {location}\n"
        f"  👥 People: {faces}\n"
    )

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
    
    if name == "add-video" and arguments:
        name = arguments.get("name")
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
        query = arguments.get("query")
        detailed_response = False
        if not query:
            raise ValueError("Missing query")

        videos = vj.video_files.search(query)
        #logging.info(f"num videos are: {len(videos)}")
        if videos:
            logging.info(f"{videos[0]}")
        if len(videos) == 1:
            return [
                types.TextContent(
                    type="text",
                    text=format_video_info_long(videos[0]),
                )
            ]
        # try to fit into context window
        b = [
            types.TextContent(
                type="text",
                text=(
                    f"Number of Videos Returned: {len(videos)}\n\n"
                    + "\n".join(format_video_info(video) for video in videos)
                ),
            )
        ]
        return b # type: ignore
    
    if name == "search-local-videos" and arguments:
        if not os.environ.get("LOAD_PHOTOS_DB"):
            raise ValueError("You must set the LOAD_PHOTOS_DB environment variable to True to use this tool")
        
        keyword = arguments.get("keyword")
        if not keyword:
            raise ValueError("Missing keyword")
        try:
            db = photos_loader.db
            videos = get_videos_by_keyword(db, keyword)
            total_videos = len(videos)
            display_limit = 100
            videos_to_show = videos[:display_limit]
            result_message = (
                f"Found {total_videos} videos matching '{keyword}'\n"
                + (f"Showing first {display_limit} results:\n" if total_videos > display_limit else "\n")
                + "\n".join(format_local_searched_video_details(video) for video in videos_to_show)
            )
            return [
            types.TextContent(
                type="text",
                text=result_message
            )
        ]
        except Exception as e:
            raise RuntimeError("Local Photos database not yet initialized")
        
    if name == "generate-edit-from-videos" and arguments:
        edit = arguments.get("edit")
        project = arguments.get("project_id")
        resolution = arguments.get("resolution")
        created = False

        logging.info(f"edit is: {edit} and the type is: {type(edit)}")

        if not edit:
            raise ValueError("Missing edit")
        if not project:
            raise ValueError("Missing project")
        if not resolution:
            resolution = "1080x1920"
        
        if resolution == "1080p":
            resolution = "1920x1080"
        elif resolution == "720p":
            resolution = "1280x720"
        
        try:
            w, h = resolution.split("x")
            _ = f"{int(w)}x{int(w)}"
        except Exception as e:
            raise ValueError(f"Resolution must be in the format 'widthxheight' where width and height are integers: {e}")
        
        updated_edit = [{**cut, "type": "videofile", 
                        "audio_levels": [{
                         "audio_level": "0.5",
                         "start_time": cut["video_start_time"],
                         "end_time": cut["video_end_time"],}]
                         } for cut in edit]

        logging.info(f"updated edit is: {updated_edit}")

        json_edit = {
            "video_edit_version": "1.0",
            "video_output_format": "mp4",
            "video_output_resolution": resolution,
            "video_output_fps": 60.0,
            "video_output_filename": "output_video.mp4",
            "audio_overlay": [], # TODO: add this back in 
            "video_series_sequential": updated_edit
        }

        try: 
            proj = vj.projects.get(project)
        except Exception as e:
            logging.info(f"project not found, creating new project because {e}")
            proj = vj.projects.create(name=project, description="Claude generated project")
            project = proj.id
            created = True

        logging.info(f"video edit is: {json_edit}")

        edit = vj.projects.render_edit(project, json_edit)

        try:
            os.chdir("./tools")
            logging.info(f"in directory: {os.getcwd()}")
            # don't block, because this might take a while
            env_vars = {"VJ_API_KEY": VJ_API_KEY,
                        'PATH': os.environ['PATH']}
            logging.info(f"launching viewer with: {edit['asset_id']} {project}.mp4 {proj.name}")
            subprocess.Popen(["uv", "run", "viewer", edit['asset_id'], f"video-edit-{project}.mp4", proj.name], 
                             env=env_vars)
        except Exception as e:
            logging.info(f"Error running viewer: {e}")
            
        if created:
            # we created a new project so let the user / LLM know
            return [
                types.TextContent(
                    type="text",
                    text=f"Created new project {proj.name} and created edit {edit} with raw edit info: {updated_edit}"
                )
            ]
        
        return [
            types.TextContent(
                type="text",
                text=f"Generated edit in existing project {proj.name} with generated asset info: {edit} and raw edit info: {updated_edit}",
            )
        ]
    if name == "generate-edit-from-single-video" and arguments:
        edit = arguments.get("edit")
        project = arguments.get("project_id")
        video_id = arguments.get("video_id")

        resolution = arguments.get("resolution")
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
            _ = f"{int(w)}x{int(w)}"
        except Exception as e:
            raise ValueError(f"Resolution must be in the format 'widthxheight' where width and height are integers: {e}")
        
        try:
            updated_edit = [{"video_id": video_id,
                            "video_start_time": cut["video_start_time"],
                            "video_end_time": cut["video_end_time"],
                            "type": "videofile", 
                            "audio_levels": [{
                            "audio_level": "0.5",
                            "start_time": cut["video_start_time"],
                            "end_time": cut["video_end_time"],}]
                            } for cut in edit]
        except Exception as e:
            raise ValueError(f"Error updating edit: {e}")
        
        logging.info(f"updated edit is: {updated_edit}")

        json_edit = {
            "video_edit_version": "1.0",
            "video_output_format": "mp4",
            "video_output_resolution": resolution,
            "video_output_fps": 60.0,
            "video_output_filename": "output_video.mp4",
            "audio_overlay": [], # TODO: add this back in 
            "video_series_sequential": updated_edit
        }

        try: 
            proj = vj.projects.get(project)
        except Exception as e:
            proj = vj.projects.create(name=project, description=f"Claude generated project")
            project = proj.id
            created = True

        logging.info(f"video edit is: {json_edit}")

        edit = vj.projects.render_edit(project, json_edit)
        logging.info(f"edit is: {edit}")
        try:
            os.chdir("./tools")
            logging.info(f"in directory: {os.getcwd()}")
            # don't block, because this might take a while
            env_vars = {"VJ_API_KEY": VJ_API_KEY,
                        'PATH': os.environ['PATH']}
            logging.info(f"launching viewer with: {edit['asset_id']} {project}.mp4 {proj.name}")
            subprocess.Popen(["uv", "run", "viewer", edit['asset_id'], f"video-edit-{project}.mp4", proj.name], 
                             env=env_vars)
        except Exception as e:
            logging.info(f"Error running viewer: {e}")
        if created:
            # we created a new project so let the user / LLM know
            logging.info(f"created new project {proj.name} and created edit {edit}")
            return [
                types.TextContent(
                    type="text",
                    text=f"Created new project {proj.name} with raw edit info: {edit}"
                )
            ]

        return [
            types.TextContent(
                type="text",
                text=f"Generated edit in project {proj.name} with raw edit info: {edit}",
            )
        ]

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
