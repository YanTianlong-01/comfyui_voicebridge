"""Top-level package for comfyui_voicebridge."""
import os

__author__ = """SevnFading"""
__email__ = "sevnfading@gmail.com"
__version__ = "0.0.1"

from .src.comfyui_voicebridge.nodes import NODE_CLASS_MAPPINGS
from .src.comfyui_voicebridge.nodes import NODE_DISPLAY_NAME_MAPPINGS
WEB_DIRECTORY = "./web/js"


__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]