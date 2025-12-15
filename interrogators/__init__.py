"""Interrogator implementations for different models."""

from .clip_interrogator import CLIPInterrogator
from .wd_interrogator import WDInterrogator

__all__ = [
    'CLIPInterrogator',
    'WDInterrogator',
]
