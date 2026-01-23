"""Interrogator implementations for different models."""

from .clip_interrogator import CLIPInterrogator
from .wd_interrogator import WDInterrogator
from .camie_interrogator import CamieInterrogator 

__all__ = [
    'CLIPInterrogator',
    'WDInterrogator',
    'CamieInterrogator',
]
