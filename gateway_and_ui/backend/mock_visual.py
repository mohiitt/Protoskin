"""Stable visual stand-in used by the gateway until integration.

Swap the import in main.py for visual_engine.pipeline.generate_concept
when Person A's pipeline is ready. This copy never loads a model.
"""

from visual_engine.pipeline import generate_concept

__all__ = ["generate_concept"]
