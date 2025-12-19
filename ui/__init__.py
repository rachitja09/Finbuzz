"""ui package for reusable UI components

Re-export commonly used components for easy imports like `from ui import metric_sparkline`.
"""
from .components import metric_sparkline as metric_sparkline
from .components import render_debug_panel as render_debug_panel
from .components import compact_columns as compact_columns

__all__ = ["metric_sparkline", "render_debug_panel", "compact_columns"]
