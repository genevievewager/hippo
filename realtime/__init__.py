"""Real-time causal closed-loop decoding from hippocampal spike activity.

Pipeline scripts:
  run_decoder_comparison.py  — model/window optimization (run first)
  run_realtime_decoding.py   — single closed-loop replay
  run_decoder_visualization.py — plot-only (run last)
"""

from realtime.realtime_decoder import RealTimeDecoder

__all__ = ["RealTimeDecoder"]
