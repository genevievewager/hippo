"""Real-time causal closed-loop decoding from hippocampal spike activity.

Public entry point: ``run_full_decoder_workflow.py``
(orchestration lives in ``realtime.workflow``).
"""

from realtime.realtime_decoder import RealTimeDecoder

__all__ = ["RealTimeDecoder"]
