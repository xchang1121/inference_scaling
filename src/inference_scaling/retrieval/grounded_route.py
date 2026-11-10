"""Gate-based allocation, fixed before any fresh question was generated.

Verification is meaningful only with an accepted reference. Otherwise spend the
search budget on the existing anchored SC method. Both choices use the same
paid probe and immutable root; routing never sees answers or evaluation labels.
"""
from inference_scaling.retrieval.grounded import GroundedConfig, grounded_condition


def select_grounded_arm(context, evidence_present, content="relevant"):
    if context not in {"static", "dynamic"} or type(evidence_present) is not bool:
        raise ValueError("a valid context and boolean evidence decision are required")
    if content not in {"relevant", "unrelated"}:
        raise ValueError("unknown evidence content control")
    if not evidence_present:
        return context+"_anchor_sc"
    return context+("_verified_is" if content == "relevant" else "_verified_noise")


def adaptive_grounded_condition(ctrl, common, packet, content, cfg=GroundedConfig()):
    """Execute only the chosen search, with no preliminary candidate generation."""
    return grounded_condition(ctrl, common, packet, content, bool(packet), cfg)
