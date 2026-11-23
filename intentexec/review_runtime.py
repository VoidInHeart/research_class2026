"""Serve the frozen proposal review with a process-local copy of the service predictor."""
from functools import lru_cache
from hashlib import sha256
from pathlib import Path

from .executors import BASE
from .expert_review import ExpertReview


@lru_cache(maxsize=1)
def deployed_service_cues():
    from .within_schema_cues import load_schema_cues
    path = Path(__file__).with_name("resources") / "review_schema_cues.npz"
    if sha256(path.read_bytes()).hexdigest() != "8842c43f8ab0a47f925906c620987c3b92d4a7b1ea5239567675681ac8b68c2b":
        raise ValueError("The deployed review feature model differs from the validated resource")
    return load_schema_cues(path)


def reviewed_call(pool, text, primary, mode):
    collaborate = mode == "review"
    cues = deployed_service_cues() if collaborate and primary != BASE else None
    calls = []

    def propose(expert, prefix):
        calls.append(expert)
        return pool.execute(expert, prefix)

    result = ExpertReview(pool, cues).run(text, primary, propose, collaborate=collaborate)
    return result["review"], {"expert_review": mode, "primary_executor": primary,
                              "secondary_executor": result["secondary"], "proposal_count": len(calls),
                              "review_reason": result["reason"], "review_action": result["action"]}
