"""GA main loop — Phase 5 wires all operators together into run_ga().

This file currently contains only the termination helper used in Phase 4
validation.  The full run_ga() implementation is added in Phase 5.
"""


def _is_stagnant(best_history: list[float], window: int = 20) -> bool:
    """Return True when best raw fitness has not improved for `window` gens.

    Uses raw (not shared) fitness because shared fitness fluctuates as
    population composition changes even when the best chromosome is stable.
    """
    if len(best_history) <= window:
        return False
    return best_history[-window] == best_history[-1]
