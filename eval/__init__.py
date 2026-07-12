"""Local evaluation environment for the PTCG AI Battle Challenge.

Public building blocks:

* :mod:`eval.environment` — the cabt engine boundary (``Environment``) that
  confines the engine's process-global / ctypes state.
* :mod:`eval.agents` — the Agent Protocol (``act(obs) -> list[int]``) and
  reference agents.
* :mod:`eval.match` — ``play_match`` agent-vs-agent loop with structured results.
* :mod:`eval.registry` — ``CardRegistry`` / ``get_registry()``, the single
  read-only access point for card + attack master data (SOT-1625).
"""
