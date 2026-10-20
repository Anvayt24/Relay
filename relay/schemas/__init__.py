"""Internal data shapes shared by every layer: the OpenAI-shaped chat schema and the error types.

This package is not in PRD §9.6. It exists because ``api``, ``pipeline`` and ``providers`` all
need these types, and putting them in any one of those packages would make the others depend
on it for the wrong reason (e.g. providers importing from api).
"""
