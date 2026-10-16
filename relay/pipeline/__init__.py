"""The request pipeline (PRD §9.1, §9.3). Filled in from Phase 1.

Every request runs through an ordered list of stages that implement the ``Stage`` protocol:
auth -> admission -> cache -> router -> reservation -> provider. Each stage either continues,
short-circuits with a response (e.g. a cache hit), or rejects with an error. After the
response, or after the stream ends, each stage's ``finalize`` runs in reverse order. That is
where reservations are reconciled and log rows are enqueued.
"""
