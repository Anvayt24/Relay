"""Response cache (PRD FR-6). Phase 10.

Exact-match layer in Redis, then a semantic layer in Postgres + pgvector, with the similarity
threshold tuned for >= 99% precision on a held-out labeled set.
"""
