"""Response cache.

Exact-match layer in Redis, then a semantic layer in Postgres + pgvector, with the similarity
threshold tuned for >= 99% precision on a held-out labeled set.
"""
