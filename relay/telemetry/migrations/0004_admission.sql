-- Phase 6: time a request spent waiting in the priority queue for a concurrency slot.
-- A rising p95 here is the earliest sign a tenant needs more concurrency (or is overloaded).
ALTER TABLE requests ADD COLUMN queue_wait_ms integer NOT NULL DEFAULT 0;
