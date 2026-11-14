-- Phase 11: why each request went to the model it did, and the trained router's score.
--
-- router_reason: alias | explicit | router | router_fallback:<cause>. Set for every request, so
--   "how often does the router fall back to strong, and why?" is a plain GROUP BY.
-- router_score: the router's estimated P(cheap answer is win-or-tie vs strong), only when the
--   trained router decided. Logged for every routed request because a later phase samples the
--   score band that would move under a lower threshold (PRD §10.2 item 5).
ALTER TABLE requests ADD COLUMN router_reason text;
ALTER TABLE requests ADD COLUMN router_score  real;
