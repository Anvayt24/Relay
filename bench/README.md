# bench/

| Folder | What | Phase |
|---|---|---|
| `simulator/` | Fault-injecting provider simulator with ground-truth quality and a mock judge (PRD §11.1) | 3 |
| `chaos/` | Chaos scenarios + runner comparing a naive client, LiteLLM, and Relay (PRD E-C) | 8 |
| `load/` | k6 load scripts for gateway overhead (PRD M1, E-D) | 8 |

Load and chaos tests never hit real providers.
