# bench/

| Folder | What | Phase |
|---|---|---|
| `simulator/` | Experiment scripts that drive the simulator. The simulator itself lives in `relay/simulator/`, because it is also Relay's in-process `mock` provider and must ship in the image | 3 (engine), 8+ (scripts) |
| `chaos/` | Chaos scenarios + runner comparing a naive client, LiteLLM, and Relay (PRD E-C) | 8 |
| `load/` | k6 load scripts for gateway overhead (PRD M1, E-D) | 8 |

Load and chaos tests never hit real providers.
