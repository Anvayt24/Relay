# bench/

| Folder | What |
|---|---|
| `simulator/` | Experiment scripts that drive the simulator. The simulator itself lives in `relay/simulator/`, because it is also Relay's in-process `mock` provider and must ship in the image |
| `chaos/` | Chaos scenarios + runner comparing a naive client, LiteLLM, and Relay |
| `load/` | k6 load scripts for gateway overhead |

Load and chaos tests never hit real providers.
