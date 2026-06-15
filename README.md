# CliqueAI

CliqueAI - AI-Powered Maximum Clique Solver Network

## Maximum Clique Mechanism
Four-stage autonomous mechanism for distributed problem solving

### Problem Selection
Advanced AI algorithms curate complex graph problems from our distributed database. The system intelligently categorizes challenges by difficulty, graph structure, and computational requirements.

### Miner Selection
Smart allocation engine filters eligible miners and samples each miner with the same difficulty-adjusted probability, keeping problem distribution fair while scaling participation by challenge difficulty.

### Scoring
Dual-metric evaluation system assesses both solution optimality and algorithmic diversity. This approach rewards accuracy while encouraging innovative problem-solving methodologies.

### Weight Setting
Exponential moving average algorithms continuously adjust miner reputation scores. Historical performance data influences future problem allocation and reward distribution.

## Documentation
- [Mechanism](docs/mechanism.md) – Detailed explanation of our mechanism.
- [W&B Logging](docs/wandb_logging.md) – Data logging schema for monitoring, analysis, and debugging via Weights & Biases.

## Getting Started
### Miner
```
./start_miner.sh --wallet.name <coldkey-name> --wallet.hotkey <hotkey-name> --subtensor.network finney --netuid 83 --logging.info --axon.ip <your-miner-ip> --axon.port <your-miner-port>
```

### Emmanuel ops / PM2
Single-hotkey local solver:
```
CLIQUEAI_WALLET_NAME=main \
CLIQUEAI_WALLET_HOTKEY=sn83_1 \
CLIQUEAI_AXON_BASE_PORT=8083 \
pm2 start miner.config.js
```

Multi-hotkey local solver:
```
CLIQUEAI_WALLET_NAME=main \
CLIQUEAI_WALLET_HOTKEYS=sn83_1,sn83_2 \
CLIQUEAI_AXON_PORTS=8083,8084 \
pm2 start miner.config.js
```

Synth-style proxy mode, after a compatible target is verified:
```
CLIQUEAI_WALLET_NAME=main \
CLIQUEAI_WALLET_HOTKEY=sn83_1 \
CLIQUEAI_PROXY_ENABLED=true \
CLIQUEAI_PROXY_HOST=<target-ip> \
CLIQUEAI_PROXY_PORT=<target-port> \
CLIQUEAI_PROXY_SPOOFED_HOTKEY=<validator-hotkey> \
pm2 start miner.config.js
```

Operator-visible query logs:
```
✅ REAL VALIDATOR QUERY ARRIVED | hotkey=... ip=... uuid=... label=... nodes=...
✅ PROXY SUCCESS | target=... clique_size=... elapsed=...s | ...
✅ LOCAL SOLVE SUCCESS | clique_size=... elapsed=...s | ...
❌ PROXY FAILED | ... | error=...
```

### Validator
```
./start_validator.sh --wallet.name <coldkey-name> --wallet.hotkey <hotkey-name> --subtensor.network finney --netuid 83 --logging.info --axon.ip <your-validator-ip> --axon.port <your-validator-port>
```
