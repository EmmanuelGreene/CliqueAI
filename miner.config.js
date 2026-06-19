// PM2 ecosystem for Emmanuel's CliqueAI (SN83) miner ops.
//
// Required env:
//   CLIQUEAI_WALLET_NAME=<coldkey-name>
//   CLIQUEAI_WALLET_HOTKEY=<hotkey-name>
// Optional multi-hotkey env:
//   CLIQUEAI_WALLET_HOTKEYS=sn83_1,sn83_2
//   CLIQUEAI_AXON_PORTS=8083,8084
// Optional proxy mode env:
//   CLIQUEAI_PROXY_ENABLED=true
//   CLIQUEAI_PROXY_HOST=<target-ip>
//   CLIQUEAI_PROXY_PORT=<target-port>
//   CLIQUEAI_PROXY_TARGETS=<target-ip:port,target-ip:port>
//   CLIQUEAI_PROXY_SPOOFED_HOTKEY=<validator-hotkey-to-forward-as>

const walletName = process.env.CLIQUEAI_WALLET_NAME;
const walletHotkey = process.env.CLIQUEAI_WALLET_HOTKEY;
const walletHotkeys = (process.env.CLIQUEAI_WALLET_HOTKEYS || "")
  .split(",")
  .map((value) => value.trim())
  .filter(Boolean);
const axonPorts = (process.env.CLIQUEAI_AXON_PORTS || "")
  .split(",")
  .map((value) => value.trim())
  .filter(Boolean);

if (!walletName) {
  throw new Error("Set CLIQUEAI_WALLET_NAME before starting miner.config.js");
}
if (walletHotkeys.length === 0 && !walletHotkey) {
  throw new Error("Set CLIQUEAI_WALLET_HOTKEY or CLIQUEAI_WALLET_HOTKEYS before starting miner.config.js");
}

const netuid = process.env.CLIQUEAI_NETUID || "83";
const subtensorNetwork = process.env.CLIQUEAI_SUBTENSOR_NETWORK || "finney";
const minStakeArgs = (process.env.CLIQUEAI_FORCE_VALIDATOR_PERMIT || "false") === "true"
  ? ["--blacklist.force_validator_permit"]
  : [];
const proxyEnabled = (process.env.CLIQUEAI_PROXY_ENABLED || "false") === "true";
const proxyHost = process.env.CLIQUEAI_PROXY_HOST || "";
const proxyPort = process.env.CLIQUEAI_PROXY_PORT || "";
const proxyTargets = process.env.CLIQUEAI_PROXY_TARGETS || "";
const proxyTimeout = process.env.CLIQUEAI_PROXY_TIMEOUT || "30";
const proxySpoofedHotkey = process.env.CLIQUEAI_PROXY_SPOOFED_HOTKEY || "";
const basePort = Number(process.env.CLIQUEAI_AXON_BASE_PORT || "8083");

const hotkeys = walletHotkeys.length > 0 ? walletHotkeys : [walletHotkey];

function buildArgs(hotkey, port) {
  const args = [
    "-m", "CliqueAI.miner",
    "--netuid", netuid,
    "--subtensor.network", subtensorNetwork,
    "--wallet.name", walletName,
    "--wallet.hotkey", hotkey,
    "--axon.port", String(port),
    "--logging.info",
    "--neuron.autoupdate", "0",
    ...minStakeArgs,
  ];

  if (proxyEnabled) {
    if (!proxyTargets && (!proxyHost || !proxyPort)) {
      throw new Error("Proxy mode requires CLIQUEAI_PROXY_TARGETS or CLIQUEAI_PROXY_HOST and CLIQUEAI_PROXY_PORT");
    }
    args.push("--neuron.proxy.enabled");
    if (proxyTargets) {
      args.push("--neuron.proxy.targets", proxyTargets);
    }
    if (proxyHost && proxyPort) {
      args.push("--neuron.proxy.host", proxyHost, "--neuron.proxy.port", proxyPort);
    }
    args.push("--neuron.proxy.timeout", proxyTimeout);
    if (proxySpoofedHotkey) {
      args.push("--neuron.proxy.spoofed_hotkey", proxySpoofedHotkey);
    }
  }

  return args;
}

module.exports = {
  apps: hotkeys.map((hotkey, index) => {
    const port = axonPorts[index] || basePort + index;
    return {
      name: `cliqueai-sn83-${hotkey}`,
      cwd: __dirname,
      script: "python3",
      args: buildArgs(hotkey, port),
      interpreter: "none",
      autorestart: true,
      max_memory_restart: process.env.CLIQUEAI_MAX_MEMORY || "1500M",
      time: true,
      env: {
        PYTHONUNBUFFERED: "1",
      },
    };
  }),
};
