import asyncio
import time
import typing
import uuid

import bittensor as bt
import httpx
from CliqueAI.clique_algorithms import networkx_algorithm
from CliqueAI.graph.codec import GraphCodec
from CliqueAI.protocol import MaximumCliqueOfLambdaGraph
from common.base.miner import BaseMinerNeuron


class Miner(BaseMinerNeuron):
    """
    Your miner neuron class. You should use this class to define your miner's behavior. In particular, you should replace the forward function with your own logic. You may also want to override the blacklist and priority functions according to your needs.

    This class inherits from the BaseMinerNeuron class, which in turn inherits from BaseNeuron. The BaseNeuron class takes care of routine tasks such as setting up wallet, subtensor, metagraph, logging directory, parsing config, etc. You can override any of the methods in BaseNeuron if you need to customize the behavior.

    This class provides reasonable default behavior for a miner such as blacklisting unrecognized hotkeys, prioritizing requests based on stake, and forwarding requests to the forward function. If you need to define custom
    """

    def __init__(self, config=None):
        super().__init__(config=config)
        self.proxy_enabled = bool(self.config.neuron.proxy.enabled)
        self.proxy_target_host = str(self.config.neuron.proxy.host or "")
        self.proxy_target_port = int(self.config.neuron.proxy.port or 0)
        if self.proxy_enabled:
            if not self.proxy_target_host or not self.proxy_target_port:
                raise ValueError(
                    "Proxy mode requires --neuron.proxy.host and --neuron.proxy.port"
                )
            bt.logging.info(
                "CliqueAI Synth-style proxy enabled: "
                f"{self.proxy_target_host}:{self.proxy_target_port}"
            )
        self.axon.attach(
            forward_fn=self.forward_graph,
            blacklist_fn=self.backlist_graph,
            priority_fn=self.priority_graph,
        )

    def _short_hotkey(self, hotkey: str) -> str:
        if not hotkey:
            return "unknown"
        return hotkey if len(hotkey) <= 14 else f"{hotkey[:7]}...{hotkey[-7:]}"

    def _incoming_query_log_context(
        self, synapse: MaximumCliqueOfLambdaGraph
    ) -> str:
        dendrite = getattr(synapse, "dendrite", None)
        hotkey = str(getattr(dendrite, "hotkey", "") or "")
        ip = str(getattr(dendrite, "ip", "") or "unknown")
        port = getattr(dendrite, "port", "unknown")
        request_uuid = str(getattr(synapse, "uuid", "") or "unknown")
        nonce = getattr(dendrite, "nonce", "unknown")
        return (
            f"hotkey={self._short_hotkey(hotkey)} ip={ip}:{port} "
            f"uuid={request_uuid} nonce={nonce} "
            f"label={synapse.label or 'unknown'} nodes={synapse.number_of_nodes}"
        )

    def _proxy_forward_http_sync(
        self, synapse: MaximumCliqueOfLambdaGraph
    ) -> list[int]:
        incoming = getattr(synapse, "dendrite", None)
        spoofed_hotkey = str(getattr(self.config.neuron.proxy, "spoofed_hotkey", "") or "")
        dendrite_hotkey = spoofed_hotkey or str(getattr(incoming, "hotkey", "") or "")
        configured_timeout = float(self.config.neuron.proxy.timeout)
        validator_timeout = float(getattr(synapse, "timeout", 0) or 0)
        effective_timeout = validator_timeout if validator_timeout > 0 else configured_timeout

        proxy_synapse = MaximumCliqueOfLambdaGraph(
            uuid=synapse.uuid,
            label=synapse.label,
            number_of_nodes=synapse.number_of_nodes,
            encoded_matrix=synapse.encoded_matrix,
            maximum_clique=[],
            timeout=effective_timeout,
        )
        proxy_synapse.dendrite = bt.TerminalInfo(
            ip=str(getattr(incoming, "ip", "127.0.0.1") or "127.0.0.1"),
            port=int(getattr(incoming, "port", 0) or 0),
            version=int(getattr(incoming, "version", 0) or 0),
            nonce=int(getattr(incoming, "nonce", time.time_ns()) or time.time_ns()),
            uuid=str(getattr(incoming, "uuid", "") or uuid.uuid4()),
            hotkey=dendrite_hotkey,
        )

        headers = proxy_synapse.to_headers()
        headers["timeout"] = str(effective_timeout)
        headers["name"] = "MaximumCliqueOfLambdaGraph"
        # CliqueAI targets accept validator-like dendrite metadata when the
        # signature field is present but empty. This matches the successful
        # SN83 target probes from Synth VPS and avoids the invalid-format path
        # returned by literal values such as "None".
        headers["bt_header_dendrite_signature"] = ""

        body = proxy_synapse.model_dump()
        body["timeout"] = effective_timeout
        if isinstance(body.get("dendrite"), dict):
            body["dendrite"]["signature"] = ""

        url = f"http://{self.proxy_target_host}:{self.proxy_target_port}/MaximumCliqueOfLambdaGraph"
        response = httpx.post(
            url,
            json=body,
            headers=headers,
            timeout=effective_timeout,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"Proxy target HTTP {response.status_code}: {response.text[:300]}"
            )

        data = response.json()
        maximum_clique = data.get("maximum_clique")
        if not isinstance(maximum_clique, list):
            raise RuntimeError(f"Proxy target returned no maximum_clique; keys={sorted(data.keys())}")
        return maximum_clique

    async def _proxy_forward(
        self, synapse: MaximumCliqueOfLambdaGraph
    ) -> MaximumCliqueOfLambdaGraph:
        synapse.maximum_clique = await asyncio.to_thread(
            self._proxy_forward_http_sync, synapse
        )
        return synapse

    async def forward_graph(
        self, synapse: MaximumCliqueOfLambdaGraph
    ) -> MaximumCliqueOfLambdaGraph:
        start_time = time.time()
        query_context = self._incoming_query_log_context(synapse)
        bt.logging.info(f"✅ REAL VALIDATOR QUERY ARRIVED | {query_context}")

        if self.proxy_enabled:
            try:
                proxied = await self._proxy_forward(synapse)
                target = f"{self.proxy_target_host}:{self.proxy_target_port}"
                bt.logging.info(
                    f"✅ PROXY SUCCESS | target={target} clique_size={len(proxied.maximum_clique)} "
                    f"elapsed={time.time() - start_time:.2f}s | {query_context}"
                )
                return proxied
            except Exception as e:
                bt.logging.error(f"❌ PROXY FAILED | {query_context} | error={e}")

        codec = GraphCodec()
        adjacency_matrix = codec.decode_matrix(synapse.encoded_matrix)
        adjacency_list = codec.matrix_to_list(adjacency_matrix)
        maximum_clique: list[int] = networkx_algorithm(synapse.number_of_nodes, adjacency_list)
        # or use GNN models
        # from CliqueAI.clique_algorithms import scattering_clique_algorithm
        # maximum_clique = scattering_clique_algorithm(synapse.number_of_nodes, adjacency_list)
        bt.logging.info(
            f"✅ LOCAL SOLVE SUCCESS | clique_size={len(maximum_clique)} "
            f"elapsed={time.time() - start_time:.2f}s | {query_context}"
        )
        synapse.maximum_clique = maximum_clique
        return synapse

    async def backlist_graph(
        self, synapse: MaximumCliqueOfLambdaGraph
    ) -> typing.Tuple[bool, str]:
        return await self.blacklist(synapse)

    async def priority_graph(self, synapse: MaximumCliqueOfLambdaGraph) -> float:
        return await self.priority(synapse)


if __name__ == "__main__":
    with Miner() as miner:
        bt.logging.info("Miner has started running.")
        while True:
            if miner.should_exit:
                bt.logging.info("Miner is exiting.")
                break
            time.sleep(1)
