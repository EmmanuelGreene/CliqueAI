import asyncio
import concurrent.futures
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
        self.proxy_targets = self._parse_proxy_targets()
        if self.proxy_enabled:
            if not self.proxy_targets:
                raise ValueError(
                    "Proxy mode requires --neuron.proxy.targets or "
                    "--neuron.proxy.host/--neuron.proxy.port"
                )
            bt.logging.info(
                "CliqueAI Synth-style proxy enabled: " + ",".join(
                    f"{host}:{port}" for host, port in self.proxy_targets
                )
            )
        self.axon.attach(
            forward_fn=self.forward_graph,
            blacklist_fn=self.backlist_graph,
            priority_fn=self.priority_graph,
        )

    def _parse_proxy_targets(self) -> list[tuple[str, int]]:
        targets: list[tuple[str, int]] = []
        raw_targets = str(getattr(self.config.neuron.proxy, "targets", "") or "")
        for raw_target in raw_targets.split(","):
            raw_target = raw_target.strip()
            if not raw_target:
                continue
            if ":" not in raw_target:
                raise ValueError(f"Invalid proxy target {raw_target!r}; expected host:port")
            host, port = raw_target.rsplit(":", 1)
            targets.append((host.strip(), int(port)))

        if self.proxy_target_host and self.proxy_target_port:
            single_target = (self.proxy_target_host, self.proxy_target_port)
            if single_target not in targets:
                targets.append(single_target)
        return targets

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
        timeout = getattr(synapse, "timeout", "unknown")
        return (
            f"hotkey={self._short_hotkey(hotkey)} ip={ip}:{port} "
            f"uuid={request_uuid} nonce={nonce} "
            f"label={synapse.label or 'unknown'} nodes={synapse.number_of_nodes} "
            f"timeout={timeout}"
        )

    def _local_solve_sync(self, synapse: MaximumCliqueOfLambdaGraph) -> list[int]:
        codec = GraphCodec()
        adjacency_matrix = codec.decode_matrix(synapse.encoded_matrix)
        adjacency_list = codec.matrix_to_list(adjacency_matrix)
        return networkx_algorithm(synapse.number_of_nodes, adjacency_list)

    def _proxy_forward_http_sync(
        self, synapse: MaximumCliqueOfLambdaGraph
    ) -> tuple[list[int], str]:
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
            # Use fresh dendrite replay metadata. Empty-signature targets only
            # need validator-like hotkey/IP metadata; reusing the validator's
            # original nonce can become stale by the time our forwarded request
            # reaches a busy target.
            nonce=time.time_ns(),
            uuid=str(uuid.uuid4()),
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

        def post_to_target(target: tuple[str, int]) -> tuple[list[int], str]:
            host, port = target
            target_label = f"{host}:{port}"
            url = f"http://{target_label}/MaximumCliqueOfLambdaGraph"
            response = httpx.post(
                url,
                json=body,
                headers=headers,
                timeout=effective_timeout,
            )
            if response.status_code != 200:
                raise RuntimeError(
                    f"{target_label} HTTP {response.status_code}: {response.text[:300]}"
                )

            data = response.json()
            maximum_clique = data.get("maximum_clique")
            if not isinstance(maximum_clique, list):
                raise RuntimeError(
                    f"{target_label} returned no maximum_clique; keys={sorted(data.keys())}"
                )
            return maximum_clique, target_label

        errors: list[str] = []
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(self.proxy_targets))
        futures = {
            executor.submit(post_to_target, target): target for target in self.proxy_targets
        }
        try:
            for future in concurrent.futures.as_completed(futures):
                target = futures[future]
                target_label = f"{target[0]}:{target[1]}"
                try:
                    result = future.result()
                    for pending in futures:
                        if pending is not future:
                            pending.cancel()
                    return result
                except Exception as exc:
                    errors.append(f"{target_label}: {exc}")
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

        raise RuntimeError("; ".join(errors[-5:]) or "all proxy targets failed")

    async def _proxy_forward(
        self, synapse: MaximumCliqueOfLambdaGraph
    ) -> tuple[MaximumCliqueOfLambdaGraph, str]:
        maximum_clique, target = await asyncio.to_thread(
            self._proxy_forward_http_sync, synapse
        )
        synapse.maximum_clique = maximum_clique
        return synapse, target

    async def forward_graph(
        self, synapse: MaximumCliqueOfLambdaGraph
    ) -> MaximumCliqueOfLambdaGraph:
        start_time = time.time()
        query_context = self._incoming_query_log_context(synapse)
        bt.logging.info(f"✅ REAL VALIDATOR QUERY ARRIVED | {query_context}")

        if self.proxy_enabled:
            proxy_task = asyncio.create_task(self._proxy_forward(synapse))
            local_task = asyncio.create_task(asyncio.to_thread(self._local_solve_sync, synapse))
            proxy_error: Exception | None = None
            local_result: list[int] | None = None
            proxy_grace = min(3.0, max(0.5, float(getattr(synapse, "timeout", 10) or 10) * 0.1))

            while True:
                done, _ = await asyncio.wait(
                    {proxy_task, local_task}, return_when=asyncio.FIRST_COMPLETED
                )
                if proxy_task in done:
                    try:
                        proxied, target = proxy_task.result()
                        if not local_task.done():
                            local_task.cancel()
                        bt.logging.info(
                            f"✅ PROXY SUCCESS | target={target} clique_size={len(proxied.maximum_clique)} "
                            f"elapsed={time.time() - start_time:.2f}s | {query_context}"
                        )
                        return proxied
                    except Exception as e:
                        proxy_error = e
                        bt.logging.error(f"❌ PROXY FAILED | {query_context} | error={e}")
                        if local_task.done():
                            local_result = local_task.result()
                            break
                        local_result = await local_task
                        break

                if local_task in done:
                    local_result = local_task.result()
                    if not proxy_task.done():
                        try:
                            proxied, target = await asyncio.wait_for(proxy_task, timeout=proxy_grace)
                            bt.logging.info(
                                f"✅ PROXY SUCCESS | target={target} clique_size={len(proxied.maximum_clique)} "
                                f"elapsed={time.time() - start_time:.2f}s | {query_context}"
                            )
                            return proxied
                        except asyncio.TimeoutError:
                            proxy_task.cancel()
                        except Exception as e:
                            proxy_error = e
                            bt.logging.error(f"❌ PROXY FAILED | {query_context} | error={e}")
                    break

            synapse.maximum_clique = local_result or []
            bt.logging.info(
                f"✅ LOCAL SOLVE SUCCESS | clique_size={len(synapse.maximum_clique)} "
                f"elapsed={time.time() - start_time:.2f}s proxy_error={proxy_error} | {query_context}"
            )
            return synapse

        maximum_clique: list[int] = self._local_solve_sync(synapse)
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
