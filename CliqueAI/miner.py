import asyncio
import concurrent.futures
import json
import os
import time
import typing
import uuid

import bittensor as bt
import httpx
from CliqueAI.clique_algorithms import anytime_clique_algorithm
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
        self._proxy_success_count = 0
        self._proxy_target_successes: dict[str, int] = {}
        self._proxy_target_failures: dict[str, int] = {}
        self._graph_sample_dir = "/opt/CliqueAI/live_graph_samples"
        self._max_graph_samples = 50
        # Start conservative: live SN83 targets often hang behind source-IP gates.
        # A proxy that is genuinely fast can still win before local finishes and
        # will reset this streak on its first success.
        self._proxy_miss_streak = 3
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

    def _save_graph_sample(self, synapse: MaximumCliqueOfLambdaGraph) -> None:
        """Persist a bounded rolling set of live graphs for offline solver tuning."""
        try:
            os.makedirs(self._graph_sample_dir, exist_ok=True)
            request_uuid = str(getattr(synapse, "uuid", "") or uuid.uuid4())
            safe_uuid = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in request_uuid)
            path = os.path.join(self._graph_sample_dir, f"{int(time.time())}_{safe_uuid}.json")
            payload = {
                "uuid": request_uuid,
                "label": synapse.label,
                "number_of_nodes": synapse.number_of_nodes,
                "encoded_matrix": synapse.encoded_matrix,
                "timeout": float(getattr(synapse, "timeout", 0) or 0),
                "captured_at": time.time(),
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f)

            samples = [
                os.path.join(self._graph_sample_dir, name)
                for name in os.listdir(self._graph_sample_dir)
                if name.endswith(".json")
            ]
            if len(samples) > self._max_graph_samples:
                samples.sort(key=lambda p: os.path.getmtime(p))
                for old_path in samples[: len(samples) - self._max_graph_samples]:
                    try:
                        os.remove(old_path)
                    except OSError:
                        pass
        except Exception as exc:
            bt.logging.warning(f"⚠️ GRAPH SAMPLE SAVE FAILED | error={exc}")

    def _local_solve_sync(
        self, synapse: MaximumCliqueOfLambdaGraph, deadline: float | None = None
    ) -> list[int]:
        codec = GraphCodec()
        adjacency_matrix = codec.decode_matrix(synapse.encoded_matrix)
        adjacency_list = codec.matrix_to_list(adjacency_matrix)
        return anytime_clique_algorithm(
            synapse.number_of_nodes,
            adjacency_list,
            deadline=deadline,
            time_limit=float(getattr(synapse, "timeout", 0) or 0) or None,
            seed_material=str(getattr(synapse, "uuid", "") or synapse.encoded_matrix[:64]),
        )

    def _local_solve_with_timing_sync(
        self, synapse: MaximumCliqueOfLambdaGraph, deadline: float | None = None
    ) -> tuple[list[int], float]:
        started = time.time()
        return self._local_solve_sync(synapse, deadline), time.time() - started

    def _proxy_grace_seconds(
        self, validator_timeout: float, local_elapsed: float | None
    ) -> tuple[float, str]:
        """Return how long to wait for proxy after a valid local answer is ready.

        Proxy racing is only useful when a target actually wins sometimes.  If the
        live targets have been missing repeatedly, a fixed 3s grace turns strong
        local answers into slower answers.  Keep an initial/default grace while
        probing, but shrink it after a miss streak until a proxy success resets
        the streak.
        """
        base = min(3.0, max(0.5, validator_timeout * 0.1)) if validator_timeout > 0 else 1.0
        if self._proxy_miss_streak >= 3:
            if validator_timeout > 0 and validator_timeout <= 10.0:
                return min(base, 0.15), "proxy-miss-streak-short-timeout"
            return min(base, 0.50), "proxy-miss-streak"
        if self._proxy_success_count < 3 or self._proxy_miss_streak > 0:
            if validator_timeout > 0 and validator_timeout <= 10.0:
                return min(base, 0.25), "limited-proxy-evidence-short-timeout"
            return min(base, 0.75), "limited-proxy-evidence"
        if local_elapsed is not None and validator_timeout > 0 and local_elapsed >= validator_timeout * 0.75:
            return min(base, 0.25), "protect-deadline"
        return base, "default"

    def _record_proxy_success(self, target: str) -> None:
        self._proxy_success_count += 1
        self._proxy_miss_streak = 0
        self._proxy_target_successes[target] = self._proxy_target_successes.get(target, 0) + 1

    def _record_proxy_miss(self) -> None:
        self._proxy_miss_streak += 1

    def _record_proxy_target_failure(self, target: str) -> None:
        self._proxy_target_failures[target] = self._proxy_target_failures.get(target, 0) + 1

    def _proxy_target_summary(self) -> str:
        labels = [f"{host}:{port}" for host, port in self.proxy_targets]
        parts = []
        for label in labels:
            wins = self._proxy_target_successes.get(label, 0)
            fails = self._proxy_target_failures.get(label, 0)
            if wins or fails:
                parts.append(f"{label}=w{wins}/f{fails}")
        return ",".join(parts) or "none"

    def _min_proxy_clique_size(self, number_of_nodes: int) -> int:
        """Reject obviously weak proxy answers so local fallback can compete."""
        if number_of_nodes >= 850:
            return 24
        if number_of_nodes >= 650:
            return 24
        if number_of_nodes >= 450:
            return 20
        if number_of_nodes >= 250:
            return 18
        return 1

    def _is_valid_clique(
        self, synapse: MaximumCliqueOfLambdaGraph, maximum_clique: typing.Any
    ) -> bool:
        if not isinstance(maximum_clique, list):
            return False
        if len(maximum_clique) != len(set(maximum_clique)):
            return False
        if any(not isinstance(node, int) for node in maximum_clique):
            return False
        if any(node < 0 or node >= synapse.number_of_nodes for node in maximum_clique):
            return False

        codec = GraphCodec()
        adjacency_matrix = codec.decode_matrix(synapse.encoded_matrix)
        for index, left in enumerate(maximum_clique):
            for right in maximum_clique[index + 1 :]:
                if not adjacency_matrix[left][right]:
                    return False
        return True

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
            if not self._is_valid_clique(synapse, maximum_clique):
                raise RuntimeError(
                    f"{target_label} returned invalid clique size={len(maximum_clique)}"
                )
            min_proxy_size = self._min_proxy_clique_size(synapse.number_of_nodes)
            if len(maximum_clique) < min_proxy_size:
                raise RuntimeError(
                    f"{target_label} returned weak clique size={len(maximum_clique)} "
                    f"below floor={min_proxy_size}"
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
                    self._record_proxy_target_failure(target_label)
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
        self._save_graph_sample(synapse)

        if self.proxy_enabled:
            proxy_error: Exception | None = None
            local_result: list[int] | None = None
            validator_timeout = float(getattr(synapse, "timeout", 0) or 0)
            safety_buffer = min(1.0, max(0.25, validator_timeout * 0.05)) if validator_timeout > 0 else 0.5
            deadline = start_time + max(0.5, validator_timeout - safety_buffer) if validator_timeout > 0 else None
            proxy_task = asyncio.create_task(self._proxy_forward(synapse))
            local_deadline = None
            if deadline is not None:
                local_deadline = time.perf_counter() + max(0.05, deadline - time.time())
            local_task = asyncio.create_task(
                asyncio.to_thread(self._local_solve_with_timing_sync, synapse, local_deadline)
            )
            local_elapsed: float | None = None

            def remaining_budget() -> float | None:
                if deadline is None:
                    return None
                return max(0.0, deadline - time.time())

            while True:
                remaining = remaining_budget()
                if remaining is not None and remaining <= 0:
                    proxy_task.cancel()
                    local_task.cancel()
                    bt.logging.warning(
                        f"⚠️ DEADLINE EXHAUSTED | returning empty before validator timeout "
                        f"elapsed={time.time() - start_time:.2f}s buffer={safety_buffer:.2f}s | {query_context}"
                    )
                    synapse.maximum_clique = []
                    return synapse
                done, _ = await asyncio.wait(
                    {proxy_task, local_task},
                    return_when=asyncio.FIRST_COMPLETED,
                    timeout=remaining,
                )
                if not done:
                    continue
                if proxy_task in done:
                    try:
                        proxied, target = proxy_task.result()
                        self._record_proxy_success(target)
                        if not local_task.done():
                            remaining = remaining_budget()
                            compare_grace = 0.25 if validator_timeout <= 10.0 else 0.50
                            if remaining is not None:
                                compare_grace = min(compare_grace, remaining)
                            if compare_grace > 0:
                                await asyncio.wait({local_task}, timeout=compare_grace)
                        if local_task.done():
                            local_candidate, local_elapsed = local_task.result()
                            if self._is_valid_clique(synapse, local_candidate) and len(local_candidate) >= len(proxied.maximum_clique):
                                synapse.maximum_clique = local_candidate
                                bt.logging.info(
                                    f"✅ LOCAL BEATS EARLY PROXY | local_clique_size={len(local_candidate)} "
                                    f"proxy_target={target} proxy_clique_size={len(proxied.maximum_clique)} "
                                    f"elapsed={time.time() - start_time:.2f}s local_elapsed={local_elapsed:.2f}s "
                                    f"target_stats={self._proxy_target_summary()} | {query_context}"
                                )
                                return synapse
                        else:
                            local_task.cancel()
                        bt.logging.info(
                            f"✅ PROXY SUCCESS | target={target} clique_size={len(proxied.maximum_clique)} "
                            f"elapsed={time.time() - start_time:.2f}s target_stats={self._proxy_target_summary()} | {query_context}"
                        )
                        return proxied
                    except Exception as e:
                        proxy_error = e
                        self._record_proxy_miss()
                        bt.logging.error(f"❌ PROXY FAILED | {query_context} | error={e}")
                        if local_task.done():
                            local_result, local_elapsed = local_task.result()
                            break
                        remaining = remaining_budget()
                        if remaining is None:
                            local_result, local_elapsed = await local_task
                            break
                        try:
                            local_result, local_elapsed = await asyncio.wait_for(local_task, timeout=remaining)
                            break
                        except asyncio.TimeoutError:
                            local_task.cancel()
                            bt.logging.warning(
                                f"⚠️ LOCAL FALLBACK MISSED DEADLINE | "
                                f"elapsed={time.time() - start_time:.2f}s buffer={safety_buffer:.2f}s "
                                f"proxy_error={proxy_error} | {query_context}"
                            )
                            synapse.maximum_clique = []
                            return synapse

                if local_task in done:
                    local_result, local_elapsed = local_task.result()
                    if not self._is_valid_clique(synapse, local_result):
                        local_result = []
                        bt.logging.warning(
                            f"⚠️ LOCAL SOLVE INVALID CLIQUE | elapsed={time.time() - start_time:.2f}s | {query_context}"
                        )
                    if not proxy_task.done():
                        remaining = remaining_budget()
                        adaptive_grace, grace_policy = self._proxy_grace_seconds(
                            validator_timeout, local_elapsed
                        )
                        grace = adaptive_grace if remaining is None else min(adaptive_grace, remaining)
                        try:
                            proxied, target = await asyncio.wait_for(proxy_task, timeout=grace)
                            self._record_proxy_success(target)
                            bt.logging.info(
                                f"✅ PROXY SUCCESS | target={target} clique_size={len(proxied.maximum_clique)} "
                                f"elapsed={time.time() - start_time:.2f}s grace={grace:.2f}s "
                                f"grace_policy={grace_policy} target_stats={self._proxy_target_summary()} | {query_context}"
                            )
                            return proxied
                        except asyncio.TimeoutError:
                            self._record_proxy_miss()
                            proxy_error = TimeoutError(
                                f"proxy still pending after local result + {grace:.2f}s grace"
                            )
                            proxy_task.cancel()
                            bt.logging.info(
                                f"⏱️ PROXY STILL PENDING | using local fallback "
                                f"elapsed={time.time() - start_time:.2f}s grace={grace:.2f}s "
                                f"grace_policy={grace_policy} miss_streak={self._proxy_miss_streak} "
                                f"proxy_successes={self._proxy_success_count} target_stats={self._proxy_target_summary()} | {query_context}"
                            )
                        except Exception as e:
                            proxy_error = e
                            self._record_proxy_miss()
                            bt.logging.error(f"❌ PROXY FAILED | {query_context} | error={e}")
                    break

            synapse.maximum_clique = local_result or []
            bt.logging.info(
                f"✅ LOCAL SOLVE SUCCESS | clique_size={len(synapse.maximum_clique)} "
                f"elapsed={time.time() - start_time:.2f}s local_elapsed={local_elapsed:.2f}s "
                f"proxy_error={proxy_error} | {query_context}"
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
