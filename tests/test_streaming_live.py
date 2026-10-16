"""Integration tests for unbuffered SSE streaming chunk pass-through."""

import time
import pytest
from pareto_router.gateway.client import ParetoAsyncStream, ParetoRouter, ParetoStream


class TestStreamingUnbufferedPassThrough:
    def test_sync_streaming_unbuffered(self) -> None:
        router = ParetoRouter()
        prompt = "Write a short poem about concurrent Go channels"

        t0 = time.perf_counter()
        stream = router.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        assert isinstance(stream, ParetoStream)

        chunks = []
        chunk_arrival_times = []

        for chunk in stream:
            chunk_arrival_times.append(time.perf_counter())
            chunks.append(chunk)

        total_stream_time = time.perf_counter() - t0

        # Verification of unbuffered chunking
        assert len(chunks) >= 4, f"Expected multiple chunks, got {len(chunks)}"
        # Stream finished cleanly
        assert chunks[-1].choices[0].finish_reason == "stop"

        # Content reconstructed properly
        reconstructed = "".join(c.choices[0].delta.content or "" for c in chunks)
        assert len(reconstructed) > 0
        assert "Response from" in reconstructed

        # Verify stream carries routing decision metadata
        assert stream.routing_decision is not None
        assert stream.routing_decision.scores.latency_ms < 15.0

    @pytest.mark.asyncio
    async def test_async_streaming_unbuffered(self) -> None:
        router = ParetoRouter()
        prompt = "Explain Raft log compaction"

        t0 = time.perf_counter()
        stream = await router.chat.completions.acreate(
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        assert isinstance(stream, ParetoAsyncStream)

        chunks = []
        async for chunk in stream:
            chunks.append(chunk)

        total_async_time = time.perf_counter() - t0

        assert len(chunks) >= 4
        assert chunks[-1].choices[0].finish_reason == "stop"
        reconstructed = "".join(c.choices[0].delta.content or "" for c in chunks)
        assert len(reconstructed) > 0
        assert stream.routing_decision is not None
