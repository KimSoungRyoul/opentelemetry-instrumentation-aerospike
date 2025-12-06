# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for OpenTelemetry Aerospike Instrumentation.

These tests require a running Aerospike instance.
Run with: pytest tests/test_aerospike_integration.py -v
"""

import contextlib
import os
import time

import pytest

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

# Check if aerospike is available
try:
    import aerospike

    AEROSPIKE_AVAILABLE = True
except ImportError:
    AEROSPIKE_AVAILABLE = False

# Configuration from environment
AEROSPIKE_HOST = os.environ.get("AEROSPIKE_HOST", "127.0.0.1")
AEROSPIKE_PORT = int(os.environ.get("AEROSPIKE_PORT", "3000"))
AEROSPIKE_NAMESPACE = os.environ.get("AEROSPIKE_NAMESPACE", "test")


def wait_for_aerospike(host: str, port: int, timeout: int = 30) -> bool:
    """Wait for Aerospike to be ready."""
    if not AEROSPIKE_AVAILABLE:
        return False

    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            config = {"hosts": [(host, port)]}
            client = aerospike.client(config)
            client.connect()
            client.close()
            return True
        except Exception:
            time.sleep(1)
    return False


@pytest.fixture(scope="module")
def aerospike_ready():
    """Check if Aerospike is ready."""
    if not AEROSPIKE_AVAILABLE:
        pytest.skip("aerospike package not installed")

    if not wait_for_aerospike(AEROSPIKE_HOST, AEROSPIKE_PORT):
        pytest.skip(f"Aerospike not available at {AEROSPIKE_HOST}:{AEROSPIKE_PORT}")

    return True


@pytest.fixture
def tracer_setup():
    """Create a tracer provider with in-memory exporter."""
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider, exporter


@pytest.fixture
def instrumented_client(aerospike_ready, tracer_setup):
    """Create an instrumented Aerospike client.

    IMPORTANT: Instrumentor must be called BEFORE creating the client!
    """
    from opentelemetry.instrumentation.aerospike import AerospikeInstrumentor

    provider, exporter = tracer_setup

    # 1. First, instrument
    instrumentor = AerospikeInstrumentor()
    instrumentor.instrument(tracer_provider=provider)

    # 2. Then, create client (now using instrumented aerospike.client)
    config = {"hosts": [(AEROSPIKE_HOST, AEROSPIKE_PORT)]}
    client = aerospike.client(config)
    client.connect()

    yield client, exporter

    # Cleanup
    client.close()
    instrumentor.uninstrument()


class TestAerospikeIntegration:
    """Integration tests with real Aerospike instance."""

    def test_put_get_roundtrip(self, instrumented_client):
        """Test put and get operations."""
        client, exporter = instrumented_client

        key = (AEROSPIKE_NAMESPACE, "demo", "integration_test_key_1")
        bins = {"name": "test_user", "age": 30, "active": True}

        # Put
        client.put(key, bins)

        # Get
        _, meta, record = client.get(key)

        assert record["name"] == "test_user"
        assert record["age"] == 30
        assert record["active"] is True

        # Check spans
        spans = exporter.get_finished_spans()
        assert len(spans) == 2

        put_span = spans[0]
        get_span = spans[1]

        assert put_span.name == f"PUT {AEROSPIKE_NAMESPACE}.demo"
        assert put_span.attributes["db.system"] == "aerospike"
        assert put_span.attributes["db.namespace"] == AEROSPIKE_NAMESPACE
        assert put_span.attributes["db.collection.name"] == "demo"
        assert put_span.attributes["db.operation.name"] == "PUT"

        assert get_span.name == f"GET {AEROSPIKE_NAMESPACE}.demo"
        assert get_span.attributes["db.operation.name"] == "GET"
        assert "db.aerospike.generation" in get_span.attributes
        assert "db.aerospike.ttl" in get_span.attributes

        # Cleanup
        client.remove(key)

    def test_exists_operation(self, instrumented_client):
        """Test exists operation."""
        client, exporter = instrumented_client

        key = (AEROSPIKE_NAMESPACE, "demo", "integration_test_key_exists")
        bins = {"data": "test"}

        # Put first
        client.put(key, bins)
        exporter.clear()

        # Check exists
        _, meta = client.exists(key)

        assert meta is not None

        spans = exporter.get_finished_spans()
        assert len(spans) == 1

        span = spans[0]
        assert span.name == f"EXISTS {AEROSPIKE_NAMESPACE}.demo"
        assert span.attributes["db.operation.name"] == "EXISTS"

        # Cleanup
        client.remove(key)

    def test_remove_operation(self, instrumented_client):
        """Test remove operation."""
        client, exporter = instrumented_client

        key = (AEROSPIKE_NAMESPACE, "demo", "integration_test_key_remove")
        bins = {"data": "to_delete"}

        # Put first
        client.put(key, bins)
        exporter.clear()

        # Remove
        client.remove(key)

        spans = exporter.get_finished_spans()
        assert len(spans) == 1

        span = spans[0]
        assert span.name == f"REMOVE {AEROSPIKE_NAMESPACE}.demo"
        assert span.attributes["db.operation.name"] == "REMOVE"

        # Verify removed
        _, meta = client.exists(key)
        assert meta is None

    def test_select_operation(self, instrumented_client):
        """Test select operation (partial bin retrieval)."""
        client, exporter = instrumented_client

        key = (AEROSPIKE_NAMESPACE, "demo", "integration_test_key_select")
        bins = {"name": "user", "email": "user@test.com", "age": 25}

        # Put first
        client.put(key, bins)
        exporter.clear()

        # Select only specific bins
        _, meta, record = client.select(key, ["name", "age"])

        assert "name" in record
        assert "age" in record

        spans = exporter.get_finished_spans()
        assert len(spans) == 1

        span = spans[0]
        assert span.name == f"SELECT {AEROSPIKE_NAMESPACE}.demo"
        assert span.attributes["db.operation.name"] == "SELECT"

        # Cleanup
        client.remove(key)

    def test_increment_operation(self, instrumented_client):
        """Test increment operation."""
        client, exporter = instrumented_client

        key = (AEROSPIKE_NAMESPACE, "demo", "integration_test_key_incr")
        bins = {"counter": 10}

        # Put first
        client.put(key, bins)
        exporter.clear()

        # Increment
        client.increment(key, "counter", 5)

        spans = exporter.get_finished_spans()
        assert len(spans) == 1

        span = spans[0]
        assert span.name == f"INCREMENT {AEROSPIKE_NAMESPACE}.demo"
        assert span.attributes["db.operation.name"] == "INCREMENT"

        # Verify
        _, _, record = client.get(key)
        assert record["counter"] == 15

        # Cleanup
        client.remove(key)

    def test_append_operation(self, instrumented_client):
        """Test append operation."""
        client, exporter = instrumented_client

        key = (AEROSPIKE_NAMESPACE, "demo", "integration_test_key_append")
        bins = {"message": "Hello"}

        # Put first
        client.put(key, bins)
        exporter.clear()

        # Append
        client.append(key, "message", " World")

        spans = exporter.get_finished_spans()
        assert len(spans) == 1

        span = spans[0]
        assert span.name == f"APPEND {AEROSPIKE_NAMESPACE}.demo"
        assert span.attributes["db.operation.name"] == "APPEND"

        # Verify
        _, _, record = client.get(key)
        assert record["message"] == "Hello World"

        # Cleanup
        client.remove(key)

    def test_prepend_operation(self, instrumented_client):
        """Test prepend operation."""
        client, exporter = instrumented_client

        key = (AEROSPIKE_NAMESPACE, "demo", "integration_test_key_prepend")
        bins = {"message": "World"}

        # Put first
        client.put(key, bins)
        exporter.clear()

        # Prepend
        client.prepend(key, "message", "Hello ")

        spans = exporter.get_finished_spans()
        assert len(spans) == 1

        span = spans[0]
        assert span.name == f"PREPEND {AEROSPIKE_NAMESPACE}.demo"
        assert span.attributes["db.operation.name"] == "PREPEND"

        # Verify
        _, _, record = client.get(key)
        assert record["message"] == "Hello World"

        # Cleanup
        client.remove(key)

    def test_query_operation(self, instrumented_client):
        """Test query operation."""
        client, exporter = instrumented_client

        # Create query object (result intentionally unused - testing span creation)
        _ = client.query(AEROSPIKE_NAMESPACE, "demo")

        spans = exporter.get_finished_spans()
        assert len(spans) == 1

        span = spans[0]
        assert span.name == f"QUERY {AEROSPIKE_NAMESPACE}.demo"
        assert span.attributes["db.operation.name"] == "QUERY"

    def test_scan_operation(self, instrumented_client):
        """Test scan operation."""
        client, exporter = instrumented_client

        # Create scan object (result intentionally unused - testing span creation)
        _ = client.scan(AEROSPIKE_NAMESPACE, "demo")

        spans = exporter.get_finished_spans()
        assert len(spans) == 1

        span = spans[0]
        assert span.name == f"SCAN {AEROSPIKE_NAMESPACE}.demo"
        assert span.attributes["db.operation.name"] == "SCAN"

    def test_error_handling_key_not_found(self, instrumented_client):
        """Test error handling for key not found."""
        client, exporter = instrumented_client

        key = (AEROSPIKE_NAMESPACE, "demo", "nonexistent_key_12345")

        with contextlib.suppress(aerospike.exception.RecordNotFound):
            client.get(key)

        spans = exporter.get_finished_spans()
        assert len(spans) == 1

        span = spans[0]
        assert span.status.status_code == StatusCode.ERROR
        assert "db.response.status_code" in span.attributes
        assert span.attributes["error.type"] == "RecordNotFound"

    def test_operate_operation(self, instrumented_client):
        """Test operate (multi-operation) command."""
        client, exporter = instrumented_client

        key = (AEROSPIKE_NAMESPACE, "demo", "integration_test_key_operate")
        bins = {"counter": 10, "name": "test"}

        # Put first
        client.put(key, bins)
        exporter.clear()

        # Multi-operation using operations helpers
        from aerospike_helpers.operations import operations

        ops = [
            operations.increment("counter", 5),
            operations.read("counter"),
        ]

        _, _, record = client.operate(key, ops)

        spans = exporter.get_finished_spans()
        assert len(spans) == 1

        span = spans[0]
        assert span.name == f"OPERATE {AEROSPIKE_NAMESPACE}.demo"
        assert span.attributes["db.operation.name"] == "OPERATE"

        # Verify
        assert record["counter"] == 15

        # Cleanup
        client.remove(key)

    def test_hooks(self, aerospike_ready, tracer_setup):
        """Test hooks with real Aerospike client."""
        from opentelemetry.instrumentation.aerospike import AerospikeInstrumentor

        provider, exporter = tracer_setup

        request_data = []
        response_data = []

        def request_hook(span, operation, args, kwargs):
            request_data.append({"operation": operation, "time": time.time()})

        def response_hook(span, operation, result):
            response_data.append({"operation": operation, "result_type": type(result).__name__})

        instrumentor = AerospikeInstrumentor()
        instrumentor.instrument(
            tracer_provider=provider, request_hook=request_hook, response_hook=response_hook
        )

        try:
            config = {"hosts": [(AEROSPIKE_HOST, AEROSPIKE_PORT)]}
            client = aerospike.client(config)
            client.connect()

            key = (AEROSPIKE_NAMESPACE, "demo", "hook_test_key")
            client.put(key, {"data": "test"})
            client.get(key)

            assert len(request_data) == 2
            assert len(response_data) == 2
            assert request_data[0]["operation"] == "PUT"
            assert request_data[1]["operation"] == "GET"

            # Cleanup
            client.remove(key)
            client.close()
        finally:
            instrumentor.uninstrument()

    def test_capture_key(self, aerospike_ready, tracer_setup):
        """Test key capture with real Aerospike client."""
        from opentelemetry.instrumentation.aerospike import AerospikeInstrumentor

        provider, exporter = tracer_setup

        instrumentor = AerospikeInstrumentor()
        instrumentor.instrument(tracer_provider=provider, capture_key=True)

        try:
            config = {"hosts": [(AEROSPIKE_HOST, AEROSPIKE_PORT)]}
            client = aerospike.client(config)
            client.connect()

            key = (AEROSPIKE_NAMESPACE, "demo", "my_captured_key")
            client.put(key, {"data": "test"})

            spans = exporter.get_finished_spans()
            span = spans[0]

            assert span.attributes.get("db.aerospike.key") == "my_captured_key"

            # Cleanup
            client.remove(key)
            client.close()
        finally:
            instrumentor.uninstrument()

    def test_capture_key_disabled_by_default(self, aerospike_ready, tracer_setup):
        """Test that key capture is disabled by default for security."""
        from opentelemetry.instrumentation.aerospike import AerospikeInstrumentor

        provider, exporter = tracer_setup

        instrumentor = AerospikeInstrumentor()
        instrumentor.instrument(tracer_provider=provider)

        try:
            config = {"hosts": [(AEROSPIKE_HOST, AEROSPIKE_PORT)]}
            client = aerospike.client(config)
            client.connect()

            key = (AEROSPIKE_NAMESPACE, "demo", "secret_key")
            client.put(key, {"data": "test"})

            spans = exporter.get_finished_spans()
            span = spans[0]

            # Key should NOT be captured by default
            assert "db.aerospike.key" not in span.attributes

            # Cleanup
            client.remove(key)
            client.close()
        finally:
            instrumentor.uninstrument()


class TestAsyncPatternCompatibility:
    """Test compatibility with async wrapper pattern from CLAUDE.md."""

    def test_async_wrapper_pattern(self, aerospike_ready, tracer_setup):
        """Test that instrumentation works with async wrapper pattern."""
        import asyncio
        import functools
        from concurrent.futures import ThreadPoolExecutor

        from opentelemetry.instrumentation.aerospike import AerospikeInstrumentor

        provider, exporter = tracer_setup

        instrumentor = AerospikeInstrumentor()
        instrumentor.instrument(tracer_provider=provider)

        try:
            # Create instrumented client first
            config = {"hosts": [(AEROSPIKE_HOST, AEROSPIKE_PORT)]}
            base_client = aerospike.client(config)
            base_client.connect()

            # Simulate the async wrapper pattern from CLAUDE.md
            # Using ThreadPoolExecutor for sync to async conversion
            executor = ThreadPoolExecutor(max_workers=4)

            class AsyncAerospike:
                def __init__(self, client):
                    self._aerospike = client

                def __getattr__(self, attr: str):
                    async def meth(*args, **kwargs):
                        loop = asyncio.get_running_loop()
                        return await loop.run_in_executor(
                            executor,
                            functools.partial(getattr(self._aerospike, attr), *args, **kwargs),
                        )

                    return meth

            async def test_async_operations():
                async_client = AsyncAerospike(base_client)

                key = (AEROSPIKE_NAMESPACE, "demo", "async_test_key")

                # Async put
                await async_client.put(key, {"async_data": "test"})

                # Async get
                await async_client.get(key)

                # Async remove
                await async_client.remove(key)

            # Run async test
            asyncio.run(test_async_operations())

            # Verify spans were created
            spans = exporter.get_finished_spans()
            assert len(spans) == 3

            operations = [span.attributes["db.operation.name"] for span in spans]
            assert "PUT" in operations
            assert "GET" in operations
            assert "REMOVE" in operations

            base_client.close()
            executor.shutdown(wait=True)
        finally:
            instrumentor.uninstrument()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
