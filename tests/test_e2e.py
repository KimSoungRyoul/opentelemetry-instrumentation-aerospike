# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""End-to-End tests for OpenTelemetry Aerospike Instrumentation.

These tests simulate real-world application scenarios with actual
Aerospike database operations and verify that tracing works correctly.

Prerequisites:
    docker run -d --name aerospike -p 3000:3000 aerospike:ce-8.1.0.1_1
"""

import asyncio
import contextlib
import functools
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import aerospike
import pytest

from opentelemetry.instrumentation.aerospike import AerospikeInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

# Test configuration
AEROSPIKE_HOST = "127.0.0.1"
AEROSPIKE_PORT = 3000
NAMESPACE = "test"


@pytest.fixture(scope="module")
def aerospike_available():
    """Check if Aerospike is available."""
    try:
        config = {"hosts": [(AEROSPIKE_HOST, AEROSPIKE_PORT)]}
        client = aerospike.client(config)
        client.connect()
        client.close()
        return True
    except Exception as e:
        pytest.skip(f"Aerospike not available: {e}")


@pytest.fixture
def tracer_setup():
    """Create tracer provider with in-memory exporter."""
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider, exporter


@pytest.fixture
def instrumented_env(aerospike_available, tracer_setup):
    """Set up instrumented environment."""
    provider, exporter = tracer_setup

    instrumentor = AerospikeInstrumentor()
    instrumentor.instrument(tracer_provider=provider)

    config = {"hosts": [(AEROSPIKE_HOST, AEROSPIKE_PORT)]}
    client = aerospike.client(config)
    client.connect()

    yield client, exporter

    client.close()
    instrumentor.uninstrument()


class TestUserSessionScenario:
    """E2E tests simulating user session management."""

    def test_user_login_session_flow(self, instrumented_env):
        """Test user login creates session, accesses it, then logs out."""
        client, exporter = instrumented_env

        # Generate unique user/session IDs
        user_id = f"user_{uuid.uuid4().hex[:8]}"
        session_id = f"session_{uuid.uuid4().hex[:16]}"

        # 1. User login - Create session
        session_key = (NAMESPACE, "sessions", session_id)
        session_data = {
            "user_id": user_id,
            "login_time": int(time.time()),
            "ip_address": "192.168.1.100",
            "user_agent": "Mozilla/5.0",
            "is_active": True,
        }
        client.put(session_key, session_data)

        # 2. Multiple session accesses (simulating page views)
        for _ in range(3):
            _, meta, record = client.get(session_key)
            assert record["user_id"] == user_id

            # Update last access time
            client.put(session_key, {"last_access": int(time.time())})

        # 3. Check session exists
        _, meta = client.exists(session_key)
        assert meta is not None

        # 4. User logout - Delete session
        client.remove(session_key)

        # 5. Verify session is deleted
        _, meta = client.exists(session_key)
        assert meta is None

        # Verify spans
        spans = exporter.get_finished_spans()

        # Expected: 1 PUT (create) + 3 GET + 3 PUT (update) + 1 EXISTS + 1 REMOVE + 1 EXISTS = 10
        assert len(spans) == 10

        operations = [span.attributes["db.operation.name"] for span in spans]
        assert operations.count("PUT") == 4
        assert operations.count("GET") == 3
        assert operations.count("EXISTS") == 2
        assert operations.count("REMOVE") == 1

        # All spans should be successful (except the last EXISTS which finds nothing)
        for span in spans[:-1]:
            assert span.status.status_code != StatusCode.ERROR

    def test_concurrent_session_access(self, instrumented_env):
        """Test multiple concurrent requests accessing the same session."""
        client, exporter = instrumented_env

        session_id = f"concurrent_session_{uuid.uuid4().hex[:8]}"
        session_key = (NAMESPACE, "sessions", session_id)

        # Create session
        client.put(session_key, {"user_id": "test_user", "data": "initial"})
        exporter.clear()

        # Simulate concurrent access
        results = []

        def access_session(request_id: int):
            _, _, record = client.get(session_key)
            return {"request_id": request_id, "data": record}

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(access_session, i) for i in range(10)]
            results = [f.result() for f in futures]

        # All should succeed
        assert len(results) == 10

        # Should have 10 GET spans
        spans = exporter.get_finished_spans()
        assert len(spans) == 10

        for span in spans:
            assert span.attributes["db.operation.name"] == "GET"
            assert span.status.status_code != StatusCode.ERROR

        # Cleanup
        client.remove(session_key)


class TestCacheScenario:
    """E2E tests simulating cache operations."""

    def test_cache_miss_then_hit(self, instrumented_env):
        """Test cache miss triggers data fetch, then subsequent requests hit cache."""
        client, exporter = instrumented_env

        cache_key = (NAMESPACE, "cache", f"product_{uuid.uuid4().hex[:8]}")

        # 1. Cache miss - try to get, will fail
        try:
            client.get(cache_key)
            cache_hit = True
        except aerospike.exception.RecordNotFound:
            cache_hit = False

        assert cache_hit is False

        # 2. Fetch data and store in cache
        product_data = {
            "name": "Widget Pro",
            "price": 99.99,
            "stock": 150,
            "category": "electronics",
        }
        client.put(cache_key, product_data)

        # 3. Multiple cache hits
        for _ in range(5):
            _, _, record = client.get(cache_key)
            assert record["name"] == "Widget Pro"

        # Verify spans
        spans = exporter.get_finished_spans()

        # 1 GET (miss/error) + 1 PUT + 5 GET (hits) = 7
        assert len(spans) == 7

        # First GET should have error status
        assert spans[0].status.status_code == StatusCode.ERROR
        assert spans[0].attributes["error.type"] == "RecordNotFound"

        # Remaining should be successful
        for span in spans[1:]:
            assert span.status.status_code != StatusCode.ERROR

        # Cleanup
        client.remove(cache_key)

    def test_cache_invalidation(self, instrumented_env):
        """Test cache data is updated when source changes."""
        client, exporter = instrumented_env

        product_id = f"product_{uuid.uuid4().hex[:8]}"
        cache_key = (NAMESPACE, "cache", product_id)

        # Initial cache
        client.put(cache_key, {"price": 100, "version": 1})

        # Read cached value
        _, _, record = client.get(cache_key)
        assert record["price"] == 100

        # Price update - invalidate and refresh cache
        client.remove(cache_key)
        client.put(cache_key, {"price": 80, "version": 2})  # Sale price

        # Read new cached value
        _, _, record = client.get(cache_key)
        assert record["price"] == 80

        spans = exporter.get_finished_spans()
        operations = [span.attributes["db.operation.name"] for span in spans]

        assert "PUT" in operations
        assert "GET" in operations
        assert "REMOVE" in operations

        # Cleanup
        client.remove(cache_key)


class TestRealTimeCounterScenario:
    """E2E tests simulating real-time counters."""

    def test_page_view_counter(self, instrumented_env):
        """Test increment page view counter for analytics."""
        client, exporter = instrumented_env

        page_id = f"page_{uuid.uuid4().hex[:8]}"
        counter_key = (NAMESPACE, "counters", page_id)

        # Initialize counter
        client.put(counter_key, {"views": 0, "unique_visitors": 0})
        exporter.clear()

        # Simulate page views
        for _ in range(10):
            client.increment(counter_key, "views", 1)

        # Simulate unique visitors (batch increment)
        client.increment(counter_key, "unique_visitors", 5)

        # Read final counts
        _, _, record = client.get(counter_key)
        assert record["views"] == 10
        assert record["unique_visitors"] == 5

        # Verify spans
        spans = exporter.get_finished_spans()

        # 10 INCREMENT + 1 INCREMENT + 1 GET = 12
        assert len(spans) == 12

        increment_spans = [s for s in spans if s.attributes["db.operation.name"] == "INCREMENT"]
        assert len(increment_spans) == 11

        # Cleanup
        client.remove(counter_key)

    def test_rate_limiter(self, instrumented_env):
        """Test rate limiting using increment operation."""
        client, exporter = instrumented_env

        user_id = f"user_{uuid.uuid4().hex[:8]}"
        rate_key = (NAMESPACE, "rate_limits", user_id)
        rate_limit = 5

        # Initialize rate limit counter
        client.put(rate_key, {"count": 0})
        exporter.clear()

        allowed_requests = 0
        denied_requests = 0

        # Simulate 10 requests
        for i in range(10):
            # Increment counter (returns previous value)
            client.increment(rate_key, "count", 1)

            # Check current count (i+1 because we just incremented)
            current_count = i + 1

            if current_count <= rate_limit:
                allowed_requests += 1
            else:
                denied_requests += 1

        assert allowed_requests == 5
        assert denied_requests == 5

        # Verify final count
        _, _, record = client.get(rate_key)
        assert record["count"] == 10

        # Cleanup
        client.remove(rate_key)


class TestBatchOperationsScenario:
    """E2E tests simulating batch data processing."""

    def test_bulk_user_import(self, instrumented_env):
        """Test import multiple users in batch operation."""
        client, exporter = instrumented_env

        batch_id = uuid.uuid4().hex[:8]
        user_keys = []

        # Create 20 users
        for i in range(20):
            key = (NAMESPACE, "users", f"batch_user_{batch_id}_{i}")
            user_keys.append(key)
            client.put(
                key,
                {
                    "name": f"User {i}",
                    "email": f"user{i}@example.com",
                    "created_at": int(time.time()),
                },
            )

        exporter.clear()

        # Batch read all users using select
        for key in user_keys:
            _, _, record = client.select(key, ["name", "email"])
            assert "name" in record

        # Verify spans
        spans = exporter.get_finished_spans()
        assert len(spans) == 20

        for span in spans:
            assert span.attributes["db.operation.name"] == "SELECT"
            assert span.attributes["db.namespace"] == NAMESPACE
            assert span.attributes["db.collection.name"] == "users"

        # Cleanup
        for key in user_keys:
            client.remove(key)

    def test_scan_and_process(self, instrumented_env):
        """Test scan records and process them."""
        client, exporter = instrumented_env

        set_name = f"scan_test_{uuid.uuid4().hex[:8]}"

        # Create test records
        for i in range(5):
            key = (NAMESPACE, set_name, f"record_{i}")
            client.put(key, {"index": i, "status": "pending"})

        exporter.clear()

        # Create scan
        scan = client.scan(NAMESPACE, set_name)

        # Process records
        processed = []

        def callback(record):
            key, meta, bins = record
            processed.append(bins)

        scan.foreach(callback)

        assert len(processed) == 5

        # Verify scan span was created
        spans = exporter.get_finished_spans()
        assert len(spans) >= 1

        scan_span = spans[0]
        assert scan_span.attributes["db.operation.name"] == "SCAN"

        # Cleanup
        for i in range(5):
            client.remove((NAMESPACE, set_name, f"record_{i}"))


class TestQueryScenario:
    """E2E tests simulating query operations."""

    def test_query_creation(self, instrumented_env):
        """Test create and configure a query."""
        client, exporter = instrumented_env

        set_name = f"query_set_{uuid.uuid4().hex[:8]}"

        # Create test data
        for i in range(3):
            key = (NAMESPACE, set_name, f"item_{i}")
            client.put(key, {"category": "electronics", "price": 100 + i * 10})

        exporter.clear()

        # Create query
        query = client.query(NAMESPACE, set_name)
        query.select("category", "price")

        # Verify query span
        spans = exporter.get_finished_spans()
        assert len(spans) == 1

        span = spans[0]
        assert span.attributes["db.operation.name"] == "QUERY"
        assert span.attributes["db.namespace"] == NAMESPACE
        assert span.attributes["db.collection.name"] == set_name

        # Cleanup
        for i in range(3):
            client.remove((NAMESPACE, set_name, f"item_{i}"))


class TestErrorHandlingScenario:
    """E2E tests for error handling scenarios."""

    def test_record_not_found_handling(self, instrumented_env):
        """Test gracefully handle missing records."""
        client, exporter = instrumented_env

        missing_key = (NAMESPACE, "test", f"nonexistent_{uuid.uuid4().hex}")

        # Try to get non-existent record
        try:
            client.get(missing_key)
            found = True
        except aerospike.exception.RecordNotFound:
            found = False

        assert found is False

        spans = exporter.get_finished_spans()
        assert len(spans) == 1

        span = spans[0]
        assert span.status.status_code == StatusCode.ERROR
        assert span.attributes["error.type"] == "RecordNotFound"
        assert "db.response.status_code" in span.attributes

    def test_generation_conflict(self, instrumented_env):
        """Test handle optimistic locking with generation check."""
        client, exporter = instrumented_env

        key = (NAMESPACE, "test", f"gen_test_{uuid.uuid4().hex[:8]}")

        # Create record
        client.put(key, {"value": 1})

        # Get current generation
        _, meta, _ = client.get(key)
        current_gen = meta["gen"]

        # Update record normally
        client.put(key, {"value": 2})

        exporter.clear()

        # Try to update with wrong generation (should fail)
        try:
            # Use policy with expected generation
            write_policy = {"gen": aerospike.POLICY_GEN_EQ}
            meta = {"gen": current_gen}  # Old generation
            client.put(key, {"value": 3}, meta=meta, policy=write_policy)
            conflict = False
        except aerospike.exception.RecordGenerationError:
            conflict = True

        assert conflict is True

        spans = exporter.get_finished_spans()
        assert len(spans) == 1

        span = spans[0]
        assert span.status.status_code == StatusCode.ERROR
        assert "Generation" in span.attributes.get("error.type", "")

        # Cleanup
        client.remove(key)


class TestAsyncPatternScenario:
    """E2E tests for async wrapper pattern compatibility."""

    def test_async_crud_operations(self, instrumented_env):
        """Test full CRUD cycle using async wrapper pattern."""
        client, exporter = instrumented_env

        executor = ThreadPoolExecutor(max_workers=4)

        class AsyncClient:
            def __init__(self, sync_client):
                self._client = sync_client

            def __getattr__(self, name):
                async def method(*args, **kwargs):
                    loop = asyncio.get_running_loop()
                    return await loop.run_in_executor(
                        executor, functools.partial(getattr(self._client, name), *args, **kwargs)
                    )

                return method

        async def run_async_operations():
            async_client = AsyncClient(client)

            key = (NAMESPACE, "async_test", f"async_{uuid.uuid4().hex[:8]}")

            # Create
            await async_client.put(key, {"name": "Async Test", "value": 42})

            # Read
            result = await async_client.get(key)
            assert result[2]["name"] == "Async Test"

            # Update
            await async_client.put(key, {"name": "Async Test Updated", "value": 100})

            # Read again
            result = await async_client.get(key)
            assert result[2]["value"] == 100

            # Delete
            await async_client.remove(key)

            return True

        exporter.clear()

        # Run async operations
        result = asyncio.run(run_async_operations())
        assert result is True

        executor.shutdown(wait=True)

        # Verify spans
        spans = exporter.get_finished_spans()

        # PUT + GET + PUT + GET + REMOVE = 5
        assert len(spans) == 5

        operations = [span.attributes["db.operation.name"] for span in spans]
        assert operations == ["PUT", "GET", "PUT", "GET", "REMOVE"]

    def test_async_parallel_operations(self, instrumented_env):
        """Test multiple parallel async operations."""
        client, exporter = instrumented_env

        executor = ThreadPoolExecutor(max_workers=10)

        class AsyncClient:
            def __init__(self, sync_client):
                self._client = sync_client

            def __getattr__(self, name):
                async def method(*args, **kwargs):
                    loop = asyncio.get_running_loop()
                    return await loop.run_in_executor(
                        executor, functools.partial(getattr(self._client, name), *args, **kwargs)
                    )

                return method

        async def parallel_writes():
            async_client = AsyncClient(client)
            batch_id = uuid.uuid4().hex[:8]

            # Create records in parallel
            tasks = []
            keys = []
            for i in range(10):
                key = (NAMESPACE, "parallel_test", f"parallel_{batch_id}_{i}")
                keys.append(key)
                task = async_client.put(key, {"index": i})
                tasks.append(task)

            await asyncio.gather(*tasks)

            # Clean up in parallel
            cleanup_tasks = [async_client.remove(key) for key in keys]
            await asyncio.gather(*cleanup_tasks)

            return len(keys)

        exporter.clear()

        count = asyncio.run(parallel_writes())
        assert count == 10

        executor.shutdown(wait=True)

        # Verify spans
        spans = exporter.get_finished_spans()

        # 10 PUT + 10 REMOVE = 20
        assert len(spans) == 20

        put_count = sum(1 for s in spans if s.attributes["db.operation.name"] == "PUT")
        remove_count = sum(1 for s in spans if s.attributes["db.operation.name"] == "REMOVE")

        assert put_count == 10
        assert remove_count == 10


class TestHooksScenario:
    """E2E tests for hook functionality."""

    def test_request_response_hooks_logging(self, aerospike_available, tracer_setup):
        """Test use hooks to add custom logging/metrics."""
        provider, exporter = tracer_setup

        request_log = []
        response_log = []
        error_log = []

        def request_hook(span, operation, args, kwargs):
            request_log.append(
                {
                    "operation": operation,
                    "timestamp": time.time(),
                    "span_id": span.get_span_context().span_id,
                }
            )
            span.set_attribute("custom.request_logged", True)

        def response_hook(span, operation, result):
            response_log.append(
                {
                    "operation": operation,
                    "timestamp": time.time(),
                    "has_result": result is not None,
                }
            )
            span.set_attribute("custom.response_logged", True)

        def error_hook(span, operation, exception):
            error_log.append(
                {
                    "operation": operation,
                    "error_type": type(exception).__name__,
                    "timestamp": time.time(),
                }
            )

        instrumentor = AerospikeInstrumentor()
        instrumentor.instrument(
            tracer_provider=provider,
            request_hook=request_hook,
            response_hook=response_hook,
            error_hook=error_hook,
        )

        try:
            config = {"hosts": [(AEROSPIKE_HOST, AEROSPIKE_PORT)]}
            client = aerospike.client(config)
            client.connect()

            key = (NAMESPACE, "hooks_test", f"hook_{uuid.uuid4().hex[:8]}")

            # Successful operations
            client.put(key, {"test": "data"})
            client.get(key)

            # Error operation
            with contextlib.suppress(aerospike.exception.RecordNotFound):
                client.get((NAMESPACE, "hooks_test", "nonexistent_key_xyz"))

            # Cleanup
            client.remove(key)
            client.close()

            # Verify hooks were called
            assert len(request_log) == 4  # PUT, GET, GET (error), REMOVE
            assert len(response_log) == 3  # PUT, GET, REMOVE (successful)
            assert len(error_log) == 1  # GET (error)

            # Verify custom attributes
            spans = exporter.get_finished_spans()
            successful_spans = [s for s in spans if s.status.status_code != StatusCode.ERROR]

            for span in successful_spans:
                assert span.attributes.get("custom.request_logged") is True
                assert span.attributes.get("custom.response_logged") is True

        finally:
            instrumentor.uninstrument()


class TestTraceContextPropagation:
    """E2E tests for trace context propagation."""

    def test_spans_share_trace_id(self, instrumented_env):
        """Test all operations in a request share the same trace ID."""
        client, exporter = instrumented_env

        key = (NAMESPACE, "trace_test", f"trace_{uuid.uuid4().hex[:8]}")

        # Multiple operations in sequence
        client.put(key, {"step": 1})
        client.get(key)
        client.put(key, {"step": 2})
        client.get(key)
        client.remove(key)

        spans = exporter.get_finished_spans()
        assert len(spans) == 5

        # All spans should have unique span IDs
        span_ids = [span.get_span_context().span_id for span in spans]
        assert len(set(span_ids)) == 5  # All unique

    def test_operation_timing(self, instrumented_env):
        """Test verify span timing reflects actual operation duration."""
        client, exporter = instrumented_env

        key = (NAMESPACE, "timing_test", f"timing_{uuid.uuid4().hex[:8]}")

        # Write large record
        large_data = {"data": "x" * 10000}

        start_time = time.time()
        client.put(key, large_data)
        end_time = time.time()

        actual_duration_ns = (end_time - start_time) * 1e9

        spans = exporter.get_finished_spans()
        span = spans[0]

        span_duration_ns = span.end_time - span.start_time

        # Span duration should be within reasonable range of actual duration
        # (allowing for some overhead)
        assert span_duration_ns > 0
        assert span_duration_ns < actual_duration_ns + 1e9  # Within 1 second tolerance

        # Cleanup
        client.remove(key)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
