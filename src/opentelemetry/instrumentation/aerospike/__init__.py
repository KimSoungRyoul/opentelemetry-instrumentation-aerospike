# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""OpenTelemetry Aerospike Instrumentation.

This library allows tracing Aerospike database operations using OpenTelemetry.

Usage
-----

.. code:: python

    import aerospike
    from opentelemetry.instrumentation.aerospike import AerospikeInstrumentor

    AerospikeInstrumentor().instrument()

    config = {'hosts': [('127.0.0.1', 3000)]}
    client = aerospike.client(config)
    client.connect()

    # All subsequent operations will be traced
    client.put(('test', 'demo', 'key1'), {'bin1': 'value1'})
    (key, meta, bins) = client.get(('test', 'demo', 'key1'))

API
---

The ``instrument()`` method accepts the following keyword arguments:

tracer_provider (TracerProvider)
    Optional tracer provider to use. If not provided, the global tracer provider is used.

request_hook (Callable)
    A function called before each database operation.
    Signature: ``def request_hook(span: Span, operation: str, args: tuple, kwargs: dict) -> None``

response_hook (Callable)
    A function called after a successful database operation.
    Signature: ``def response_hook(span: Span, operation: str, result: Any) -> None``

error_hook (Callable)
    A function called when a database operation fails.
    Signature: ``def error_hook(span: Span, operation: str, exception: Exception) -> None``

capture_key (bool)
    Whether to capture the record key in span attributes. Default: False (for security)
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Collection
from typing import Any

from wrapt import wrap_function_wrapper

from opentelemetry import trace
from opentelemetry.instrumentation.aerospike.package import _instruments
from opentelemetry.instrumentation.aerospike.version import __version__
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.trace import Span, SpanKind, Status, StatusCode, Tracer

# Semantic convention constants
_DB_SYSTEM = "aerospike"
_DB_SYSTEM_ATTR = "db.system"
_DB_NAMESPACE_ATTR = "db.namespace"
_DB_COLLECTION_NAME_ATTR = "db.collection.name"
_DB_OPERATION_NAME_ATTR = "db.operation.name"
_DB_OPERATION_BATCH_SIZE_ATTR = "db.operation.batch.size"
_DB_RESPONSE_STATUS_CODE_ATTR = "db.response.status_code"
_SERVER_ADDRESS_ATTR = "server.address"
_SERVER_PORT_ATTR = "server.port"
_ERROR_TYPE_ATTR = "error.type"

# Aerospike-specific attributes
_DB_AEROSPIKE_KEY_ATTR = "db.aerospike.key"
_DB_AEROSPIKE_GENERATION_ATTR = "db.aerospike.generation"
_DB_AEROSPIKE_TTL_ATTR = "db.aerospike.ttl"


class AerospikeInstrumentor(BaseInstrumentor):
    """OpenTelemetry Aerospike Instrumentor.

    This instrumentor wraps Aerospike client methods to automatically
    create spans for database operations.

    Note: Aerospike Python client is a C extension, so we wrap the client
    factory function (aerospike.client) to instrument each client instance.
    """

    # Methods to instrument
    _SINGLE_RECORD_METHODS = [
        "put",
        "get",
        "select",
        "exists",
        "remove",
        "touch",
        "operate",
        "append",
        "prepend",
        "increment",
    ]

    _BATCH_METHODS = [
        "batch_read",
        "batch_write",
        "batch_operate",
        "batch_remove",
        "batch_apply",
        "get_many",
        "exists_many",
        "select_many",
    ]

    _QUERY_SCAN_METHODS = ["query", "scan"]

    _UDF_METHODS = ["apply", "scan_apply", "query_apply"]

    _ADMIN_METHODS = ["truncate", "info_all"]

    _original_client = None

    def instrumentation_dependencies(self) -> Collection[str]:
        """Return the dependencies required for this instrumentation."""
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        """Instrument Aerospike client factory function."""
        import aerospike

        tracer_provider = kwargs.get("tracer_provider")
        tracer = trace.get_tracer(
            __name__,
            __version__,
            tracer_provider,
            schema_url="https://opentelemetry.io/schemas/1.28.0",
        )

        request_hook = kwargs.get("request_hook")
        response_hook = kwargs.get("response_hook")
        error_hook = kwargs.get("error_hook")
        capture_key = kwargs.get("capture_key", False)

        # Store original client function
        self._original_client = aerospike.client

        # Wrap the client factory function
        wrap_function_wrapper(
            "aerospike",
            "client",
            _create_client_wrapper(tracer, request_hook, response_hook, error_hook, capture_key),
        )

    def _uninstrument(self, **kwargs: Any) -> None:
        """Remove instrumentation from Aerospike client factory."""
        import aerospike

        unwrap(aerospike, "client")


def _create_client_wrapper(
    tracer: Tracer,
    request_hook: Callable | None,
    response_hook: Callable | None,
    error_hook: Callable | None,
    capture_key: bool,
) -> Callable:
    """Create a wrapper for aerospike.client() factory function."""

    def client_wrapper(wrapped: Callable, instance: Any, args: tuple, kwargs: dict) -> Any:
        # Create the original client
        client = wrapped(*args, **kwargs)

        # Wrap the client instance with our instrumented proxy
        return InstrumentedAerospikeClient(
            client, tracer, request_hook, response_hook, error_hook, capture_key
        )

    return client_wrapper


class InstrumentedAerospikeClient:
    """Instrumented wrapper for Aerospike Client.

    This class wraps an Aerospike client instance and adds
    OpenTelemetry tracing to all database operations.
    """

    _SINGLE_RECORD_METHODS = [
        "put",
        "get",
        "select",
        "exists",
        "remove",
        "touch",
        "operate",
        "append",
        "prepend",
        "increment",
    ]

    _BATCH_METHODS = [
        "batch_read",
        "batch_write",
        "batch_operate",
        "batch_remove",
        "batch_apply",
        "get_many",
        "exists_many",
        "select_many",
    ]

    _QUERY_SCAN_METHODS = ["query", "scan"]

    _UDF_METHODS = ["apply", "scan_apply", "query_apply"]

    _ADMIN_METHODS = ["truncate", "info_all"]

    def __init__(
        self,
        client: Any,
        tracer: Tracer,
        request_hook: Callable | None,
        response_hook: Callable | None,
        error_hook: Callable | None,
        capture_key: bool,
    ):
        self._client = client
        self._tracer = tracer
        self._request_hook = request_hook
        self._response_hook = response_hook
        self._error_hook = error_hook
        self._capture_key = capture_key

        # Store hosts config for connection attributes
        self._hosts = None

    def __getattr__(self, name: str) -> Any:
        """Proxy attribute access to the wrapped client."""
        attr = getattr(self._client, name)

        # If it's a method we want to instrument, wrap it
        if callable(attr):
            if name in self._SINGLE_RECORD_METHODS:
                return self._wrap_single_record_method(attr, name.upper())
            elif name in self._BATCH_METHODS:
                op_name = _get_batch_operation_name(name)
                return self._wrap_batch_method(attr, op_name)
            elif name in self._QUERY_SCAN_METHODS:
                return self._wrap_query_scan_method(attr, name.upper())
            elif name in self._UDF_METHODS:
                op_name = name.upper().replace("_", " ")
                return self._wrap_udf_method(attr, op_name)
            elif name in self._ADMIN_METHODS:
                return self._wrap_admin_method(attr, name.upper())

        return attr

    def connect(self, *args, **kwargs) -> InstrumentedAerospikeClient:
        """Connect to the Aerospike cluster."""
        self._client.connect(*args, **kwargs)
        return self

    def close(self) -> None:
        """Close the connection."""
        self._client.close()

    def is_connected(self) -> bool:
        """Check if connected."""
        return self._client.is_connected()

    def _get_hosts(self) -> list | None:
        """Get hosts from client configuration."""
        if self._hosts is None:
            try:
                # Try to get hosts from shm_key or other means
                # The config might be accessible through different attributes
                if hasattr(self._client, "config"):
                    config = self._client.config
                    if callable(config):
                        config = config()
                    self._hosts = config.get("hosts", [])
            except Exception:
                pass
        return self._hosts

    def _set_connection_attributes(self, span: Span) -> None:
        """Set connection-related attributes on span."""
        hosts = self._get_hosts()
        if hosts:
            try:
                first_host = hosts[0]
                if isinstance(first_host, tuple):
                    host = first_host[0]
                    port = first_host[1] if len(first_host) > 1 else 3000
                else:
                    host = first_host
                    port = 3000
                span.set_attribute(_SERVER_ADDRESS_ATTR, str(host))
                span.set_attribute(_SERVER_PORT_ATTR, int(port))
            except Exception:
                pass

    def _wrap_single_record_method(self, method: Callable, operation: str) -> Callable:
        """Wrap a single record operation method."""

        @functools.wraps(method)
        def wrapper(*args, **kwargs) -> Any:
            key_tuple = args[0] if args else None
            namespace, set_name = _extract_namespace_set_from_key(key_tuple)

            span_name = _generate_span_name(operation, namespace, set_name)

            with self._tracer.start_as_current_span(
                span_name,
                kind=SpanKind.CLIENT,
            ) as span:
                if span.is_recording():
                    span.set_attribute(_DB_SYSTEM_ATTR, _DB_SYSTEM)

                    if namespace:
                        span.set_attribute(_DB_NAMESPACE_ATTR, namespace)
                    if set_name:
                        span.set_attribute(_DB_COLLECTION_NAME_ATTR, set_name)

                    span.set_attribute(_DB_OPERATION_NAME_ATTR, operation)
                    self._set_connection_attributes(span)

                    # Optional: capture key
                    if self._capture_key and key_tuple and len(key_tuple) > 2:
                        user_key = key_tuple[2]
                        if user_key is not None:
                            span.set_attribute(_DB_AEROSPIKE_KEY_ATTR, str(user_key))

                # Request hook
                if self._request_hook:
                    self._request_hook(span, operation, args, kwargs)

                try:
                    result = method(*args, **kwargs)

                    # Response hook
                    if self._response_hook:
                        self._response_hook(span, operation, result)

                    # Set generation/TTL from result
                    if span.is_recording():
                        _set_result_attributes(span, result)

                    return result

                except Exception as exc:
                    if span.is_recording():
                        _set_error_attributes(span, exc)

                    if self._error_hook:
                        self._error_hook(span, operation, exc)

                    raise

        return wrapper

    def _wrap_batch_method(self, method: Callable, operation: str) -> Callable:
        """Wrap a batch operation method."""

        @functools.wraps(method)
        def wrapper(*args, **kwargs) -> Any:
            keys = args[0] if args else None
            namespace, set_name = _extract_namespace_set_from_batch(keys)

            span_name = _generate_span_name(operation, namespace, set_name)

            with self._tracer.start_as_current_span(
                span_name,
                kind=SpanKind.CLIENT,
            ) as span:
                if span.is_recording():
                    span.set_attribute(_DB_SYSTEM_ATTR, _DB_SYSTEM)

                    if namespace:
                        span.set_attribute(_DB_NAMESPACE_ATTR, namespace)
                    if set_name:
                        span.set_attribute(_DB_COLLECTION_NAME_ATTR, set_name)

                    span.set_attribute(_DB_OPERATION_NAME_ATTR, operation)
                    self._set_connection_attributes(span)

                    # Batch size
                    if keys and isinstance(keys, list | tuple):
                        span.set_attribute(_DB_OPERATION_BATCH_SIZE_ATTR, len(keys))

                if self._request_hook:
                    self._request_hook(span, operation, args, kwargs)

                try:
                    result = method(*args, **kwargs)

                    if self._response_hook:
                        self._response_hook(span, operation, result)

                    return result

                except Exception as exc:
                    if span.is_recording():
                        _set_error_attributes(span, exc)

                    if self._error_hook:
                        self._error_hook(span, operation, exc)

                    raise

        return wrapper

    def _wrap_query_scan_method(self, method: Callable, operation: str) -> Callable:
        """Wrap a query/scan operation method."""

        @functools.wraps(method)
        def wrapper(*args, **kwargs) -> Any:
            namespace = args[0] if args else None
            set_name = args[1] if len(args) > 1 else None

            span_name = _generate_span_name(operation, namespace, set_name)

            with self._tracer.start_as_current_span(
                span_name,
                kind=SpanKind.CLIENT,
            ) as span:
                if span.is_recording():
                    span.set_attribute(_DB_SYSTEM_ATTR, _DB_SYSTEM)

                    if namespace:
                        span.set_attribute(_DB_NAMESPACE_ATTR, namespace)
                    if set_name:
                        span.set_attribute(_DB_COLLECTION_NAME_ATTR, set_name)

                    span.set_attribute(_DB_OPERATION_NAME_ATTR, operation)
                    self._set_connection_attributes(span)

                if self._request_hook:
                    self._request_hook(span, operation, args, kwargs)

                try:
                    result = method(*args, **kwargs)

                    if self._response_hook:
                        self._response_hook(span, operation, result)

                    return result

                except Exception as exc:
                    if span.is_recording():
                        _set_error_attributes(span, exc)

                    if self._error_hook:
                        self._error_hook(span, operation, exc)

                    raise

        return wrapper

    def _wrap_udf_method(self, method: Callable, operation: str) -> Callable:
        """Wrap a UDF operation method."""

        @functools.wraps(method)
        def wrapper(*args, **kwargs) -> Any:
            key_tuple = args[0] if args else None
            namespace, set_name = _extract_namespace_set_from_key(key_tuple)

            span_name = _generate_span_name(operation, namespace, set_name)

            with self._tracer.start_as_current_span(
                span_name,
                kind=SpanKind.CLIENT,
            ) as span:
                if span.is_recording():
                    span.set_attribute(_DB_SYSTEM_ATTR, _DB_SYSTEM)

                    if namespace:
                        span.set_attribute(_DB_NAMESPACE_ATTR, namespace)
                    if set_name:
                        span.set_attribute(_DB_COLLECTION_NAME_ATTR, set_name)

                    span.set_attribute(_DB_OPERATION_NAME_ATTR, operation)
                    self._set_connection_attributes(span)

                    # UDF info
                    if len(args) > 1:
                        span.set_attribute("db.aerospike.udf.module", str(args[1]))
                    if len(args) > 2:
                        span.set_attribute("db.aerospike.udf.function", str(args[2]))

                    # Optional: capture key
                    if self._capture_key and key_tuple and len(key_tuple) > 2:
                        user_key = key_tuple[2]
                        if user_key is not None:
                            span.set_attribute(_DB_AEROSPIKE_KEY_ATTR, str(user_key))

                if self._request_hook:
                    self._request_hook(span, operation, args, kwargs)

                try:
                    result = method(*args, **kwargs)

                    if self._response_hook:
                        self._response_hook(span, operation, result)

                    return result

                except Exception as exc:
                    if span.is_recording():
                        _set_error_attributes(span, exc)

                    if self._error_hook:
                        self._error_hook(span, operation, exc)

                    raise

        return wrapper

    def _wrap_admin_method(self, method: Callable, operation: str) -> Callable:
        """Wrap an admin operation method."""

        @functools.wraps(method)
        def wrapper(*args, **kwargs) -> Any:
            namespace = args[0] if args and isinstance(args[0], str) else None
            set_name = args[1] if len(args) > 1 and isinstance(args[1], str) else None

            span_name = _generate_span_name(operation, namespace, set_name)

            with self._tracer.start_as_current_span(
                span_name,
                kind=SpanKind.CLIENT,
            ) as span:
                if span.is_recording():
                    span.set_attribute(_DB_SYSTEM_ATTR, _DB_SYSTEM)

                    if namespace:
                        span.set_attribute(_DB_NAMESPACE_ATTR, namespace)
                    if set_name:
                        span.set_attribute(_DB_COLLECTION_NAME_ATTR, set_name)

                    span.set_attribute(_DB_OPERATION_NAME_ATTR, operation)
                    self._set_connection_attributes(span)

                if self._request_hook:
                    self._request_hook(span, operation, args, kwargs)

                try:
                    result = method(*args, **kwargs)

                    if self._response_hook:
                        self._response_hook(span, operation, result)

                    return result

                except Exception as exc:
                    if span.is_recording():
                        _set_error_attributes(span, exc)

                    if self._error_hook:
                        self._error_hook(span, operation, exc)

                    raise

        return wrapper


# Helper functions


def _get_batch_operation_name(method: str) -> str:
    """Convert batch method name to operation name."""
    method_upper = method.upper()
    if method_upper.startswith("BATCH_"):
        return f"BATCH {method_upper[6:]}"
    elif method_upper.endswith("_MANY"):
        return f"BATCH {method_upper[:-5]}"
    return f"BATCH {method_upper}"


def _extract_namespace_set_from_key(key_tuple: tuple | None) -> tuple[str | None, str | None]:
    """Extract namespace and set from a single key tuple.

    Key format: (namespace, set, key[, digest])
    """
    if not key_tuple or not isinstance(key_tuple, tuple):
        return None, None

    namespace = key_tuple[0] if len(key_tuple) > 0 else None
    set_name = key_tuple[1] if len(key_tuple) > 1 else None
    return namespace, set_name


def _extract_namespace_set_from_batch(keys: list | tuple | None) -> tuple[str | None, str | None]:
    """Extract namespace and set from batch keys (uses first key)."""
    if not keys or not isinstance(keys, list | tuple):
        return None, None

    first_key = keys[0]
    if isinstance(first_key, tuple) and len(first_key) >= 2:
        return first_key[0], first_key[1]
    return None, None


def _generate_span_name(operation: str, namespace: str | None, set_name: str | None) -> str:
    """Generate span name following convention: {operation} {namespace}.{set}."""
    if namespace and set_name:
        return f"{operation} {namespace}.{set_name}"
    elif namespace:
        return f"{operation} {namespace}"
    return operation


def _set_result_attributes(span: Span, result: Any) -> None:
    """Set attributes from operation result."""
    if isinstance(result, tuple) and len(result) >= 2:
        # Format: (key, meta, bins) or (key, meta)
        meta = result[1] if len(result) > 1 else None
        if isinstance(meta, dict):
            if "gen" in meta:
                span.set_attribute(_DB_AEROSPIKE_GENERATION_ATTR, meta["gen"])
            if "ttl" in meta:
                span.set_attribute(_DB_AEROSPIKE_TTL_ATTR, meta["ttl"])


def _set_error_attributes(span: Span, exc: Exception) -> None:
    """Set error attributes on span."""
    span.set_status(Status(StatusCode.ERROR, str(exc)))
    span.set_attribute(_ERROR_TYPE_ATTR, type(exc).__name__)

    # Aerospike specific error code
    if hasattr(exc, "code"):
        span.set_attribute(_DB_RESPONSE_STATUS_CODE_ATTR, str(exc.code))


# Public API
__all__ = [
    "AerospikeInstrumentor",
    "InstrumentedAerospikeClient",
    "__version__",
]
