# OpenTelemetry Aerospike Instrumentation

[![PyPI version](https://badge.fury.io/py/opentelemetry-instrumentation-aerospike.svg)](https://badge.fury.io/py/opentelemetry-instrumentation-aerospike)
[![Python Version](https://img.shields.io/pypi/pyversions/opentelemetry-instrumentation-aerospike.svg)](https://pypi.org/project/opentelemetry-instrumentation-aerospike/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

OpenTelemetry instrumentation for the [Aerospike Python Client](https://github.com/aerospike/aerospike-client-python).

This library enables automatic tracing of Aerospike database operations, providing visibility into your application's database interactions.

## Installation

```bash
pip install opentelemetry-instrumentation-aerospike
```

ㅣ
## Requirements

- Python >= 3.9
- aerospike >= 17.2.0
- opentelemetry-api >= 1.12
- opentelemetry-instrumentation >= 0.40b0

## Usage

### Basic Usage

```python
import aerospike
from opentelemetry.instrumentation.aerospike import AerospikeInstrumentor

# Instrument Aerospike BEFORE creating any clients
AerospikeInstrumentor().instrument()

# Create and use client as normal - all operations will be traced
config = {'hosts': [('127.0.0.1', 3000)]}
client = aerospike.client(config)
client.connect()

# Operations are now automatically traced
client.put(('test', 'demo', 'key1'), {'name': 'John', 'age': 30})
key, meta, bins = client.get(('test', 'demo', 'key1'))

client.close()
```

### With Custom Tracer Provider

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

# Setup tracer provider
provider = TracerProvider()
processor = BatchSpanProcessor(OTLPSpanExporter())
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)

# Instrument with custom tracer provider
AerospikeInstrumentor().instrument(tracer_provider=provider)
```

### Using Hooks

You can use hooks to customize span attributes or add custom logic:

```python
def request_hook(span, operation, args, kwargs):
    """Called before each database operation."""
    span.set_attribute("custom.request.id", get_request_id())

def response_hook(span, operation, result):
    """Called after a successful operation."""
    if isinstance(result, tuple) and len(result) >= 3:
        span.set_attribute("custom.record.bins", len(result[2]))

def error_hook(span, operation, exception):
    """Called when an operation fails."""
    span.set_attribute("custom.error.code", getattr(exception, 'code', -1))

AerospikeInstrumentor().instrument(
    request_hook=request_hook,
    response_hook=response_hook,
    error_hook=error_hook
)
```

### Capturing Record Keys

By default, record keys are not captured for security reasons. To enable:

```python
AerospikeInstrumentor().instrument(capture_key=True)
```

⚠️ **Security Warning**: Only enable key capture in development or when keys don't contain sensitive information.

### Uninstrumenting

```python
instrumentor = AerospikeInstrumentor()
instrumentor.instrument()

# ... use client ...

# Remove instrumentation
instrumentor.uninstrument()
```

## Span Attributes

The instrumentation follows [OpenTelemetry Semantic Conventions for Database](https://opentelemetry.io/docs/specs/semconv/database/database-spans/).

### Standard Attributes

| Attribute | Description | Example |
|-----------|-------------|---------|
| `db.system` | Database system identifier | `aerospike` |
| `db.namespace` | Aerospike namespace | `test` |
| `db.collection.name` | Aerospike set name | `users` |
| `db.operation.name` | Operation name | `PUT`, `GET`, `QUERY` |
| `db.operation.batch.size` | Batch operation size | `100` |
| `server.address` | Server hostname/IP | `127.0.0.1` |
| `server.port` | Server port | `3000` |

### Error Attributes

| Attribute | Description | Example |
|-----------|-------------|---------|
| `error.type` | Exception class name | `RecordNotFound` |
| `db.response.status_code` | Aerospike error code | `2` |

### Aerospike-Specific Attributes

| Attribute | Description | Example |
|-----------|-------------|---------|
| `db.aerospike.key` | Record key (if enabled) | `user123` |
| `db.aerospike.generation` | Record generation | `5` |
| `db.aerospike.ttl` | Record TTL in seconds | `86400` |
| `db.aerospike.udf.module` | UDF module name | `mymodule` |
| `db.aerospike.udf.function` | UDF function name | `myfunction` |

## Span Naming

Spans are named following the convention: `{OPERATION} {namespace}.{set}`

Examples:
- `PUT test.users`
- `GET production.orders`
- `BATCH GET test.demo`
- `QUERY test.events`
- `SCAN production.logs`

## Supported Operations

### Single Record Operations
- `put`, `get`, `select`, `exists`, `remove`, `touch`
- `operate`, `append`, `prepend`, `increment`

### Batch Operations
- `batch_read`, `batch_write`, `batch_operate`, `batch_remove`, `batch_apply`

### Query/Scan Operations
- `query`, `scan`

### UDF Operations
- `apply`, `scan_apply`, `query_apply`

### Admin Operations
- `truncate`, `info_all`

## Async Pattern Support

The instrumentation works with async wrapper patterns commonly used in production:

```python
import asyncio
import functools
from concurrent.futures import ThreadPoolExecutor

class AsyncAerospikeClient:
    def __init__(self, client):
        self._client = client
        self._executor = ThreadPoolExecutor(max_workers=10)
    
    def __getattr__(self, attr):
        async def method(*args, **kwargs):
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                self._executor,
                functools.partial(getattr(self._client, attr), *args, **kwargs)
            )
        return method

# Instrument before creating clients
AerospikeInstrumentor().instrument()

# Create sync client (instrumented)
config = {'hosts': [('127.0.0.1', 3000)]}
sync_client = aerospike.client(config)
sync_client.connect()

# Wrap with async pattern
async_client = AsyncAerospikeClient(sync_client)

# Async operations are traced!
async def main():
    await async_client.put(('test', 'demo', 'key1'), {'data': 'value'})
    result = await async_client.get(('test', 'demo', 'key1'))
```




## Development

### Setup

```bash
# Clone repository
git clone https://github.com/your-repo/opentelemetry-instrumentation-aerospike.git
cd opentelemetry-instrumentation-aerospike

# Create virtual environment
uv venv
source .venv/bin/activate

# Install with dev dependencies
uv pip install -e ".[dev]"
```

### Running Tests

```bash
# Start Aerospike (Docker)
docker run -d --name aerospike-test -p 3000:3000 aerospike/aerospike-server:latest

# Run integration tests
pytest tests/test_aerospike_integration.py -v

# Run with coverage
pytest --cov=opentelemetry.instrumentation.aerospike --cov-report=html
```

### Code Quality

```bash
# Format code
black src tests
isort src tests

# Lint
ruff check src tests

# Type check
mypy src
```

## License

Apache License 2.0

## Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) for details.

## Links

- [OpenTelemetry Python](https://github.com/open-telemetry/opentelemetry-python)
- [OpenTelemetry Python Contrib](https://github.com/open-telemetry/opentelemetry-python-contrib)
- [Aerospike Python Client](https://github.com/aerospike/aerospike-client-python)
- [OpenTelemetry Database Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/database/)

