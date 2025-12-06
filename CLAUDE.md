# OpenTelemetry Instrumentation for Aerospike 개발 계획서

## 1. 프로젝트 개요

### 1.1 목적
Aerospike Python Client(`aerospike`)에 대한 OpenTelemetry instrumentation을 개발하여 `opentelemetry-python-contrib` 저장소에 기여한다. 이를 통해 Aerospike 데이터베이스 작업에 대한 분산 추적(distributed tracing)과 메트릭 수집을 가능하게 한다.

### 1.2 범위
- Aerospike Python Client의 주요 데이터베이스 작업에 대한 자동 계측(auto-instrumentation)
- OpenTelemetry Database Semantic Conventions 준수
- 동기(sync) API 지원 (비동기 API는 향후 확장)

### 1.3 참조 문서
- [OpenTelemetry Semantic Conventions for Database](https://opentelemetry.io/docs/specs/semconv/database/database-spans/)
- [opentelemetry-python-contrib Repository](https://github.com/open-telemetry/opentelemetry-python-contrib)
- [Aerospike Python Client Documentation](https://aerospike-python-client.readthedocs.io/)

---

## 2. 기술 분석

### 2.1 Aerospike Python Client 분석

#### 2.1.1 계측 대상 메서드

**Single Record Operations**
| 메서드 | 설명 | Operation Name |
|--------|------|----------------|
| `put` | 레코드 생성/갱신 | `PUT` |
| `get` | 레코드 조회 | `GET` |
| `select` | 특정 bin만 조회 | `SELECT` |
| `exists` | 레코드 존재 여부 확인 | `EXISTS` |
| `remove` | 레코드 삭제 | `REMOVE` |
| `touch` | 레코드 TTL 갱신 | `TOUCH` |
| `operate` | 복합 연산 | `OPERATE` |
| `append` | 문자열 append | `APPEND` |
| `prepend` | 문자열 prepend | `PREPEND` |
| `increment` | 숫자 증가 | `INCREMENT` |

**Batch Operations**
| 메서드 | 설명 | Operation Name |
|--------|------|----------------|
| `batch_read` | 다중 레코드 조회 | `BATCH READ` |
| `batch_write` | 다중 레코드 쓰기 | `BATCH WRITE` |
| `batch_operate` | 다중 레코드 연산 | `BATCH OPERATE` |
| `batch_remove` | 다중 레코드 삭제 | `BATCH REMOVE` |
| `batch_apply` | 다중 레코드 UDF 적용 | `BATCH APPLY` |
| `get_many` | 다중 레코드 조회 (legacy) | `BATCH GET` |
| `exists_many` | 다중 존재 확인 | `BATCH EXISTS` |
| `select_many` | 다중 선택 조회 | `BATCH SELECT` |

**Query & Scan Operations**
| 메서드 | 설명 | Operation Name |
|--------|------|----------------|
| `query` | Secondary Index 쿼리 | `QUERY` |
| `scan` | 전체 스캔 | `SCAN` |

**UDF Operations**
| 메서드 | 설명 | Operation Name |
|--------|------|----------------|
| `apply` | UDF 실행 | `APPLY` |
| `scan_apply` | 스캔 + UDF | `SCAN APPLY` |
| `query_apply` | 쿼리 + UDF | `QUERY APPLY` |

**Admin Operations** (선택적 계측)
| 메서드 | 설명 | Operation Name |
|--------|------|----------------|
| `truncate` | Set truncate | `TRUNCATE` |
| `info_all` | 클러스터 정보 조회 | `INFO` |

#### 2.1.2 Aerospike 데이터 모델 용어 매핑

| Aerospike 용어 | OpenTelemetry Semantic Convention |
|---------------|-----------------------------------|
| Namespace | `db.namespace` |
| Set | `db.collection.name` |
| Key | - (digest로 변환됨) |
| Bin | - (컬럼에 해당) |

### 2.2 Semantic Conventions 매핑

#### 2.2.1 필수 Attributes

```python
# Required
"db.system.name": "aerospike"  # Well-known identifier에 추가 필요

# Conditionally Required
"db.namespace": "<namespace>"      # Aerospike namespace
"db.collection.name": "<set>"      # Aerospike set
"db.operation.name": "<operation>" # PUT, GET, QUERY 등
"server.address": "<host>"
"server.port": <port>

# Recommended
"db.query.summary": "<operation> <set>"
"network.peer.address": "<node_address>"
"network.peer.port": <node_port>
```

#### 2.2.2 Aerospike 전용 Attributes (Experimental)

```python
# Aerospike-specific attributes
"db.aerospike.key": "<user_key>"           # 사용자 키 (선택적, 보안 고려)
"db.aerospike.generation": <generation>     # 레코드 버전
"db.aerospike.ttl": <ttl>                   # Time-to-live
"db.aerospike.batch.size": <size>          # 배치 크기
```

#### 2.2.3 Span Naming Convention

```
# Single record operations
{db.operation.name} {db.namespace}.{db.collection.name}
예: "GET test.users", "PUT production.orders"

# Batch operations  
BATCH {db.operation.name} {db.namespace}
예: "BATCH GET test", "BATCH WRITE production"

# Query/Scan operations
{db.operation.name} {db.namespace}.{db.collection.name}
예: "QUERY test.users", "SCAN test.logs"
```

### 2.3 Error Handling

```python
# Aerospike 에러 코드를 db.response.status_code로 매핑
"db.response.status_code": str(exception.code)  # e.g., "2" (KEY_NOT_FOUND)
"error.type": exception.__class__.__name__       # e.g., "RecordNotFound"
```

주요 Aerospike 에러 코드:
- `0`: OK
- `1`: SERVER_ERROR  
- `2`: KEY_NOT_FOUND_ERROR
- `3`: GENERATION_ERROR
- `4`: PARAMETER_ERROR
- `5`: KEY_EXISTS_ERROR
- `-1`: CLIENT_ERROR
- `-10`: TIMEOUT

---

## 3. 프로젝트 구조

### 3.1 디렉토리 구조

```
instrumentation/opentelemetry-instrumentation-aerospike/
├── src/
│   └── opentelemetry/
│       └── instrumentation/
│           └── aerospike/
│               ├── __init__.py          # AerospikeInstrumentor 클래스
│               ├── _instruments.py      # 의존성 정의
│               ├── package.py           # 패키지 메타데이터
│               ├── version.py           # 버전 정보
│               └── utils.py             # 유틸리티 함수
├── tests/
│   ├── __init__.py
│   ├── test_aerospike_instrumentation.py    # 단위 테스트
│   └── test_aerospike_integration.py        # 통합 테스트
├── pyproject.toml
├── README.rst
├── LICENSE
└── CHANGELOG.md
```

### 3.2 핵심 모듈 설계

#### 3.2.1 `__init__.py` - AerospikeInstrumentor

```python
"""
OpenTelemetry Aerospike Instrumentation
=======================================

Usage
-----

.. code:: python

    from aerospike import client
    from opentelemetry.instrumentation.aerospike import AerospikeInstrumentor

    AerospikeInstrumentor().instrument()
    
    config = {'hosts': [('127.0.0.1', 3000)]}
    client = aerospike.client(config)
    
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
    Whether to capture the record key in span attributes. Default: False (보안 고려)

"""

from typing import Any, Callable, Collection, Optional
from wrapt import wrap_function_wrapper

from opentelemetry import trace
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.aerospike.package import _instruments
from opentelemetry.instrumentation.aerospike.version import __version__
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.semconv.trace import SpanAttributes
from opentelemetry.trace import Span, SpanKind, Status, StatusCode


class AerospikeInstrumentor(BaseInstrumentor):
    """OpenTelemetry Aerospike Instrumentor"""
    
    _DB_SYSTEM = "aerospike"
    
    # 계측 대상 메서드 목록
    _SINGLE_RECORD_METHODS = [
        "put", "get", "select", "exists", "remove", "touch",
        "operate", "append", "prepend", "increment"
    ]
    
    _BATCH_METHODS = [
        "batch_read", "batch_write", "batch_operate", "batch_remove",
        "batch_apply", "get_many", "exists_many", "select_many"
    ]
    
    _QUERY_SCAN_METHODS = ["query", "scan"]
    
    _UDF_METHODS = ["apply", "scan_apply", "query_apply"]
    
    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments
    
    def _instrument(self, **kwargs: Any) -> None:
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
        
        # Wrap Client methods
        self._wrap_client_methods(
            tracer, request_hook, response_hook, error_hook, capture_key
        )
    
    def _uninstrument(self, **kwargs: Any) -> None:
        import aerospike
        
        all_methods = (
            self._SINGLE_RECORD_METHODS + 
            self._BATCH_METHODS + 
            self._QUERY_SCAN_METHODS + 
            self._UDF_METHODS
        )
        
        for method in all_methods:
            unwrap(aerospike.Client, method)
    
    def _wrap_client_methods(
        self,
        tracer: trace.Tracer,
        request_hook: Optional[Callable],
        response_hook: Optional[Callable],
        error_hook: Optional[Callable],
        capture_key: bool
    ) -> None:
        """Wrap all Aerospike Client methods for instrumentation"""
        import aerospike
        
        # Single record operations
        for method in self._SINGLE_RECORD_METHODS:
            wrap_function_wrapper(
                aerospike.Client,
                method,
                self._create_wrapper(
                    tracer, method.upper(), 
                    request_hook, response_hook, error_hook,
                    capture_key, is_batch=False
                )
            )
        
        # Batch operations
        for method in self._BATCH_METHODS:
            op_name = f"BATCH {method.upper().replace('BATCH_', '').replace('_MANY', '')}"
            wrap_function_wrapper(
                aerospike.Client,
                method,
                self._create_wrapper(
                    tracer, op_name,
                    request_hook, response_hook, error_hook,
                    capture_key, is_batch=True
                )
            )
        
        # Query/Scan - Query와 Scan은 iterator를 반환하므로 별도 처리
        # (실제 구현에서는 Query/Scan 클래스도 wrap 필요)
        for method in self._QUERY_SCAN_METHODS:
            wrap_function_wrapper(
                aerospike.Client,
                method,
                self._create_query_scan_wrapper(
                    tracer, method.upper(),
                    request_hook, response_hook, error_hook
                )
            )
    
    def _create_wrapper(
        self,
        tracer: trace.Tracer,
        operation: str,
        request_hook: Optional[Callable],
        response_hook: Optional[Callable],
        error_hook: Optional[Callable],
        capture_key: bool,
        is_batch: bool
    ) -> Callable:
        """Create a wrapper function for Aerospike operations"""
        
        def wrapper(wrapped, instance, args, kwargs):
            # Extract key information
            key_tuple = args[0] if args else None
            namespace, set_name = self._extract_namespace_set(key_tuple, is_batch)
            
            # Generate span name
            target = f"{namespace}.{set_name}" if set_name else namespace
            span_name = f"{operation} {target}" if target else operation
            
            with tracer.start_as_current_span(
                span_name,
                kind=SpanKind.CLIENT,
            ) as span:
                if span.is_recording():
                    # Set standard database attributes
                    span.set_attribute(SpanAttributes.DB_SYSTEM, self._DB_SYSTEM)
                    
                    if namespace:
                        span.set_attribute(SpanAttributes.DB_NAMESPACE, namespace)
                    if set_name:
                        span.set_attribute(SpanAttributes.DB_COLLECTION_NAME, set_name)
                    
                    span.set_attribute(SpanAttributes.DB_OPERATION_NAME, operation)
                    
                    # Set connection info from client
                    self._set_connection_attributes(span, instance)
                    
                    # Batch size for batch operations
                    if is_batch and args:
                        batch_size = len(args[0]) if isinstance(args[0], (list, tuple)) else 1
                        span.set_attribute("db.operation.batch.size", batch_size)
                    
                    # Optional: capture key
                    if capture_key and key_tuple and not is_batch:
                        user_key = key_tuple[2] if len(key_tuple) > 2 else None
                        if user_key:
                            span.set_attribute("db.aerospike.key", str(user_key))
                
                # Request hook
                if request_hook:
                    request_hook(span, operation, args, kwargs)
                
                try:
                    result = wrapped(*args, **kwargs)
                    
                    # Response hook
                    if response_hook:
                        response_hook(span, operation, result)
                    
                    # Set generation/TTL from result if available
                    if span.is_recording():
                        self._set_result_attributes(span, result)
                    
                    return result
                    
                except Exception as exc:
                    if span.is_recording():
                        self._set_error_attributes(span, exc)
                    
                    if error_hook:
                        error_hook(span, operation, exc)
                    
                    raise
        
        return wrapper
    
    def _extract_namespace_set(
        self, 
        key_tuple: Optional[tuple], 
        is_batch: bool
    ) -> tuple[Optional[str], Optional[str]]:
        """Extract namespace and set from key tuple"""
        if not key_tuple:
            return None, None
        
        if is_batch:
            # Batch operations: first element of list
            if isinstance(key_tuple, (list, tuple)) and key_tuple:
                first_key = key_tuple[0]
                if isinstance(first_key, tuple) and len(first_key) >= 2:
                    return first_key[0], first_key[1]
            return None, None
        
        # Single record: (namespace, set, key[, digest])
        if isinstance(key_tuple, tuple):
            namespace = key_tuple[0] if len(key_tuple) > 0 else None
            set_name = key_tuple[1] if len(key_tuple) > 1 else None
            return namespace, set_name
        
        return None, None
    
    def _set_connection_attributes(self, span: Span, client) -> None:
        """Set connection-related attributes from client config"""
        try:
            # Aerospike client config에서 호스트 정보 추출
            config = getattr(client, '_config', None) or {}
            hosts = config.get('hosts', [])
            if hosts:
                host, port = hosts[0] if isinstance(hosts[0], tuple) else (hosts[0], 3000)
                span.set_attribute(SpanAttributes.SERVER_ADDRESS, str(host))
                span.set_attribute(SpanAttributes.SERVER_PORT, int(port))
        except Exception:
            pass  # Config 접근 실패 시 무시
    
    def _set_result_attributes(self, span: Span, result) -> None:
        """Set attributes from operation result"""
        if isinstance(result, tuple) and len(result) >= 2:
            # (key, meta, bins) or (key, meta) format
            meta = result[1] if len(result) > 1 else None
            if isinstance(meta, dict):
                if 'gen' in meta:
                    span.set_attribute("db.aerospike.generation", meta['gen'])
                if 'ttl' in meta:
                    span.set_attribute("db.aerospike.ttl", meta['ttl'])
    
    def _set_error_attributes(self, span: Span, exc: Exception) -> None:
        """Set error attributes on span"""
        span.set_status(Status(StatusCode.ERROR, str(exc)))
        span.set_attribute(SpanAttributes.ERROR_TYPE, type(exc).__name__)
        
        # Aerospike specific error code
        if hasattr(exc, 'code'):
            span.set_attribute(SpanAttributes.DB_RESPONSE_STATUS_CODE, str(exc.code))
    
    def _create_query_scan_wrapper(
        self,
        tracer: trace.Tracer,
        operation: str,
        request_hook: Optional[Callable],
        response_hook: Optional[Callable],
        error_hook: Optional[Callable]
    ) -> Callable:
        """Create wrapper for query/scan operations (returns iterator)"""
        
        def wrapper(wrapped, instance, args, kwargs):
            namespace = args[0] if args else None
            set_name = kwargs.get('set') or (args[1] if len(args) > 1 else None)
            
            target = f"{namespace}.{set_name}" if set_name else namespace
            span_name = f"{operation} {target}" if target else operation
            
            # Query/Scan 객체 생성 시점에 span 생성
            # 실제 실행(results(), foreach())은 별도 instrumentation 필요
            with tracer.start_as_current_span(
                span_name,
                kind=SpanKind.CLIENT,
            ) as span:
                if span.is_recording():
                    span.set_attribute(SpanAttributes.DB_SYSTEM, self._DB_SYSTEM)
                    if namespace:
                        span.set_attribute(SpanAttributes.DB_NAMESPACE, namespace)
                    if set_name:
                        span.set_attribute(SpanAttributes.DB_COLLECTION_NAME, set_name)
                    span.set_attribute(SpanAttributes.DB_OPERATION_NAME, operation)
                    self._set_connection_attributes(span, instance)
                
                if request_hook:
                    request_hook(span, operation, args, kwargs)
                
                try:
                    result = wrapped(*args, **kwargs)
                    if response_hook:
                        response_hook(span, operation, result)
                    return result
                except Exception as exc:
                    if span.is_recording():
                        self._set_error_attributes(span, exc)
                    if error_hook:
                        error_hook(span, operation, exc)
                    raise
        
        return wrapper
```

#### 3.2.2 `package.py`

```python
_instruments = ("aerospike >= 11.0.0",)
```

#### 3.2.3 `version.py`

```python
__version__ = "0.1.0"
```

---

## 4. 개발 단계별 계획

### Phase 1: 기본 구조 및 Single Record Operations (2주)

**Week 1: 프로젝트 셋업**
- [ ] 프로젝트 디렉토리 구조 생성
- [ ] pyproject.toml 설정
- [ ] 기본 테스트 환경 구성 (pytest, testcontainers)
- [ ] CI/CD 파이프라인 설정 (GitHub Actions)

**Week 2: 기본 Instrumentation 구현**
- [ ] `AerospikeInstrumentor` 기본 클래스 구현
- [ ] `put`, `get`, `remove`, `exists` 메서드 instrumentation
- [ ] 기본 attribute 설정 로직
- [ ] 단위 테스트 작성

### Phase 2: 확장 Operations 및 Hooks (2주)

**Week 3: 추가 Operations**
- [ ] `select`, `touch`, `operate`, `append`, `prepend`, `increment` 구현
- [ ] Batch operations (`batch_read`, `batch_write`, `batch_operate`) 구현
- [ ] Error handling 및 status code 매핑

**Week 4: Hook 시스템**
- [ ] `request_hook`, `response_hook`, `error_hook` 구현
- [ ] Generation/TTL metadata 캡처
- [ ] 통합 테스트 작성

### Phase 3: Query/Scan 및 고급 기능 (2주)

**Week 5: Query/Scan Operations**
- [ ] `query`, `scan` 메서드 instrumentation
- [ ] Query/Scan 결과 iteration tracing (선택적)
- [ ] UDF operations (`apply`, `scan_apply`, `query_apply`)

**Week 6: 문서화 및 최적화**
- [ ] README.rst 작성
- [ ] API 문서 작성
- [ ] 성능 최적화 및 벤치마크
- [ ] CHANGELOG.md 작성

### Phase 4: Contrib 기여 준비 (1주)

**Week 7: 기여 준비**
- [ ] opentelemetry-python-contrib 기여 가이드라인 준수 확인
- [ ] Linting, formatting 적용 (black, isort, flake8)
- [ ] 전체 테스트 커버리지 확인 (80% 이상)
- [ ] PR 작성 및 제출

---

## 5. 테스트 전략

### 5.1 단위 테스트

```python
# tests/test_aerospike_instrumentation.py

import pytest
from unittest.mock import Mock, patch, MagicMock
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.instrumentation.aerospike import AerospikeInstrumentor


@pytest.fixture
def tracer_provider():
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider, exporter


@pytest.fixture
def mock_aerospike_client():
    with patch('aerospike.Client') as mock_client:
        instance = MagicMock()
        instance._config = {'hosts': [('127.0.0.1', 3000)]}
        mock_client.return_value = instance
        yield instance


class TestAerospikeInstrumentor:
    
    def test_instrument_put(self, tracer_provider, mock_aerospike_client):
        provider, exporter = tracer_provider
        AerospikeInstrumentor().instrument(tracer_provider=provider)
        
        # Simulate put operation
        mock_aerospike_client.put(('test', 'demo', 'key1'), {'bin1': 'value1'})
        
        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        
        span = spans[0]
        assert span.name == "PUT test.demo"
        assert span.attributes["db.system"] == "aerospike"
        assert span.attributes["db.namespace"] == "test"
        assert span.attributes["db.collection.name"] == "demo"
        assert span.attributes["db.operation.name"] == "PUT"
    
    def test_instrument_get_with_error(self, tracer_provider, mock_aerospike_client):
        provider, exporter = tracer_provider
        AerospikeInstrumentor().instrument(tracer_provider=provider)
        
        # Simulate error
        error = Exception("Record not found")
        error.code = 2  # KEY_NOT_FOUND_ERROR
        mock_aerospike_client.get.side_effect = error
        
        with pytest.raises(Exception):
            mock_aerospike_client.get(('test', 'demo', 'key1'))
        
        spans = exporter.get_finished_spans()
        span = spans[0]
        assert span.status.status_code == StatusCode.ERROR
        assert span.attributes["db.response.status_code"] == "2"
    
    def test_batch_operations(self, tracer_provider, mock_aerospike_client):
        provider, exporter = tracer_provider
        AerospikeInstrumentor().instrument(tracer_provider=provider)
        
        keys = [
            ('test', 'demo', 'key1'),
            ('test', 'demo', 'key2'),
            ('test', 'demo', 'key3')
        ]
        mock_aerospike_client.get_many(keys)
        
        spans = exporter.get_finished_spans()
        span = spans[0]
        assert span.attributes["db.operation.batch.size"] == 3
    
    def test_hooks(self, tracer_provider, mock_aerospike_client):
        provider, exporter = tracer_provider
        
        request_hook_called = []
        response_hook_called = []
        
        def request_hook(span, operation, args, kwargs):
            request_hook_called.append(operation)
        
        def response_hook(span, operation, result):
            response_hook_called.append(operation)
        
        AerospikeInstrumentor().instrument(
            tracer_provider=provider,
            request_hook=request_hook,
            response_hook=response_hook
        )
        
        mock_aerospike_client.get(('test', 'demo', 'key1'))
        
        assert 'GET' in request_hook_called
        assert 'GET' in response_hook_called
```

### 5.2 통합 테스트

```python
# tests/test_aerospike_integration.py

import pytest
from testcontainers.aerospike import AerospikeContainer
import aerospike
from opentelemetry.instrumentation.aerospike import AerospikeInstrumentor


@pytest.fixture(scope="module")
def aerospike_container():
    with AerospikeContainer() as container:
        yield container


@pytest.fixture
def aerospike_client(aerospike_container):
    host = aerospike_container.get_container_host_ip()
    port = aerospike_container.get_exposed_port(3000)
    
    config = {'hosts': [(host, int(port))]}
    client = aerospike.client(config)
    client.connect()
    yield client
    client.close()


class TestAerospikeIntegration:
    
    @pytest.fixture(autouse=True)
    def setup_instrumentation(self, tracer_provider):
        provider, self.exporter = tracer_provider
        AerospikeInstrumentor().instrument(tracer_provider=provider)
        yield
        AerospikeInstrumentor().uninstrument()
    
    def test_put_get_roundtrip(self, aerospike_client):
        key = ('test', 'demo', 'integration_test_key')
        bins = {'name': 'test', 'value': 42}
        
        aerospike_client.put(key, bins)
        _, meta, record = aerospike_client.get(key)
        
        assert record['name'] == 'test'
        assert record['value'] == 42
        
        spans = self.exporter.get_finished_spans()
        assert len(spans) == 2
        
        put_span = spans[0]
        get_span = spans[1]
        
        assert put_span.name == "PUT test.demo"
        assert get_span.name == "GET test.demo"
```

---

## 6. 배포 및 기여 계획

### 6.1 패키지 메타데이터 (pyproject.toml)

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "opentelemetry-instrumentation-aerospike"
dynamic = ["version"]
description = "OpenTelemetry Aerospike instrumentation"
readme = "README.rst"
license = "Apache-2.0"
requires-python = ">=3.9"
authors = [
    { name = "OpenTelemetry Authors", email = "cncf-opentelemetry-contributors@lists.cncf.io" }
]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: Apache Software License",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
]
dependencies = [
    "opentelemetry-api ~= 1.12",
    "opentelemetry-instrumentation == 0.48b0",
    "opentelemetry-semantic-conventions == 0.48b0",
    "wrapt >= 1.0.0, < 2.0.0",
    "aerospike>=17.2.0",
]

[project.optional-dependencies]
instruments = ["aerospike >= 11.0.0"]
test = [
    "opentelemetry-instrumentation-aerospike[instruments]",
    "opentelemetry-test-utils == 0.48b0",
    "pytest",
    "pytest-cov",
]

[project.entry-points.opentelemetry_instrumentor]
aerospike = "opentelemetry.instrumentation.aerospike:AerospikeInstrumentor"

[project.urls]
Homepage = "https://github.com/open-telemetry/opentelemetry-python-contrib/tree/main/instrumentation/opentelemetry-instrumentation-aerospike"

[tool.hatch.version]
path = "src/opentelemetry/instrumentation/aerospike/version.py"
```

### 6.2 Contrib 기여 체크리스트

- [ ] `CONTRIBUTING.md` 가이드라인 준수
- [ ] 코드 스타일: black, isort, flake8 적용
- [ ] 테스트 커버리지 80% 이상
- [ ] 문서화: README.rst, docstrings
- [ ] Entry point 등록: `opentelemetry_instrumentor`
- [ ] Semantic conventions 버전 명시
- [ ] CHANGELOG.md 업데이트
- [ ] `instrumentation/README.md` 테이블에 추가
- [ ] `bootstrap_gen.py`에 추가

### 6.3 향후 확장 계획

1. **Metrics 지원**
   - `db.client.operation.duration` histogram
   - `db.client.connection.count` gauge

2. **비동기 클라이언트 지원**

* 비동기 클라이언트는 아래와 같은 방식으로 사용하고있음
  이러한 패턴으로 사용시에도 문제 없도록 고려해서 개발할것
~~~Python
class AerospikeClient:
    def __init__(self, address):
        self._address = address

    async def connect(self) -> None:
        hosts = list()
        for addr in self._address.split(','):
            host, port = addr.split(':')
            hosts.append((host, int(port)))
        connect_args = dict()
        if username := os.getenv('AEROSPIKE_USERNAME'):
            connect_args['user'] = username
            connect_args['policies'] = {'auth_mode': aerospike.AUTH_INTERNAL},
            connect_args['password'] = os.getenv('AEROSPIKE_PASSWORD')
        if len(hosts) > 1:
            connect_args['shm'] = {}
        self._aerospike = aerospike.client({
            'hosts': hosts,
            'use_shared_connection': True,
            **connect_args
        })
        # TODO: can get rid of this when client is async
        class async_aerospike:
            def __init__(self, a):
                self._aerospike = a
                self._loop = None

            @property
            def loop(self):
                if not self._loop:
                    self._loop = asyncio.get_running_loop()
                return self._loop

            def __getattr__(self, attr: str):
                async def meth(*args, **kwargs):
                    return await self.loop.run_in_executor(
                        None,
                        functools.partial(
                            getattr(self._aerospike, attr), *args, **kwargs
                        ),
                    )
                return meth
        self.aerospike: aerospike.Client = async_aerospike(self._aerospike)
        await self.aerospike.connect()

    async def disconnect(self) -> None:
        await self.aerospike.close()

    @property
    def connected(self) -> bool:
        return self._aerospike.is_connected()

    def __getattr__(self, attr: str):
        return getattr(self.aerospike, attr)

~~~


3. **Semantic Conventions 업데이트**
   - `db.system.name`에 `aerospike` 추가 요청
   - Aerospike 전용 conventions 문서화

---

## 7. 리스크 및 대응 방안

| 리스크 | 영향 | 대응 방안 |
|--------|------|-----------|
| Aerospike 클라이언트 API 변경 | 높음 | 버전별 호환성 테스트, 최소 지원 버전 명시 |
| Semantic Conventions 변경 | 중간 | OTEL_SEMCONV_STABILITY_OPT_IN 환경변수 지원 |
| 성능 오버헤드 | 중간 | 벤치마크 테스트, 선택적 attribute 캡처 |
| Contrib 승인 지연 | 낮음 | 독립 패키지로 우선 배포 후 병합 |

---

## 8. 참고 자료

### 8.1 유사 Instrumentation 참고
- [opentelemetry-instrumentation-redis](https://github.com/open-telemetry/opentelemetry-python-contrib/tree/main/instrumentation/opentelemetry-instrumentation-redis)
- [opentelemetry-instrumentation-pymongo](https://github.com/open-telemetry/opentelemetry-python-contrib/tree/main/instrumentation/opentelemetry-instrumentation-pymongo)
- [opentelemetry-instrumentation-cassandra](https://github.com/open-telemetry/opentelemetry-python-contrib/tree/main/instrumentation/opentelemetry-instrumentation-cassandra)

### 8.2 관련 문서
- [OpenTelemetry Python Contributing Guide](https://github.com/open-telemetry/opentelemetry-python-contrib/blob/main/CONTRIBUTING.md)
- [Database Semantic Conventions Stability Migration](https://opentelemetry.io/docs/specs/semconv/non-normative/db-migration/)
- [Aerospike Python Client API Reference](https://aerospike-python-client.readthedocs.io/)

---

*문서 버전: 1.0*  
*작성일: 2025-12-06*  
*작성자: [Author]*