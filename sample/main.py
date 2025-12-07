import logging
from contextlib import asynccontextmanager

import aerospike
from aerospike import exception as ex
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.aerospike import AerospikeInstrumentor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Setup tracer provider with service name
resource = Resource(attributes={SERVICE_NAME: "aerospike-client-api"})
provider = TracerProvider(resource=resource)

# OTLP exporter configuration (send to localhost:4317)
otlp_exporter = OTLPSpanExporter(endpoint="localhost:4317", insecure=True)
processor = BatchSpanProcessor(otlp_exporter)
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)

# Instrument Aerospike
AerospikeInstrumentor().instrument(tracer_provider=provider)


# uvicorn logger configuration
logger = logging.getLogger("uvicorn")


# Aerospike configuration
AEROSPIKE_HOST = "127.0.0.1"
AEROSPIKE_PORT = 3000
AEROSPIKE_NAMESPACE = "test"
AEROSPIKE_SET = "demo"

# Global client
aerospike_client: aerospike.Client | None = None


def get_aerospike_client() -> aerospike.Client:
    """Return Aerospike client."""
    if aerospike_client is None:
        raise HTTPException(status_code=500, detail="Aerospike client not initialized")
    return aerospike_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifecycle management."""
    global aerospike_client

    # Connect to Aerospike on startup
    config = {"hosts": [(AEROSPIKE_HOST, AEROSPIKE_PORT)]}
    aerospike_client = aerospike.client(config).connect()
    print("Aerospike connection successful")

    # Create secondary indexes for query operations
    _create_secondary_indexes(aerospike_client)

    yield

    # Disconnect on shutdown
    if aerospike_client:
        aerospike_client.close()
        print("Aerospike connection closed")


def _create_secondary_indexes(client: aerospike.Client) -> None:
    """Create secondary indexes if they don't exist."""
    indexes = [
        # Index for 'age' bin (numeric)
        {
            "ns": AEROSPIKE_NAMESPACE,
            "set": AEROSPIKE_SET,
            "bin": "age",
            "index_name": "idx_demo_age",
            "index_type": aerospike.INDEX_NUMERIC,
        },
        # Index for 'name' bin (string)
        {
            "ns": AEROSPIKE_NAMESPACE,
            "set": AEROSPIKE_SET,
            "bin": "name",
            "index_name": "idx_demo_name",
            "index_type": aerospike.INDEX_STRING,
        },
        # Index for 'city' bin (string)
        {
            "ns": AEROSPIKE_NAMESPACE,
            "set": AEROSPIKE_SET,
            "bin": "city",
            "index_name": "idx_demo_city",
            "index_type": aerospike.INDEX_STRING,
        },
    ]

    for idx_config in indexes:
        try:
            if idx_config["index_type"] == aerospike.INDEX_STRING:
                client.index_string_create(
                    idx_config["ns"], idx_config["set"], idx_config["bin"], idx_config["index_name"]
                )
            else:
                client.index_integer_create(
                    idx_config["ns"], idx_config["set"], idx_config["bin"], idx_config["index_name"]
                )
            print(f"✓ Created index: {idx_config['index_name']} on {idx_config['bin']}")
        except ex.IndexFoundError:
            print(f"✓ Index already exists: {idx_config['index_name']}")


app = FastAPI(
    title="Aerospike client API",
    description="API providing Aerospike functionality",
    version="1.0.0",
    lifespan=lifespan,
)

# Instrument FastAPI
FastAPIInstrumentor.instrument_app(app, tracer_provider=provider)


# Pydantic model definitions
class ApiTestRequest(BaseModel):
    """Request model for Aerospike API test."""

    key: str
    bins: dict
    query_bin: str
    query_value: int | str
    batch_keys: list[str] = ["batch_user_1", "batch_user_2", "batch_user_3"]

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "key": "test_user",
                    "bins": {"name": "Test User", "age": 25, "city": "Seoul"},
                    "query_bin": "age",
                    "query_value": 25,
                    "batch_keys": ["batch_user_1", "batch_user_2", "batch_user_3"],
                }
            ]
        }
    }


class ApiTestResponse(BaseModel):
    """Response model for Aerospike API test."""

    success: bool
    put_result: dict
    get_result: dict
    touch_result: dict
    query_result: list[dict]
    batch_read_result: dict

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "success": True,
                    "put_result": {"status": "ok", "key": "test_user"},
                    "get_result": {
                        "key": "test_user",
                        "bins": {"name": "Test User", "age": 25},
                        "generation": 1,
                    },
                    "touch_result": {"status": "ok", "key": "test_user", "new_generation": 2},
                    "query_result": [
                        {"key": "test_user", "bins": {"name": "Test User", "age": 25}}
                    ],
                    "batch_read_result": {
                        "status": "ok",
                        "total_keys": 3,
                        "found": 2,
                        "records": [],
                    },
                }
            ]
        }
    }


@app.get("/")
async def root():
    """Check API status."""
    return {"status": "running", "message": "Aerospike client API"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    client = get_aerospike_client()
    client.is_connected()
    return {"status": "healthy", "aerospike": "connected"}


@app.post("/aerospike/api/test")
async def aerospike_api_test(request: ApiTestRequest) -> ApiTestResponse:
    """Aerospike API test endpoint.

    - client.put: Store record
    - client.get: Retrieve record
    - client.touch: Refresh TTL
    - query.select() + query.where(): Execute query
    - client.batch_read: Read multiple records at once
    """
    client = get_aerospike_client()

    # Create key tuple (namespace, set, key)
    key = (AEROSPIKE_NAMESPACE, AEROSPIKE_SET, request.key)

    # ============================================
    # 1. client.put - Store record
    # ============================================
    logger.info(f"[PUT] key={request.key}, bins={request.bins}")

    client.put(key, request.bins)
    put_result = {"status": "ok", "key": request.key, "bins": request.bins}
    logger.info(f"[PUT] Success: key={request.key}")

    # ============================================
    # 2. client.get - Retrieve record
    # ============================================
    logger.info(f"[GET] key={request.key}")

    (key_tuple, meta, bins) = client.get(key)
    get_result = {
        "status": "ok",
        "key": request.key,
        "bins": bins,
        "generation": meta.get("gen") if meta else None,
        "ttl": meta.get("ttl") if meta else None,
    }
    logger.info(f"[GET] Success: key={request.key}, bins={bins}, meta={meta}")

    # ============================================
    # 3. client.touch - Refresh TTL
    # ============================================
    logger.info(f"[TOUCH] key={request.key}")

    # TTL policy can be set when calling touch() (using default here)
    client.touch(key)
    # Get record again after touch to verify new generation
    (_, touch_meta, _) = client.get(key)
    touch_result = {
        "status": "ok",
        "key": request.key,
        "new_generation": touch_meta.get("gen") if touch_meta else None,
        "new_ttl": touch_meta.get("ttl") if touch_meta else None,
    }
    logger.info(f"[TOUCH] Success: key={request.key}, new_meta={touch_meta}")

    # ============================================
    # 4. query.select() + query.where() - Execute query
    # ============================================
    logger.info(f"[QUERY] bin={request.query_bin}, value={request.query_value}")
    query_result = []

    # Create Query object
    query = client.query(AEROSPIKE_NAMESPACE, AEROSPIKE_SET)

    # select() - Specify bins to return
    query.select(*request.bins.keys())

    # where() - Add filter condition (requires Secondary Index)
    # Using equals predicate
    query.where(aerospike.predicates.equals(request.query_bin, request.query_value))

    # Execute query and collect results
    records = query.results()
    for record in records:
        (rec_key, rec_meta, rec_bins) = record
        query_result.append(
            {
                "key": rec_key[2] if rec_key and len(rec_key) > 2 else None,
                "bins": rec_bins,
                "generation": rec_meta.get("gen") if rec_meta else None,
            }
        )
    logger.info(f"[QUERY] Success: found {len(query_result)} records")

    # ============================================
    # 5. client.batch_read() - Read multiple records at once
    # ============================================
    logger.info(f"[BATCH_READ] keys={request.batch_keys}")

    # First, create some batch records for demonstration
    for batch_key in request.batch_keys:
        batch_key_tuple = (AEROSPIKE_NAMESPACE, AEROSPIKE_SET, batch_key)
        client.put(
            batch_key_tuple,
            {"name": f"Batch User {batch_key}", "age": 20 + len(batch_key), "city": "Seoul"},
        )

    # Prepare batch read operations
    batch_keys = [(AEROSPIKE_NAMESPACE, AEROSPIKE_SET, k) for k in request.batch_keys]

    # Execute batch_read
    batch_records = client.batch_read(batch_keys)

    batch_result_records = []

    for batch_record in batch_records.batch_records:
        print(batch_record)
        record_key, record_meta, record_bins = batch_record.record
        logger.info(
            f"[BATCH_READ] record_key={record_key}, "
            f"record_meta={record_meta}, record_bins={record_bins}"
        )
        batch_result_records.append(
            {
                "key": record_key[2] if record_key and len(record_key) > 2 else None,
                "bins": record_bins,
            }
        )

    batch_read_result = {
        "status": "ok",
        "total_keys": len(request.batch_keys),
        "found": len(batch_records.batch_records),
        "records": batch_result_records,
    }

    # ============================================
    # Return results
    # ============================================
    return ApiTestResponse(
        success=put_result.get("status") == "ok" and get_result.get("status") == "ok",
        put_result=put_result,
        get_result=get_result,
        touch_result=touch_result,
        query_result=query_result,
        batch_read_result=batch_read_result,
    )


def main():
    """Run the FastAPI application."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
