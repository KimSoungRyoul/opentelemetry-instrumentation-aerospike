# Aerospike clinet API

A FastAPI application providing Aerospike client

## Installation

```shell
docker compose -f compose.yaml up -d
```

```bash
# install 
uv sync

# run server
uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000
# or 
uv run python main.py
```


## Configuration

```python
AEROSPIKE_HOST = "127.0.0.1"
AEROSPIKE_PORT = 3000
AEROSPIKE_NAMESPACE = "test"
AEROSPIKE_SET = "demo"
```

```
http://localhost:8000/docs     # fastapi
http://localhost:16686/search  # jaeger
```


![jaeger sample2](./images/image2.png)


![jaeger sample](./images/image1.png)