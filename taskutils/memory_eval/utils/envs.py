import os


def _default_serve_url() -> str:
    host = os.getenv('SERVE_HOST', '127.0.0.1')
    port = os.getenv('SERVE_PORT', '8000')
    return f"http://{host}:{port}/v1"


DATAROOT = os.getenv('DATAROOT', '/mnt/hdfs/hongli/dataset/hotpotqa')
MAX_INPUT_LEN = int(os.getenv('MAX_INPUT_LEN', '120000'))
MAX_OUTPUT_LEN = int(os.getenv('MAX_OUTPUT_LEN', '10000'))
URL = os.getenv('URL', _default_serve_url())
API_KEY = os.getenv('API_KEY', os.getenv('OPENAI_API_KEY', '123-abc'))
RECURRENT_MAX_CONTEXT_LEN = int(os.getenv('RECURRENT_MAX_CONTEXT_LEN', '120000'))
RECURRENT_CHUNK_SIZE = int(os.getenv('RECURRENT_CHUNK_SIZE', '5000'))
RECURRENT_MAX_NEW = int(os.getenv('RECURRENT_MAX_NEW', '1024'))


def override_endpoint(url: str | None = None, api_key: str | None = None) -> None:
    """Allow callers to update URL/API_KEY at runtime."""
    global URL, API_KEY
    if url:
        URL = url
    if api_key:
        API_KEY = api_key
