import time
import pytest
import requests


def wait_for_a2a_server(server_url: str):
    max_attempts = 20
    poll_interval = 0.5
    attempts = 0

    while True:
        try:
            # Try to make a basic GET request to check if server is responding
            response = requests.get(server_url, timeout=1.0)
            if response.status_code in [200, 404, 405]:  # Server is responding
                break
        except (requests.RequestException, ConnectionError):
            # Server not ready yet, continue polling
            pass
        time.sleep(poll_interval)
        attempts += 1
        if attempts >= max_attempts:
            msg = f"Server did not start in time. Tried {max_attempts} times with {poll_interval} second interval."
            raise ConnectionError(msg)

