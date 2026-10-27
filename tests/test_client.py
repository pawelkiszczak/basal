from basal.client import Basal


def test_api_key_sets_bearer_header():
    client = Basal(api_key="secret")
    try:
        assert client.http.headers["authorization"] == "Bearer secret"
    finally:
        client.http.close()
