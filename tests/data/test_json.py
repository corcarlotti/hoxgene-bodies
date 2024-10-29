import json
import pytest

# Fixture to provide sample JSON data
@pytest.fixture
def json_path():
    return "./tests/data/example.json"

def test_load_json_file(json_path):
    with open(json_path, 'r') as json_file:
        data = json.load(json_file)
    return data
