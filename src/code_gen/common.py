import json


def load_model_from_json(fname: str) -> dict:
    """Load model related data from fname.json into a dictionary."""
    with open(fname, "r", encoding="utf-8") as f:
        model_data = json.load(f)
    return model_data
