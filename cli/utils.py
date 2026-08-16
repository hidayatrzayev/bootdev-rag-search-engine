import json


def load_movies() -> dict:
    with open("data/movies.json") as f:
        return json.load(f)["movies"]
