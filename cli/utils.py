import json

from movie import Movie


def load_movies() -> list[Movie]:
    with open("data/movies.json") as f:
        return [Movie.from_dict(movie) for movie in json.load(f)["movies"]]
