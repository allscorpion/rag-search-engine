import json


def get_movies():
    with open("data/movies.json") as f:
        data = json.load(f)
        return data["movies"]
