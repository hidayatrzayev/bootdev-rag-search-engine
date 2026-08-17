from typing import TypeAlias

Vector: TypeAlias = list[float]


def add_vectors(vec1: Vector, vec2: Vector) -> Vector:
    if len(vec1) != len(vec2):
        raise ValueError(f"Vectors must have same lengths to be added: {vec1}; {vec2}")

    return [vec1[i] + vec2[i] for i in range(len(vec1))]


def subtract_vectors(vec1: Vector, vec2: Vector) -> Vector:
    if len(vec1) != len(vec2):
        raise ValueError(f"Vectors must have same lengths to be subtracted: {vec1}; {vec2}")

    return [vec1[i] - vec2[i] for i in range(len(vec1))]


def dot_product(vec1: Vector, vec2: Vector) -> float:
     if len(vec1) != len(vec2):
        raise ValueError(f"Vectors must have same lengths to perform a dot product: {vec1}; {vec2}")

     return sum([vec1[i] * vec2[i] for i in range(len(vec1))])


def magnitude(vec: Vector) -> float:
    return sum([vec[i] * vec[i] for i in range(len(vec))]) ** 0.5


def cosine_similarity(vec1: Vector, vec2: Vector) -> float:
   mag1 = magnitude(vec1)
   if mag1 == 0:
       return 0.0

   mag2 = magnitude(vec2)
   if mag2 == 0:
       return 0.0
   
   return dot_product(vec1, vec2) / (mag1 * mag2)
   