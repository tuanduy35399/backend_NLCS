from typing import Literal


POSSIBLE_ENTITIES = Literal[
    "Major",
    "MajorGroup",
    "HollandType",
    "Skill",
    "Subject",
    "Career",
]

POSSIBLE_RELATIONS = Literal[
    "BELONGS_TO",
    "MATCHES_HOLLAND",
    "REQUIRES_SKILL",
    "USES_SUBJECT",
    "LEADS_TO_CAREER",
]
