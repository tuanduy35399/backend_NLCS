from typing import Literal


POSSIBLE_ENTITIES = Literal[
    "MAJOR",
    "MAJOR_GROUP",
    "HOLLAND_TYPE",
    "SKILL",
    "SUBJECT",
    "CAREER",
]

POSSIBLE_RELATIONS = Literal[
    "BELONGS_TO",
    "MATCHES_HOLLAND",
    "REQUIRES_SKILL",
    "USES_SUBJECT",
    "LEADS_TO_CAREER",
]

VALIDATION_SCHEMA: List[Tuple[str, str, str]] = [
    ("MAJOR", "BELONGS_TO", "MAJOR_GROUP"),
    ("MAJOR", "MATCHES_HOLLAND", "HOLLAND_TYPE"),
    ("MAJOR", "REQUIRES_SKILL", "SKILL"),
    ("MAJOR", "USES_SUBJECT", "SUBJECT"),
    ("MAJOR", "LEADS_TO_CAREER", "CAREER"),
]
