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

VALIDATION_SCHEMA: List[Tuple[str, str, str]] = [
    ("Major", "BELONGS_TO", "MajorGroup"),
    ("Major", "MATCHES_HOLLAND", "HollandType"),
    ("Major", "REQUIRES_SKILL", "Skill"),
    ("Major", "USES_SUBJECT", "Subject"),
    ("Major", "LEADS_TO_CAREER", "Career"),
]