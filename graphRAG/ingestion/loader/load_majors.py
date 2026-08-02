import json
from pathlib import Path

from llama_index.core import Document


class MajorLoader:
    """Đọc dữ liệu ngành học đang được old_rag sử dụng."""

    def __init__(self, json_file: Path):
        self.json_file = json_file

    def load(self) -> list[Document]:
        items = json.loads(self.json_file.read_text(encoding="utf-8"))
        return [
            Document(
                text=item["noi_dung"],
                metadata={
                    "ten_nganh": item["ten_nganh"],
                    "url": item["url"],
                    "knowledge_base": "ctu_majors",
                },
            )
            for item in items
        ]
