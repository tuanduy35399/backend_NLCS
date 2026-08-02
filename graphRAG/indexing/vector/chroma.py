import chromadb
from pathlib import Path
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext


class ChromaStore:

    def __init__(
        self,
        db_path: str | None = None,
        collection_name: str = "ctu_majors"
    ):
        if db_path is None:
            db_path = str(Path(__file__).resolve().parents[2] / "database" / "chroma_db")
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )
        self.vector_store = ChromaVectorStore(
            chroma_collection=self.collection
        )

    def get_storage_context(self):

        return StorageContext.from_defaults(
            vector_store=self.vector_store
        )
