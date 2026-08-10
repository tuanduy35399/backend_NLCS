
#Goi buoc load file vo, roi chuan hoa, lam sach may file markdown 
import json
import pickle
from pathlib import Path

from ingestion.loader.load_majors import MajorLoader
from ingestion.processes.clean_text import TextCleaner
from ingestion.processes.normalize import TextNormalizer

#Phan tich metadata don gian tu text
from indexing.metadata.extractor import MetadataExtractor

#Goi model embedding de chuyen may cai data vua phantich ra thanh graph theo mqh trong 
#schema quy dinh roi luu vao chroma
from indexing.embedding.bge_m3 import BGE_M3_Embedding
from indexing.graph.graph_builder import GraphBuilder
from indexing.vector.chroma import ChromaStore
from indexing.graph.graph_store import GraphStore
from llm.custom import CustomLLM

from llama_index.core import Settings
# from llm.ollama import OllamaLLM
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter

class BuildIndex:

    def __init__(self):

        backend_dir = Path(__file__).resolve().parents[2]
        major_file = (
            backend_dir / "graphRAG" / "knowledge"
            / "ctu_majors.json"
        )
        self.loader = MajorLoader(major_file)
        self.cleaner = TextCleaner()
        self.normalizer = TextNormalizer()
        self.parser = SentenceSplitter(chunk_size=1024, chunk_overlap=100)

        self.graph_store = GraphStore().get_store()
        self.llm = CustomLLM().get_llm()
        self.metadata_extractor = MetadataExtractor(llm=self.llm)
        
        print(type(Settings.llm))
        print("LLM da san sang")
        
        self.embed_model = BGE_M3_Embedding().get_model()
        self.graph_builder = GraphBuilder(
            llm=self.llm,
            embed_model=self.embed_model,
            graph_store=self.graph_store
        )

        self.chroma_store = ChromaStore()

        state_dir = backend_dir / "graphRAG" / "database" / "build_state"
        state_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = state_dir / "graph_checkpoint.json"
        self.metadata_cache_file = state_dir / "metadata_nodes.pkl"

    def _load_checkpoint(self):
        if not self.checkpoint_file.exists():
            return 0
        data = json.loads(self.checkpoint_file.read_text(encoding="utf-8"))
        return int(data.get("completed_graph_nodes", 0))

    def _save_checkpoint(self, completed, total, _node=None):
        temp_file = self.checkpoint_file.with_suffix(".tmp")
        temp_file.write_text(
            json.dumps(
                {"completed_graph_nodes": completed, "total_graph_nodes": total},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        temp_file.replace(self.checkpoint_file)

    def _load_metadata_cache(self):
        if not self.metadata_cache_file.exists():
            return None
        with self.metadata_cache_file.open("rb") as cache:
            return pickle.load(cache)

    def _save_metadata_cache(self, nodes):
        temp_file = self.metadata_cache_file.with_suffix(".tmp")
        with temp_file.open("wb") as cache:
            pickle.dump(nodes, cache)
        temp_file.replace(self.metadata_cache_file)

    def run(self, resume=False, resume_from=None):

        print("Dang load file json")
        documents = self.loader.load()

        print("Lam sach va chuan hoa")
        for doc in documents:
            cleaned_text = self.cleaner.clean(doc.text)
            normalized_text = self.normalizer.normalize(cleaned_text)
            doc.set_content(normalized_text)

        print("Tach text thanh node")
        nodes = self.parser.get_nodes_from_documents(documents)

        use_cache = resume or resume_from is not None
        cached_nodes = self._load_metadata_cache() if use_cache else None
        if cached_nodes is not None:
            print("Tai metadata nodes tu cache")
            nodes = cached_nodes
        else:
            print("Giai nen metadata")
            nodes = self.metadata_extractor.extract(nodes)
            self._save_metadata_cache(nodes)
            print(f"Da luu metadata checkpoint: {self.metadata_cache_file}")
        print(len(nodes))

        if resume_from is not None:
            start_at = resume_from
            self._save_checkpoint(start_at, len(nodes))
        elif resume:
            start_at = self._load_checkpoint()
        else:
            start_at = 0

        if start_at:
            print(f"Tiep tuc build graph tu node {start_at + 1}/{len(nodes)}")
        print("Xay graph va luu tung node vao Neo4j")
        graph_index = self.graph_builder.build(
            nodes,
            progress_callback=self._save_checkpoint,
            start_at=start_at,
        )

        print("Lap chi muc vector")
        storage_context = self.chroma_store.get_storage_context()
        
        vector_index = VectorStoreIndex(
            nodes=nodes,
            storage_context=storage_context,
            embed_model=self.embed_model
        )

        print("Yeahhh hoan thanh, ngon chim luon")

        return graph_index, vector_index #Cho nay tra ve graph index voi vector index
    #dung 2 cai nay de bo sung cho nhau, moi cai co diem manh, yeu rieng
    
