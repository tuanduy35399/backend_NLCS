
#Goi buoc load file vo, roi chuan hoa, lam sach may file markdown 
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
            backend_dir / "old_rag" / "rag" / "app"
            / "google_model" / "ctu_majors.json"
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

    def run(self):

        print("Dang load file json")
        documents = self.loader.load()

        print("Lam sach va chuan hoa")
        for doc in documents:
            cleaned_text = self.cleaner.clean(doc.text)
            normalized_text = self.normalizer.normalize(cleaned_text)
            doc.set_content(normalized_text)

        print("Tach text thanh node")
        nodes = self.parser.get_nodes_from_documents(documents)

        print("Giai nen metadata")
        nodes = self.metadata_extractor.extract(nodes)
        print(len(nodes))
        print("Xay graph theo may cai mqh da quy dinh")
        graph_index = self.graph_builder.build(nodes)

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
    
