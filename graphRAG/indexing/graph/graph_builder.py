from llama_index.core.indices.property_graph import PropertyGraphIndex, SchemaLLMPathExtractor
try:
    from indexing.graph.graph_schema import POSSIBLE_RELATIONS, POSSIBLE_ENTITIES
except ImportError:
    from graphRAG.indexing.graph.graph_schema import POSSIBLE_RELATIONS, POSSIBLE_ENTITIES

class GraphBuilder:
    
    def __init__(self, llm, embed_model, graph_store):
        self.llm = llm
        self.embed_model = embed_model
        self.graph_store = graph_store
    
    def build(self, nodes):
        print("GraphBuilder bat dau")
        extractor = SchemaLLMPathExtractor(
            llm=self.llm, 
            possible_entities=POSSIBLE_ENTITIES, #sau nay qua file kia sua schema la duoc
            possible_relations=POSSIBLE_RELATIONS
        )
        print("Extractor tao xong")
        graph= PropertyGraphIndex(
            nodes=nodes,
            llm=self.llm,
            embed_model=self.embed_model,
            property_graph_store=self.graph_store,
            kg_extractors=[extractor],
        )
        print("Thong so o buoc PropertyGraphIndex thuoc file graph_builder.py")
        print(type(nodes))
        print(len(nodes))
        print(type(nodes[0]))
        print("Graph build xong")
        return graph
