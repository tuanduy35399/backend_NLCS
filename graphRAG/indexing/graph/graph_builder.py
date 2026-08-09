from llama_index.core.indices.property_graph import PropertyGraphIndex, SchemaLLMPathExtractor
try:
    from indexing.graph.graph_schema import POSSIBLE_RELATIONS, POSSIBLE_ENTITIES,VALIDATION_SCHEMA
except ImportError:
    from graphRAG.indexing.graph.graph_schema import POSSIBLE_RELATIONS, POSSIBLE_ENTITIES,VALIDATION_SCHEMA

class GraphBuilder:
    
    def __init__(self, llm, embed_model, graph_store):
        self.llm = llm
        self.embed_model = embed_model
        self.graph_store = graph_store
    
    def build(self, nodes):
        print("GraphBuilder bat dau")
        extractor = SchemaLLMPathExtractor(
            llm=self.llm, 
            possible_entities=POSSIBLE_ENTITIES,
            possible_relations=POSSIBLE_RELATIONS,
            kg_validation_schema=VALIDATION_SCHEMA,
            strict=True,
            max_triplets_per_chunk=10,
        )
        print("Extractor tao xong")
        graph= PropertyGraphIndex(
            nodes=nodes,
            llm=self.llm,
            embed_model=self.embed_model,
            property_graph_store=self.graph_store,
            kg_extractors=[extractor],
            use_async=False,
            show_progress=True,
        )
        print("Thong so o buoc PropertyGraphIndex thuoc file graph_builder.py")
        print(type(nodes))
        print(len(nodes))
        print(type(nodes[0]))
        print("Graph build xong")
        return graph
