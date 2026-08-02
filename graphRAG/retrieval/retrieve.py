try:
    from retrieval.rewrite.query_rewrite import QueryRewrite
    from retrieval.rewrite.hyde import HyDE

    from retrieval.search.vector_search import VectorSearch
    from retrieval.search.graph_search import GraphSearch
    from retrieval.search.hybrid_search import HybridSearch

    from retrieval.reranker.bge_reranker import BGEReranker
except ImportError:
    from graphRAG.retrieval.rewrite.query_rewrite import QueryRewrite
    from graphRAG.retrieval.rewrite.hyde import HyDE

    from graphRAG.retrieval.search.vector_search import VectorSearch
    from graphRAG.retrieval.search.graph_search import GraphSearch
    from graphRAG.retrieval.search.hybrid_search import HybridSearch

    from graphRAG.retrieval.reranker.bge_reranker import BGEReranker


class Retriever:

    def __init__(self,llm,vector_index,graph_index,top_k: int = 5):

        self.rewriter = QueryRewrite(llm)
        self.hyde = HyDE(llm)

        self.vector_search = VectorSearch(
            vector_index=vector_index,
            top_k=top_k
        )

        self.graph_search = GraphSearch(
            graph_index=graph_index,
            top_k=top_k
        )

        self.hybrid = HybridSearch()

        self.reranker = BGEReranker(
            top_k=top_k
        )

    def retrieve(self, question: str):


        rewritten_question = self.rewriter.rewrite(question)


        hyde_query = self.hyde.generate(rewritten_question)


        vector_results = self.vector_search.search(hyde_query)


        graph_results = self.graph_search.search(rewritten_question)

        merged_results = self.hybrid.merge(vector_results,graph_results)

        print("Reranking")
        final_results = self.reranker.rerank(rewritten_question,merged_results)

        return final_results
