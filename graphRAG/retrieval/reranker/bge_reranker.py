from FlagEmbedding import FlagReranker


class BGEReranker:

    def __init__(self,model_name: str = "BAAI/bge-reranker-v2-m3",top_k: int = 5):
        self.top_k = top_k
        self.reranker = FlagReranker(
            model_name,
            use_fp16=False
        )

    def rerank(self, question: str, nodes):

        if not nodes:
            return []

        pairs = [
            [question, getattr(node, "text", None) or node.node.get_content()]
            for node in nodes
        ]

        scores = self.reranker.compute_score(pairs)

        ranked = sorted(
            zip(nodes, scores),
            key=lambda x: x[1],
            reverse=True
        )
        for node, score in ranked:
            node.score = score
            
        return [node for node, _ in ranked[:self.top_k]]
