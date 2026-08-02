class HybridSearch:

    def __init__(self, rank_constant: int = 60):
        self.rank_constant = rank_constant

    def merge(self, vector_results, graph_results):
        """Gộp hai bảng xếp hạng bằng Reciprocal Rank Fusion (RRF)."""
        nodes = {}
        scores = {}

        for results in (vector_results, graph_results):
            for rank, result in enumerate(results, start=1):
                node_id = result.node.node_id
                nodes[node_id] = result
                scores[node_id] = scores.get(node_id, 0) + 1 / (
                    self.rank_constant + rank
                )

        ranked_ids = sorted(scores, key=scores.get, reverse=True)
        merged = []
        for node_id in ranked_ids:
            result = nodes[node_id]
            result.score = scores[node_id]
            merged.append(result)

        return merged
