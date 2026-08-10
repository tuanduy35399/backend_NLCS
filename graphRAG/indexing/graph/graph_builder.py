from collections.abc import Callable, Sequence

from llama_index.core.indices.property_graph import (
    PropertyGraphIndex,
    SchemaLLMPathExtractor,
)
from llama_index.core.schema import BaseNode
from tqdm.auto import tqdm

try:
    from indexing.graph.graph_schema import (
        POSSIBLE_ENTITIES,
        POSSIBLE_RELATIONS,
        VALIDATION_SCHEMA,
    )
except ImportError:
    from graphRAG.indexing.graph.graph_schema import (
        POSSIBLE_ENTITIES,
        POSSIBLE_RELATIONS,
        VALIDATION_SCHEMA,
    )


ProgressCallback = Callable[[int, int, BaseNode], None]


class GraphBuilder:
    """Build the property graph and persist every completed node immediately."""

    def __init__(self, llm, embed_model, graph_store, batch_size: int = 1):
        if batch_size < 1:
            raise ValueError("batch_size must be greater than or equal to 1")

        self.llm = llm
        self.embed_model = embed_model
        self.graph_store = graph_store
        self.batch_size = batch_size

    def _create_index(self, extractor: SchemaLLMPathExtractor) -> PropertyGraphIndex:
        # Creating an empty index is important here. Passing every node to the
        # constructor makes LlamaIndex finish extraction for the whole input
        # before it starts persisting data to Neo4j.
        return PropertyGraphIndex(
            nodes=[],
            llm=self.llm,
            embed_model=self.embed_model,
            property_graph_store=self.graph_store,
            kg_extractors=[extractor],
            use_async=False,
            show_progress=False,
        )

    def build(
        self,
        nodes: Sequence[BaseNode],
        progress_callback: ProgressCallback | None = None,
        start_at: int = 0,
    ) -> PropertyGraphIndex:
        """Extract and save nodes incrementally.

        With the default ``batch_size=1``, a node and its relationships are
        committed to Neo4j before the next node starts. If a later node fails,
        all nodes shown as completed by the progress bar remain available.
        """
        total = len(nodes)
        if not 0 <= start_at <= total:
            raise ValueError(f"start_at must be between 0 and {total}")
        extractor = SchemaLLMPathExtractor(
            llm=self.llm,
            possible_entities=POSSIBLE_ENTITIES,
            possible_relations=POSSIBLE_RELATIONS,
            kg_validation_schema=VALIDATION_SCHEMA,
            strict=True,
            max_triplets_per_chunk=10,
        )
        graph_index = self._create_index(extractor)

        with tqdm(
            total=total,
            initial=start_at,
            desc="Xay dung va luu graph vao Neo4j",
            unit="node",
            dynamic_ncols=True,
        ) as progress:
            for start in range(start_at, total, self.batch_size):
                batch = list(nodes[start : start + self.batch_size])

                # _insert_nodes performs extraction, embedding and Neo4j
                # upserts. The call returns only after this batch is persisted.
                graph_index.insert_nodes(batch)

                for offset, node in enumerate(batch, start=1):
                    completed = start + offset
                    progress.update(1)
                    progress.set_postfix_str(f"da luu {completed}/{total}")
                    if progress_callback is not None:
                        progress_callback(completed, total, node)

        return graph_index
