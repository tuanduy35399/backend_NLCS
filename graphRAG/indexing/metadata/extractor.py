from llama_index.core.extractors import (
    TitleExtractor,
    KeywordExtractor,
    SummaryExtractor,
)
from llama_index.core.ingestion import IngestionPipeline


class MetadataExtractor:
    def __init__(self, llm):
        self.pipeline = IngestionPipeline(
            transformations=[
                TitleExtractor(nodes=3, llm=llm),
                KeywordExtractor(keywords=3, llm=llm),
                SummaryExtractor(summaries=["self"], llm=llm),
                
            ]
        )

    def extract(self, nodes):
        return self.pipeline.run(nodes=nodes)
