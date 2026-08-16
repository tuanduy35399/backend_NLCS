# RAG evaluation summary

| config | test_cases | answerable_cases | unanswerable_cases | hit_rate | mrr | precision | recall | ndcg | faithfulness | answer_relevancy | correctness | total_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vector | 60 | 58 | 2 | 0.9828 | 0.8670 | 0.2928 | 0.9828 | 0.8971 | - | - | - | 0.3472 |
| graph | 60 | 58 | 2 | 0.7069 | 0.6485 | 0.3906 | 0.6810 | 0.6426 | - | - | - | 4.0463 |
| hybrid | 60 | 58 | 2 | 0.9655 | 0.8003 | 0.2911 | 0.9483 | 0.8286 | - | - | - | 4.3718 |
| full | 60 | 58 | 2 | 1.0000 | 0.8635 | 0.3247 | 0.9914 | 0.8905 | - | - | - | 36.2974 |
