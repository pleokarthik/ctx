from ctx_capture.schema import RunRecord


def score(record: RunRecord, ground_truth: str = None) -> dict | None:
    if not record.chunks or not record.response:
        return None

    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision
        from datasets import Dataset
    except ImportError:
        raise ImportError("RAGAS not installed. Run: pip install ragas")

    data = {
        "question": [record.query],
        "answer": [record.response],
        "contexts": [[c.content for c in record.chunks]],
    }

    metrics = [faithfulness, answer_relevancy, context_precision]

    if ground_truth:
        try:
            from ragas.metrics import context_recall

            data["ground_truth"] = [ground_truth]
            metrics.append(context_recall)
        except ImportError:
            pass

    try:
        dataset = Dataset.from_dict(data)
        result = evaluate(dataset, metrics=metrics)

        scores = {
            "faithfulness": None,
            "answer_relevancy": None,
            "context_precision": None,
            "context_recall": None,
            "evaluator": "ragas",
            "model": "unknown",
        }

        for key in scores:
            if key in ("evaluator", "model"):
                continue
            try:
                val = result[key]
                if isinstance(val, (list, tuple)) and len(val) > 0:
                    scores[key] = float(val[0])
                elif isinstance(val, (int, float)):
                    scores[key] = float(val)
            except (KeyError, TypeError, IndexError):
                pass

        return scores
    except Exception as e:
        return {
            "faithfulness": None,
            "answer_relevancy": None,
            "context_precision": None,
            "context_recall": None,
            "evaluator": "ragas",
            "model": "error",
            "error": str(e),
        }
