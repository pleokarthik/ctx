from ctx_evaluate.policy.schema import InputQualityPolicy
from ctx_evaluate import store


def load_policy(pipeline: str) -> InputQualityPolicy:
    data = store.get_policy(pipeline)
    if data is None:
        return InputQualityPolicy.default()
    return InputQualityPolicy.from_dict(data)


def save_policy(pipeline: str, policy: InputQualityPolicy) -> None:
    store.write_policy(pipeline, policy.to_dict())


def reset_policy(pipeline: str) -> None:
    conn = store._connect()
    if conn is None:
        return
    try:
        conn.execute("DELETE FROM policies WHERE pipeline = ?", (pipeline,))
        conn.commit()
    finally:
        conn.close()
