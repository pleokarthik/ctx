import logging

from ctx_capture.schema import (
    ChunkRecord,
    TokenBudget,
    TokenUsage,
    Turn,
    CacheEvent,
    RunRecord,
)
from ctx_capture import store
from ctx_capture.thread_local import set_active_run, get_active_run


def _get_logger():
    logger = logging.getLogger("ctx-capture")
    if not logger.handlers:
        log_dir = store._ctx_dir()
        log_dir.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(str(log_dir / "errors.log"))
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [ctx-capture] %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.setLevel(logging.ERROR)
    return logger


class Run:
    def __init__(self, query, pipeline=None):
        self._query = query
        self._pipeline = pipeline
        self._record = RunRecord(query=query, response="")
        self._committed = False

    def chunks(self, chunks: list) -> None:
        try:
            self._record.chunks = [
                c if isinstance(c, ChunkRecord) else ChunkRecord(**c)
                for c in chunks
            ]
        except Exception as e:
            _get_logger().error("run.chunks() failed: %s", e)

    def context(self, final_prompt: str, token_budget=None) -> None:
        try:
            self._record.final_prompt = final_prompt
            if token_budget is not None:
                self._record.token_budget = (
                    token_budget
                    if isinstance(token_budget, TokenBudget)
                    else TokenBudget(**token_budget)
                )
        except Exception as e:
            _get_logger().error("run.context() failed: %s", e)

    def history(self, pre: list, post: list, reason: str = None) -> None:
        try:
            self._record.history_pre = [
                t if isinstance(t, Turn) else Turn(**t) for t in pre
            ]
            self._record.history_post = [
                t if isinstance(t, Turn) else Turn(**t) for t in post
            ]
            self._record.eviction_reason = reason
        except Exception as e:
            _get_logger().error("run.history() failed: %s", e)

    def response(self, response: str, token_usage=None, model: str = None) -> None:
        try:
            self._record.response = response
            self._record.model = model
            if token_usage is not None:
                self._record.token_usage = (
                    token_usage
                    if isinstance(token_usage, TokenUsage)
                    else TokenUsage(**token_usage)
                )
            self.commit()
        except Exception as e:
            _get_logger().error("run.response() failed: %s", e)

    def cache(self, events: list) -> None:
        try:
            self._record.cache_events = [
                e if isinstance(e, CacheEvent) else CacheEvent(**e)
                for e in events
            ]
        except Exception as e:
            _get_logger().error("run.cache() failed: %s", e)

    def commit(self) -> None:
        if self._committed:
            return
        try:
            session_id = store.get_or_create_session(self._pipeline)
            run_seq = store.next_run_seq(session_id)
            store.write_run(session_id, run_seq, self._record, self._pipeline)
            self._committed = True
        except Exception as e:
            _get_logger().error("run.commit() failed: %s", e)


def start(query: str, pipeline: str = None) -> Run:
    run = Run(query, pipeline)
    set_active_run(run)
    return run


def capture(query: str, response: str, **kwargs) -> None:
    try:
        pipeline = kwargs.pop("pipeline", None)
        run = Run(query, pipeline)
        run._record.response = response
        if "chunks" in kwargs:
            run.chunks(kwargs["chunks"])
        if "final_prompt" in kwargs:
            run.context(kwargs["final_prompt"], kwargs.get("token_budget"))
        if "history_pre" in kwargs or "history_post" in kwargs:
            run.history(
                kwargs.get("history_pre", []),
                kwargs.get("history_post", []),
                kwargs.get("eviction_reason"),
            )
        elif "eviction_reason" in kwargs:
            run._record.eviction_reason = kwargs["eviction_reason"]
        if "cache_events" in kwargs:
            run.cache(kwargs["cache_events"])
        if "model" in kwargs:
            run._record.model = kwargs["model"]
        if "token_usage" in kwargs:
            tu = kwargs["token_usage"]
            run._record.token_usage = (
                tu if isinstance(tu, TokenUsage) else TokenUsage(**tu)
            )
        run.commit()
    except Exception as e:
        _get_logger().error("capture() failed: %s", e)


# Thread-local proxies


def chunks(chunks: list) -> None:
    run = get_active_run()
    if run is None:
        _get_logger().error("chunks() called with no active run")
        return
    run.chunks(chunks)


def context(final_prompt: str, token_budget=None) -> None:
    run = get_active_run()
    if run is None:
        _get_logger().error("context() called with no active run")
        return
    run.context(final_prompt, token_budget)


def history(pre: list, post: list, reason: str = None) -> None:
    run = get_active_run()
    if run is None:
        _get_logger().error("history() called with no active run")
        return
    run.history(pre, post, reason)


def response(response: str, token_usage=None, model: str = None) -> None:
    run = get_active_run()
    if run is None:
        _get_logger().error("response() called with no active run")
        return
    run.response(response, token_usage, model)


def cache(events: list) -> None:
    run = get_active_run()
    if run is None:
        _get_logger().error("cache() called with no active run")
        return
    run.cache(events)


def commit() -> None:
    run = get_active_run()
    if run is None:
        _get_logger().error("commit() called with no active run")
        return
    run.commit()
