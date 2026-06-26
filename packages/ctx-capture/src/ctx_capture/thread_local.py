import threading

_local = threading.local()


def set_active_run(run) -> None:
    _local.run = run


def get_active_run():
    return getattr(_local, "run", None)


def clear_active_run() -> None:
    if hasattr(_local, "run"):
        del _local.run
