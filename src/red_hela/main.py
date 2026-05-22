import uvicorn

from red_hela.adapters.http.app import app


def run() -> None:
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="error",
        access_log=False,
        loop="uvloop",
        http="httptools",
        timeout_keep_alive=30,
        limit_concurrency=1000,
    )


if __name__ == "__main__":
    run()
