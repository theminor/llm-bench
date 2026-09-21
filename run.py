#!/usr/bin/env python3
"""Entry point: python run.py [--port 8080] [--host 0.0.0.0] [--data-dir ./data]"""
import argparse

import uvicorn

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="llm-bench web server")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8090)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--reload", action="store_true")
    a = p.parse_args()
    import llmbench.app as _app

    if a.data_dir:
        _app.app, _app.DB, _app.RUNNER = _app.create_app(a.data_dir)
    uvicorn.run(_app.app, host=a.host, port=a.port, reload=False)
