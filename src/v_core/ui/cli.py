from __future__ import annotations

import argparse
import asyncio
import os
import webbrowser

import uvicorn

from v_core.config import load_config
from v_core.edition import launch_edition_monitor, resolve_edition
from v_core.llm import LLM
from v_core.main import VCore
from v_core.branding import env_value
from v_core.model_loader import RoutedModelRuntime, bootstrap_interactive_model

from .app import UIRuntime, create_app


async def run_ui(*, port: int, open_browser: bool) -> None:
    config = load_config()
    allow_manual_hierarchy = bool(
        getattr(getattr(config, "edition", None), "is_full", False)
    )
    model_session = await bootstrap_interactive_model(
        config.model_runtime_root,
        mode=config.model_loader_mode,
        allow_manual_hierarchy=allow_manual_hierarchy,
    )
    owner_monitor_started = launch_edition_monitor(
        getattr(config, "edition", None) or resolve_edition("public"),
        model_session,
    )
    model_runtime: RoutedModelRuntime | None = None
    try:
        if model_session is None:
            core = VCore(config)
        else:
            shared_llm = LLM()
            model_runtime = RoutedModelRuntime(
                model_session,
                config.model_runtime_root,
                shared_llm,
                allow_manual_hierarchy=allow_manual_hierarchy,
            )
            core = VCore(config, llm=shared_llm, model_runtime=model_runtime)
    except BaseException:
        if model_runtime is not None:
            await model_runtime.stop()
        elif model_session is not None:
            await model_session.stop()
        raise

    runtime = UIRuntime(core=core, config=config, model_session=model_session)
    app = create_app(runtime)
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host="127.0.0.1",
            port=port,
            access_log=False,
            log_level="warning",
            server_header=False,
            date_header=False,
        )
    )
    runtime.shutdown_callback = lambda: setattr(server, "should_exit", True)

    async def open_when_ready() -> None:
        while not server.started and not server.should_exit:
            await asyncio.sleep(0.05)
        if server.started:
            await asyncio.to_thread(webbrowser.open, f"http://127.0.0.1:{port}/")

    opener = asyncio.create_task(open_when_ready()) if open_browser else None
    print(f"DARKLINGER UI: http://127.0.0.1:{port}/")
    if owner_monitor_started:
        print("Owner performance monitor opened in a separate terminal.")
    print("Use HOLD TO KILL or Ctrl+C to stop V and the managed model.")
    try:
        await server.serve()
    finally:
        if opener is not None and not opener.done():
            opener.cancel()
        await runtime.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Start DARKLINGER's local graphical UI")
    parser.add_argument(
        "--port",
        type=int,
        default=int(env_value("DARKLINGER_UI_PORT", "8765")),
    )
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    if not 1024 <= args.port <= 65_535:
        parser.error("--port must be between 1024 and 65535")
    asyncio.run(run_ui(port=args.port, open_browser=not args.no_browser))


if __name__ == "__main__":
    main()
