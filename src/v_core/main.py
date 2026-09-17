from __future__ import annotations

import asyncio
import inspect
import os
import sys
from collections.abc import Callable

from .agent import Agent
from .branding import env_value
from .autonomy import (
    AuthorizationEnvelope,
    AutonomousRunner,
    AutonomousTask,
    RuntimeRegistry,
)
from .autonomy.runner import StepDriver
from .config import load_config
from .config import Config
from .edition import launch_edition_monitor, resolve_edition
from .model_loader import (
    LlamaServerStartError,
    LlamaServerUnavailable,
    ModelLoaderInteractionError,
    ModelLoaderStorageError,
    RoutedModelRuntime,
    bootstrap_interactive_model,
)
from .speech import (
    NoSpeechDetected,
    SpeechConfig,
    SpeechConfigurationError,
    SpeechRuntime,
    SpeechRuntimeError,
)

from .memory.storage import MemoryStorage
from .memory.session import Session
from .memory.reflection import Reflection
from .memory.experience import Experience
from .memory.summary import Summary
from .memory.knowledge import Knowledge
from .memory.manager import MemoryManager
from .memory.memory_engine import MemoryEngine
from .memory.proposal_filter import ProposalFilter

from .relationship import (
    RelationshipStorage,
    RelationshipUpdater,
)

from .llm import LLM


class VCore:

    def __init__(
        self,
        config: Config | None = None,
        *,
        llm: LLM | None = None,
        model_runtime: RoutedModelRuntime | None = None,
    ):

        self.config = config or load_config()
        self.model_runtime = model_runtime

        llm = llm or LLM()
        self.llm = llm

        storage = MemoryStorage(
            self.config.memory_root
        )

        manager = MemoryManager(
            storage
        )

        session = Session(
            self.config.memory_root / "conversation"
        )

        reflection = Reflection(
            llm
        )

        experience = Experience(
            llm
        )

        summary = Summary(
            llm
        )

        knowledge = Knowledge(
            llm
        )

        relationship_storage = RelationshipStorage(
            self.config.memory_root / "relationship"
        )

        relationship_updater = RelationshipUpdater(
            llm
        )

        memory = MemoryEngine(
            session=session,
            reflection=reflection,
            experience=experience,
            summary=summary,
            knowledge=knowledge,
            manager=manager,
            relationship_updater=relationship_updater,
            relationship_storage=relationship_storage,
            proposal_filter=ProposalFilter(llm),
        )

        self.agent = Agent(
            config=self.config,
            memory=memory,
            llm=llm,
            phase_router=(
                self.model_runtime.ensure_for_phase
                if self.model_runtime is not None
                else None
            ),
            response_fallback_router=(
                self.model_runtime.retry_after_rejection
                if self.model_runtime is not None
                else None
            ),
        )

        self.autonomy = AutonomousRunner(
            self.config.autonomy_root,
            learning_sink=(
                self.agent.tools.learning.capture_runtime_event
                if self.agent.tools.learning is not None
                else None
            ),
        )

        self.runtime_registry = RuntimeRegistry(
            self.config.autonomy_root / "runtime"
        )
        self.runtime_registry.register("v-core")

    async def ask(
        self,
        prompt: str,
        on_token: Callable[[str], None] | None = None,
    ) -> str:

        if self.model_runtime is not None:
            # Background reflection must release the shared client before the
            # one-model-at-a-time runtime unloads a GGUF and selects another.
            await self.agent.cancel_background_memory()
            await self.model_runtime.ensure_for(prompt)

        return await self.agent.run(
            prompt,
            on_token=on_token,
        )

    async def close(self) -> None:
        try:
            await self.agent.close()
        finally:
            try:
                if self.model_runtime is not None:
                    await self.model_runtime.stop()
            finally:
                try:
                    close = getattr(self.llm.client, "close", None)
                    if callable(close):
                        result = close()
                        if inspect.isawaitable(result):
                            await result
                finally:
                    self.runtime_registry.unregister()

    async def run_autonomous(
        self,
        objective: str,
        driver: StepDriver,
        envelope: AuthorizationEnvelope | None = None,
        *,
        task_id: str | None = None,
    ) -> AutonomousTask:
        task = AutonomousTask(
            objective=objective,
            **({"task_id": task_id} if task_id else {}),
        )

        if envelope is None:
            envelope = AuthorizationEnvelope(
                workspace=str(
                    self.config.workspace
                    / "autonomous"
                    / task.task_id
                )
            )

        return await self.autonomy.run(
            task,
            envelope,
            driver,
        )


async def chat():

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
            core = VCore(
                config,
                llm=shared_llm,
                model_runtime=model_runtime,
            )
    except BaseException:
        if model_runtime is not None:
            await model_runtime.stop()
        elif model_session is not None:
            await model_session.stop()
        raise

    print("DARKLINGER Framework powered by V")
    print("V is ready. Type 'exit' to quit.\n")
    if owner_monitor_started:
        print("Owner performance monitor opened in a separate terminal.\n")
    ptt_key = _configure_push_to_talk_hotkey()
    if ptt_key:
        print(
            f"Voice: tap {ptt_key} to start recording, tap it again to send."
        )
    print("Voice fallback: /ptt, /listen, or /voice.\n")

    speech: SpeechRuntime | None = None

    try:
        while True:
            try:

                prompt = (await _read_prompt("V > ")).strip()

                if prompt.lower() in {
                    "exit",
                    "quit",
                }:
                    break

                if speech is not None and speech.push_to_talk_recording:
                    if prompt.casefold() == "/ptt":
                        await _finish_push_to_talk(core, speech)
                    else:
                        print(
                            "Recording is active. Tap "
                            f"{ptt_key or 'the push-to-talk key'} or enter "
                            "/ptt to stop; "
                            "text was not sent."
                        )
                    continue

                if prompt.casefold() == "/ptt":
                    if speech is None:
                        try:
                            speech = SpeechRuntime(
                                SpeechConfig.load(config.voice_root)
                            )
                        except SpeechConfigurationError as error:
                            print(f"\nVoice is unavailable: {error}\n")
                            continue
                    try:
                        await speech.start_push_to_talk()
                    except SpeechRuntimeError as error:
                        print(f"\nVoice input failed: {error}\n")
                    else:
                        print(
                            "\n● RECORDING — tap "
                            f"{ptt_key or 'the push-to-talk key'} again "
                            "(or enter /ptt) "
                            "to stop and send.\n"
                        )
                    continue

                if prompt.lower() in {"/listen", "/voice"}:
                    if speech is None:
                        try:
                            speech = SpeechRuntime(
                                SpeechConfig.load(config.voice_root)
                            )
                        except SpeechConfigurationError as error:
                            print(f"\nVoice is unavailable: {error}\n")
                            continue
                    if prompt.lower() == "/listen":
                        await _voice_turn(core, speech)
                    else:
                        await _voice_loop(core, speech)
                    continue

                await _answer_in_terminal(core, prompt)

            except (KeyboardInterrupt, EOFError):
                break

            except Exception as exc:
                print(f"\n{exc}\n")
    finally:
        try:
            if speech is not None:
                await speech.close()
            await core.close()
        finally:
            if model_session is not None and model_runtime is None:
                await model_session.stop()


async def _answer_in_terminal(core: VCore, prompt: str) -> str:
    streamed = False

    def emit_token(token: str) -> None:
        nonlocal streamed
        streamed = True
        print(token, end="", flush=True)

    answer = await core.ask(prompt, on_token=emit_token)
    if streamed:
        print()
    else:
        print(answer)
    return answer


async def _voice_turn(core: VCore, speech: SpeechRuntime) -> bool:
    print("\nV is listening... Speak naturally; silence ends the recording.")
    try:
        transcript = await speech.listen()
    except NoSpeechDetected as error:
        print(f"{error}\n")
        return True
    except SpeechRuntimeError as error:
        print(f"Voice input failed: {error}\n")
        return True

    print(f"Boss (heard) > {transcript}")
    if _ends_voice_mode(transcript):
        return False

    answer = await _answer_in_terminal(core, transcript)
    print("V is speaking...")
    try:
        await speech.speak(answer)
    except SpeechRuntimeError as error:
        print(f"Voice output failed: {error}")
    print()
    return True


async def _finish_push_to_talk(core: VCore, speech: SpeechRuntime) -> None:
    print("\n■ Recording stopped. Transcribing locally...")
    try:
        transcript = await speech.stop_push_to_talk()
    except NoSpeechDetected as error:
        print(f"{error}\n")
        return
    except SpeechRuntimeError as error:
        print(f"Voice input failed: {error}\n")
        return

    print(f"Boss (heard) > {transcript}")
    answer = await _answer_in_terminal(core, transcript)
    print("V is speaking...")
    try:
        await speech.speak(answer)
    except SpeechRuntimeError as error:
        print(f"Voice output failed: {error}")
    print()


async def _voice_loop(core: VCore, speech: SpeechRuntime) -> None:
    print(
        "\nVoice mode is active. Say 'stop listening' or "
        "'wyłącz tryb głosowy' to return to the keyboard."
    )
    while await _voice_turn(core, speech):
        pass
    print("Voice mode stopped.\n")


def _ends_voice_mode(transcript: str) -> bool:
    normalized = " ".join(
        transcript.casefold().strip(" .,!?:;-—").split()
    )
    return normalized in {
        "stop listening",
        "exit voice mode",
        "stop voice mode",
        "koniec rozmowy",
        "wyłącz tryb głosowy",
        "wylacz tryb glosowy",
        "zakończ tryb głosowy",
        "zakoncz tryb glosowy",
    }


async def _read_prompt(label: str) -> str:
    """Read terminal input without blocking background memory processing."""
    return await asyncio.to_thread(input, label)


def _configure_push_to_talk_hotkey(
    *,
    environ: dict[str, str] | None = None,
    bind: Callable[[str], None] | None = None,
    stdin_is_tty: bool | None = None,
) -> str | None:
    environment = os.environ if environ is None else environ
    key = env_value(
        "DARKLINGER_PTT_KEY",
        "F2",
        environment=environment,
    ).strip().upper()
    sequences = {
        "F2": (r"\eOQ", r"\e[12~"),
        "F3": (r"\eOR", r"\e[13~"),
        "F4": (r"\eOS", r"\e[14~"),
        "F5": (r"\e[15~",),
        "F6": (r"\e[17~",),
        "F7": (r"\e[18~",),
        "F8": (r"\e[19~",),
        "F9": (r"\e[20~",),
        "F10": (r"\e[21~",),
        "F11": (r"\e[23~",),
        "F12": (r"\e[24~",),
    }
    if key not in sequences:
        return None
    interactive = sys.stdin.isatty() if stdin_is_tty is None else stdin_is_tty
    if not interactive:
        return None
    if bind is None:
        try:
            import readline
        except ImportError:
            return None
        bind = readline.parse_and_bind
    # Clear any partially typed line without ringing the terminal bell, insert
    # /ptt, and accept it immediately.
    for sequence in sequences[key]:
        bind(f'"{sequence}": "\\C-A\\C-K/ptt\\C-M"')
    return key


def main():
    try:
        asyncio.run(chat())
    except KeyboardInterrupt:
        print()
    except (
        LlamaServerStartError,
        LlamaServerUnavailable,
        ModelLoaderInteractionError,
        ModelLoaderStorageError,
        ValueError,
    ) as error:
        print(f"DARKLINGER startup failed: {error}")
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
