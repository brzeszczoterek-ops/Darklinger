"""An owner-configured local sensory model, started only while it is needed."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
import socket

import httpx

from ..llm.llm_config import load_llm_config
from ..tools.image_analysis import local_endpoint


class PerceptionBackend:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.process = None
        self.lock = asyncio.Lock()

    def settings(self) -> dict:
        path = self.root / 'settings.json'
        if not path.is_file():
            return {}
        if path.is_symlink() or path.stat().st_size > 16_384:
            raise ValueError('Invalid perception settings file')
        values = json.loads(path.read_text())
        if not isinstance(values, dict):
            raise ValueError('Perception settings must be an object')
        return values

    def status(self) -> dict:
        settings = self.settings()
        return {'managed_vision_configured': all(settings.get(k) for k in ('server', 'model', 'projector')),
                'running': bool(self.process and self.process.returncode is None),
                'pid': self.process.pid if self.process and self.process.returncode is None else None,
                'external_local_endpoint_configured': bool(os.getenv('V_CORE_VISION_BASE_URL'))}

    @asynccontextmanager
    async def vision(self):
        if os.getenv('V_CORE_VISION_BASE_URL'):
            config = load_llm_config()
            endpoint = os.environ['V_CORE_VISION_BASE_URL']
            local_endpoint(endpoint)
            yield endpoint, os.getenv('V_CORE_VISION_MODEL', config.model)
            return
        settings = self.settings()
        if not all(settings.get(k) for k in ('server', 'model', 'projector')):
            config = load_llm_config()
            local_endpoint(config.base_url)
            yield config.base_url, config.model
            return
        async with self.lock:
            for key in ('server', 'model', 'projector'):
                if not Path(settings[key]).is_file():
                    raise ValueError(f'Missing perception {key}: {settings[key]}')
            with socket.socket() as reservation:
                reservation.bind(('127.0.0.1', 0))
                port = reservation.getsockname()[1]
            endpoint = f'http://127.0.0.1:{port}'
            alias = 'darklinger-vision'
            self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
            log_path = self.root / 'vision-server.log'
            with log_path.open('wb') as log:
                os.chmod(log_path, 0o600)
                self.process = await asyncio.create_subprocess_exec(
                    str(settings['server']), '--model', str(settings['model']), '--mmproj', str(settings['projector']),
                    '--alias', alias, '--host', '127.0.0.1', '--port', str(port), '--ctx-size', '4096',
                    '--n-gpu-layers', str(int(settings.get('gpu_layers', 0))), '--threads', str(int(settings.get('threads', 4))),
                    '--parallel', '1', '--reasoning', 'off', '--offline', '--no-webui',
                    stdin=asyncio.subprocess.DEVNULL, stdout=log, stderr=log,
                )
                try:
                    async with httpx.AsyncClient(trust_env=False, timeout=2) as client:
                        while True:
                            if self.process.returncode is not None:
                                raise RuntimeError('Vision server exited during startup; inspect vision-server.log')
                            try:
                                props = await client.get(endpoint + '/props')
                                if props.status_code == 200:
                                    if props.json().get('modalities', {}).get('vision') is not True:
                                        raise ValueError('Configured sensory model does not advertise vision')
                                    break
                            except httpx.HTTPError:
                                pass
                            await asyncio.sleep(0.25)
                    yield endpoint + '/v1', alias
                finally:
                    await self.close()

    async def close(self):
        process = self.process
        if process and process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), 5)
            except TimeoutError:
                process.kill()
                await process.wait()
        self.process = None
