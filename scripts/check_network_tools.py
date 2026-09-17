"""Opt-in, fixed-target browser smoke checks; never scans or submits forms.

Run from an installed DARKLINGER environment with --allow-public-network and the
path to an already installed Playwright MCP cli.js. No browser is installed by
this script, and no existing browser profile is attached.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from urllib.parse import urlsplit

from v_core.mcp_client import MCPClient
from v_core.mcp_tools import MCPTools, MCPToolExecutionError


async def check_network_tools(tools, *, timeout=30):
    """Return evidence for these fixed cases, not a general network health claim."""
    results = []

    async def check(name, operation):
        try:
            detail = await asyncio.wait_for(operation(), timeout)
            results.append(dict(case=name, status="passed", detail=detail))
        except Exception as error:
            results.append(dict(case=name, status="failed", error_type=type(error).__name__,
                                detail=str(error)[:400]))

    tools.begin_interaction('fixed-network-smoke', 'Read-only public example page checks.')

    async def read_example():
        result = json.loads(await tools.web_read('https://example.com/'))
        assert urlsplit(result.get('url', '')).hostname == 'example.com', 'Wrong page URL'
        assert result.get('title') == 'Example Domain', 'Expected page title missing'
        assert 'documentation' in result.get('content', '').lower(), 'Expected page content missing'
        return 'web_read returned the expected public example page and content.'

    async def snapshot():
        result = await tools.browser_call('browser_snapshot', {})
        assert '- Page URL: https://example.com/' in result, 'Snapshot belongs to another page'
        assert 'Example Domain' in result and 'documentation' in result.lower(), 'Snapshot content missing'
        return 'The independent browser snapshot matches the previously read page.'

    async def unavailable():
        try:
            await tools.web_read('https://darklinger-selftest.invalid/')
        except MCPToolExecutionError:
            return 'The reserved invalid domain produced a controlled browser error, not a successful page.'
        raise AssertionError('Unavailable page was returned as a successful read')

    async def search():
        result = json.loads(await tools.web_search('site:iana.org example domains', 6))
        rows = result.get('results', [])
        assert result.get('result_count') == len(rows) and rows, 'No grounded search results'
        official = [row for row in rows if urlsplit(row.get('url', '')).hostname in {'iana.org', 'www.iana.org'}]
        assert official, 'No expected IANA result in this search response'
        return f'web_search returned {len(rows)} result URLs including an IANA page; this is not a content audit.'

    async def grounded_read():
        urls = [url for url in tools._web_discovered_urls.values()
                if urlsplit(url).hostname in {'iana.org', 'www.iana.org'}]
        if not urls:
            raise AssertionError('No IANA search evidence available for follow-up read')
        result = json.loads(await tools.web_read(urls[0]))
        assert urlsplit(result.get('url', '')).hostname in {'iana.org', 'www.iana.org'}, 'Unexpected destination'
        content = result.get('content', '').lower()
        assert 'example' in content and 'domain' in content, 'Expected IANA subject missing'
        return 'web_read opened an actual IANA URL from the search evidence.'

    await check('read_public_example', read_example)
    await check('browser_snapshot_consistency', snapshot)
    await check('unavailable_domain_error', unavailable)
    await check('public_web_search', search)
    await check('read_search_result', grounded_read)
    return {'scope': 'fixed_read_only_public_network_smoke', 'results': results,
            'counts': {status: sum(row['status'] == status for row in results)
                       for status in ('passed', 'failed')},
            'limitations': 'No authentication, forms, Tor, generated tools, vulnerability probes, or repair execution. Failures require diagnosis; they do not alone identify a tool defect.'}


async def run(cli):
    client = MCPClient(['node', str(cli), '--headless', '--isolated', '--browser=firefox',
                        '--block-service-workers'])
    tools = object.__new__(MCPTools)
    async with client.session() as session:
        tools.browser_session = session
        tools.browser_ready = True
        try:
            report = await check_network_tools(tools)
        finally:
            await asyncio.wait_for(tools.browser_call('browser_close', {}), 10)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 1 if report['counts']['failed'] else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow-public-network', action='store_true')
    parser.add_argument('--mcp-cli', required=True, type=Path)
    args = parser.parse_args()
    if not args.allow_public_network:
        parser.error('Public network checks require --allow-public-network.')
    if not args.mcp_cli.is_file():
        parser.error('Provide an already installed Playwright MCP cli.js.')
    return asyncio.run(run(args.mcp_cli.resolve()))


if __name__ == '__main__':
    raise SystemExit(main())
