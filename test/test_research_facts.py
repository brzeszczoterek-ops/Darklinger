"""Research completion must bind values to observed publisher pages."""
import json

import pytest

from v_core.autonomy.research_facts import (
    answer_fact_issues, capture_source, fact_missing, official_url,
    publisher, render_release_facts, subjects_for_request,
)
from v_core.autonomy import TaskContract

SUBJECTS = ('Python', 'Firefox')
REQUEST = ('Find the current stable releases of Python and Firefox online. '
           'Inspect two official detail pages and report the version and release date. '
           'Write in English with headings Trading & Technology and Unresolved Gaps & Limitations.')
OBSERVED = '2026-10-05T10:00:00+00:00'


def read(subject, version, day, *, actual=None, requested=None, prefix=''):
    anchor = publisher(subject)
    actual = actual or (f'https://www.python.org/downloads/release/python-{version.replace(".", "")}/'
                        if subject == 'Python' else f'https://www.firefox.com/en-US/firefox/{version}/releasenotes/')
    body = f'- Page URL: {actual}\n- Page Title: {subject} release notes\n' + prefix
    if subject == 'Python':
        body += f'- heading "Python {version}" [level=1]\n- paragraph: Release date: {day}\n- paragraph: This is a maintenance release.\n'
    else:
        body += f'- heading "{version}" [level=1]\n- paragraph: {day}\n- paragraph: Version {version}, first offered to Release channel users on {day}\n'
    result = json.dumps({'url': actual, 'content': body})
    arguments = {'url': requested or anchor.current_url}
    source = capture_source('web_read', result, arguments, SUBJECTS, REQUEST, OBSERVED)
    return {'tool':'web_read', 'status':'succeeded', 'arguments':arguments,
            'binding_source':'runtime_publisher_catalog', 'result_excerpt':result,
            'research_source':source}


@pytest.fixture
def calls():
    return [read('Python','3.14.8','Sept. 30, 2026'),read('Firefox','157.0','September 29, 2026')]


def test_requested_products_do_not_include_report_headings():
    assert subjects_for_request(REQUEST) == SUBJECTS


def test_exact_values_dates_and_provenance(calls):
    assert fact_missing(calls,SUBJECTS,official=True,current=True,request=REQUEST) == []
    for call in calls:
        record=call['research_source']
        assert len(record['content_sha256'])==64
        assert record['observed_at']==OBSERVED
        assert len(record['facts'])==2
        for fact in record['facts']:
            assert fact['quote'] and fact['url']==record['url']
    assert calls[0]['research_source']['facts'][1]['value']=='2026-09-30'


def test_preserves_evidence_before_model_excerpt_is_clipped():
    call=read('Python','3.14.8','Sept. 30, 2026',prefix='- paragraph: navigation\n'*2000)
    call['result_excerpt']=call['result_excerpt'][:500]
    assert fact_missing([call],('Python',),official=True,current=True)==[]


@pytest.mark.parametrize('url', ['https://python.org.evil.test/', 'https://evilpython.org/',
                                'https://python.org@evil.test/', 'http://www.python.org/'])
def test_publisher_identity_rejects_spoofs(url):
    assert not official_url('Python',url)


def test_redirect_to_unrelated_domain_does_not_establish_official_fact():
    call=read('Python','3.14.8','Sept. 30, 2026',actual='https://evil.test/')
    assert fact_missing([call],('Python',),official=True,current=True)


def test_unreviewed_publisher_needs_explicit_owner_anchor():
    assert not publisher('Widget','Read https://widget.test/ and report Widget.')
    request='The official source for Widget is https://widget.test/releases/'
    assert official_url('Widget','https://widget.test/releases/',request)
    assert not official_url('Widget','https://widget.test.evil.test/',request)


@pytest.mark.parametrize('version', ['3.15.0rc1', '3.15.0b2'])
def test_prerelease_is_not_standard_stable(version):
    call=read('Python',version,'Sept. 30, 2026')
    assert fact_missing([call],('Python',),official=True,current=True)


def test_future_release_is_not_current():
    call=read('Python','3.15.0','October 9, 2026')
    assert fact_missing([call],('Python',),official=True,current=True)


def test_static_archive_stable_word_does_not_prove_current():
    call=read('Python','3.13.0','October 7, 2024',requested='https://www.python.org/downloads/release/python-3130/')
    assert fact_missing([call],('Python',),official=True,current=True)


def test_conflicting_observations_block_completion(calls):
    assert fact_missing(calls+[read('Python','3.14.7','September 20, 2026')],SUBJECTS,official=True,current=True)
    assert not render_release_facts(calls+[read('Python','3.14.7','September 20, 2026')],SUBJECTS,REQUEST,True)


def test_verified_table_passes(calls):
    answer=render_release_facts(calls,SUBJECTS,REQUEST,True)
    assert 'Trading & Technology' in answer
    assert 'Unresolved Gaps & Limitations' in answer
    assert answer_fact_issues(answer,calls,SUBJECTS,official=True,current=True,request=REQUEST)==[]


@pytest.mark.parametrize('replacement', ['3.13.0','3.12.x'])
def test_old_and_placeholder_version_rejected_despite_valid_url(calls,replacement):
    answer=render_release_facts(calls,SUBJECTS,REQUEST,True).replace('3.14.8',replacement)
    issues=answer_fact_issues(answer,calls,SUBJECTS,official=True,current=True)
    assert any('unsupported_current_version=Python' in i for i in issues)


def test_wrong_table_date_rejected(calls):
    answer=render_release_facts(calls,SUBJECTS,REQUEST,True).replace('2026-09-30','2066-09-30')
    assert any('unsupported_release_date=Python' in i for i in answer_fact_issues(answer,calls,SUBJECTS,official=True,current=True))


def test_right_value_wrong_project_source_rejected(calls):
    answer=render_release_facts(calls,SUBJECTS,REQUEST,True)
    answer=answer.replace(calls[0]['research_source']['url'], calls[1]['research_source']['url'])
    assert any('unsupported_current_version=Python' in i for i in answer_fact_issues(answer,calls,SUBJECTS,official=True,current=True))


def test_vacuous_completion_without_requested_values_rejected(calls):
    assert answer_fact_issues('Done. Both official releases verified.',calls,SUBJECTS,official=True,current=True)


def test_count_cannot_change_project_or_unit():
    result=json.dumps({'url':'https://www.python.org/about/','content':'Python has 120 users.'})
    record=capture_source('web_read',result,{'url':'https://www.python.org/about/'},SUBJECTS,REQUEST,OBSERVED)
    calls=[{'tool':'web_read','status':'succeeded','research_source':record}]
    for claim in ['Python has 120 members.','Firefox has 120 users.','Python has 121 users.']:
        assert answer_fact_issues(claim,calls,SUBJECTS,official=True,current=False)
    assert not answer_fact_issues('Python has 120 users. https://www.python.org/about/',calls,SUBJECTS,official=True,current=False)


def test_contract_checks_each_official_publisher(calls):
    contract=TaskContract.from_prompt(REQUEST).with_research_sources(REQUEST)
    assert contract.required_research_subjects==SUBJECTS
    assert not contract.unmet(calls)
    missing=contract.unmet(calls[:1])
    assert 'research_evidence:official_source=Firefox' in missing
    assert contract.answer_issues(render_release_facts(calls,SUBJECTS,REQUEST,True),calls,request=REQUEST)==[]


def test_checkpoint_round_trip_and_merge_preserve_source_constraints():
    contract=TaskContract.from_prompt(REQUEST).with_research_sources(REQUEST)
    restored=TaskContract.from_dict(contract.to_dict())
    assert restored==contract
    assert TaskContract().merged(restored)==contract
    assert not restored.without_web().required_research_subjects


@pytest.mark.asyncio
@pytest.mark.parametrize('streamed',[False,True])
async def test_runtime_reads_publishers_and_recovers_two_wrong_model_reports(tmp_path,calls,streamed):
    import asyncio
    from types import SimpleNamespace
    from v_core.agent import Agent
    from v_core.memory.session import Session
    from v_core.persona.runtime import PersonaRuntime
    from v_core.persona.kernel import IdentityKernel
    from v_core.persona.voice import VoiceProfile
    from v_core.llm.llm import LLMResponse

    class Tools:
        def __init__(self): self.calls=[]
        async def openai_tool_definitions(self):
            return [{'type':'function','function':{'name':'web_read'}}]
        def bind_publisher_sources(self,*args): pass
        async def call(self,tool,arguments):
            self.calls.append((tool,arguments))
            return next(c['result_excerpt'] for c in calls if c['arguments']==arguments)
    class Model:
        config=SimpleNamespace(context=16000)
        async def respond(self,**kwargs):
            final_input=kwargs['messages'][-1]['content']
            assert 'DARKLINGER verified source facts' in final_input
            assert '3.14.8' in final_input and '2026-09-30' in final_input
            assert '157.0' in final_input and '2026-09-29' in final_input
            return LLMResponse(content=render_release_facts(calls,SUBJECTS,REQUEST,True).replace('3.14.8','3.13.0'))
    class Memory:
        session=Session()
        async def process(self,*args,**kwargs): pass
    agent=object.__new__(Agent)
    agent.llm=Model()
    agent.tools=Tools()
    agent.memory=Memory()
    agent.persona=PersonaRuntime(identity=IdentityKernel(),voice=VoiceProfile())
    agent._build_system_prompt=lambda *args,**kwargs:'system'
    agent._agent_trace_root=tmp_path/'interactive'
    agent._last_execution_context=None
    from v_core.response_preview import response_preview
    if streamed:
        async def stream(**kwargs):
            response=await agent.llm.respond(**kwargs)
            yield response.content
        agent.llm.stream=stream
    preview_token=response_preview.set((lambda *args: None) if streamed else None)
    try:
        answer=await agent._run_agent_loop(REQUEST)
    finally:
        response_preview.reset(preview_token)
    await asyncio.gather(*agent._memory_tasks)
    assert len(agent.tools.calls)==2
    assert '3.13.0' not in answer
    assert '3.14.8' in answer and '157.0' in answer
    checkpoint=json.loads(next((agent._agent_trace_root/'checkpoints').glob('*.json')).read_text())
    assert checkpoint['status']=='completed'
    assert all(c['research_source']['facts'] for c in checkpoint['tool_calls'])
    restored=TaskContract.from_dict(checkpoint['requirements'])
    assert not restored.unmet(checkpoint['tool_calls'])
    journal=next((agent._agent_trace_root/'journal').glob('*.jsonl')).read_text()
    assert 'verified_fact_report_rendered' in journal
    assert '"model_report_accepted": false' in journal


def test_release_pointer_uses_primary_heading_not_other_mentions():
    call=read('Python','3.14.8','Sept. 30, 2026')
    envelope=json.loads(call['result_excerpt'])
    envelope['content']+='\n- paragraph: The latest release uses a new install manager.\n- paragraph: Traditional installers remain in Python 3.14 and 3.15.\n'
    call['research_source']=capture_source('web_read',json.dumps(envelope),call['arguments'],SUBJECTS,REQUEST,OBSERVED)
    assert not fact_missing([call],('Python',),official=True,current=True)


def test_esr_redirect_does_not_satisfy_standard_firefox_release():
    call=read('Firefox','140.4.0','September 29, 2026')
    envelope=json.loads(call['result_excerpt'])
    envelope['content']=envelope['content'].replace('Release channel','ESR channel')
    call['research_source']=capture_source('web_read',json.dumps(envelope),call['arguments'],SUBJECTS,REQUEST,OBSERVED)
    assert fact_missing([call],('Firefox',),official=True,current=True)


def test_unattributed_extra_version_is_not_shielded_by_correct_table(calls):
    answer=render_release_facts(calls,SUBJECTS,REQUEST,True)+'\nThe latest version is 3.13.0.'
    assert answer_fact_issues(answer,calls,SUBJECTS,official=True,current=True)


def test_accessibility_heading_with_children_colon_is_normalized():
    call=read('Python','3.14.8','Sept. 30, 2026')
    envelope=json.loads(call['result_excerpt'])
    envelope['content']=envelope['content'].replace('[level=1]','[level=1]:')
    call['research_source']=capture_source('web_read',json.dumps(envelope),call['arguments'],SUBJECTS,REQUEST,OBSERVED)
    assert not fact_missing([call],('Python',),official=True,current=True)


def test_previous_release_navigation_link_is_not_primary_release():
    call=read('Firefox','157.0','September 29, 2026')
    envelope=json.loads(call['result_excerpt'])
    envelope['content']+='\n- link "Firefox 156.0" [ref=e8]:\n  - /url: /en-US/firefox/156.0/releasenotes/\n'
    call['research_source']=capture_source('web_read',json.dumps(envelope),call['arguments'],SUBJECTS,REQUEST,OBSERVED)
    assert not fact_missing([call],('Firefox',),official=True,current=True)
    assert call['research_source']['facts'][0]['value']=='157.0'


def test_english_report_labels_do_not_need_to_appear_in_polish_request(calls):
    request='Znajdź w internecie aktualne stabilne wydania Python i Firefox. Sprawdź oficjalne strony i napisz raport po angielsku.'
    contract=TaskContract.from_prompt(request).with_research_sources(request)
    for call in calls:
        call['result_excerpt']=call['result_excerpt'].replace('release notes','notes')
    answer=render_release_facts(calls,SUBJECTS,request,True)
    assert not contract.answer_issues(answer,calls,request=request)


def test_another_project_on_publisher_domain_is_not_subject_evidence():
    result=json.dumps({'url':'https://www.mozilla.org/blog/rust/',
                       'content':'The latest stable version of Rust is 1.90.0.'})
    source=capture_source('web_read',result,{'url':'https://www.mozilla.org/blog/rust/'},('Firefox',),REQUEST,OBSERVED)
    assert source['authority']=={'Firefox':'publisher_catalog'}
    assert not source['facts']


def test_requested_available_date_cannot_be_silently_omitted(calls):
    answer=render_release_facts(calls,SUBJECTS,REQUEST,True).replace('2026-09-30','Not established')
    assert 'answer:missing_release_date=Python' in answer_fact_issues(answer,calls,SUBJECTS,official=True,current=True,request=REQUEST)


def test_current_version_requires_project_attribution_and_citation(calls):
    for answer in ['The releases are 3.14.8 and 157.0.',
                   'Python 3.14.8. Firefox 157.0.']:
        assert answer_fact_issues(answer,calls,SUBJECTS,official=True,current=True,request=REQUEST)


def test_verified_release_rows_satisfy_semantic_item_facets(calls):
    from dataclasses import replace
    contract=replace(TaskContract.from_prompt(REQUEST).with_research_sources(REQUEST),
                     required_research_facets=('item_list','item_descriptions'))
    assert not contract.unmet(calls)
    assert not contract.answer_issues(render_release_facts(calls,SUBJECTS,REQUEST,True),calls,request=REQUEST)


def test_publisher_domain_alone_cannot_assign_another_projects_count():
    result=json.dumps({'url':'https://www.mozilla.org/blog/rust/',
                       'content':'Rust has 120 users.'})
    source=capture_source('web_read',result,{'url':'https://www.mozilla.org/blog/rust/'},('Firefox',),REQUEST,OBSERVED)
    assert not source['numeric_evidence']
    calls=[{'tool':'web_read','status':'succeeded','research_source':source}]
    assert answer_fact_issues('Firefox has 120 users.',calls,('Firefox',),official=True,current=False)
