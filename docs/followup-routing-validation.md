# Follow-up routing verification — 2026-09-11

The reported request asked V to research the previously discussed tool online
before creating it. The old route completed a local review of 46 tool definitions.

The fix rejects that shortcut when the current request requires online work or
creation. Classification gets one correction attempt. For referential searches,
the classifier cannot invent a search subject: a separate bounded selector picks
a numbered owner message, and runtime code derives the query from the original
text. Assistant claims are excluded. Invalid or ambiguous selections stop before
search. Earlier owner instructions are not inherited as execution authorization.

## Live observations

- Model: `mythos-9b-unhinged-Q8_0`, isolated local endpoint on port 5003.
- Exact reported Polish follow-up classified as `browser` and `learning_tool`,
  with `references_previous=true` and `continue_previous=false` after correction.
- A first reference selector that asked the model to copy a phrase failed:
  Mythos changed Polish words, so the runtime rejected the result. The final
  selector returns an index instead, avoiding model-authored copies.
- Final selection returned `message_index=1`, identifying the owner's question
  about observing traffic on a website. The derived search query retained that
  subject verbatim, excluding the short introductory sentence.
- A capability reply with a seeded prior failed-attempt answer stopped repeating
  the old failure status. It generated a new feasibility answer; that answer's
  proposed technical design was not independently validated in this check.

## Automated checks and limits

Tests cover conflicting catalog classifications, ordinary local review,
exclusion of stale assistant claims, owner-only reference selection, ambiguity,
invalid index types, out-of-range indexes, and missing dialogue.

Final suites: 967 Full and 898 Public tests passed. These checks do not establish
that the requested traffic-observation tool can be built or that a full live
research-and-creation task has completed. No such tool was created in this check.
