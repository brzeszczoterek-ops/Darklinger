# Darklinger Public architecture

The local UI and command-line interface use the same agent runtime. The model proposes actions; the runtime validates tool arguments, executes allowed calls, and returns observed results to the model. Execution claims must be grounded in those results.

Model loading and qualification, context management, memory, routing, and speech are separate components in `src/v_core`. Generated artifacts are validated and run in a restricted sandbox. Local filesystem and browser tools have their own scope; the entire application is not an OS sandbox.

A token-limited generation is not a completed answer. Eligible agent requests receive one bounded retry; further truncation is reported as an error. This does not guarantee completion for every model or context length.

See [Model routing](MODEL_ROUTING.md), [Learning](LEARNING.md), and [Validation](TEST_PLAN.md).
