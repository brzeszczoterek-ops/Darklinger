# Inference outcome memory

Routing uses compact, persistent evidence from runtime outcomes rather than model self-ratings. User feedback, runtime verification, and latency remain distinct signals. Infrastructure errors must not count as model-quality failures.

The ledger stores structural metadata and digests rather than prompts, answers, tool arguments, or generated source. Only task-relevant final calls receive outcome credit. Unverified replies are not promoted to verified successes.

Public selects exact reviewed inference presets. Learned evidence does not grant arbitrary parameter overrides or qualify a missing model. Promotion requires repeated observations; repeated verified failures can quarantine a configuration. Quality takes precedence over latency.

Use `/inference-memory` to inspect recorded decisions. These observations apply to the tested configuration and task, not to every possible use of the model.
