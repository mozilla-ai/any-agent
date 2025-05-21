from syrupy.assertion import SnapshotAssertion

from any_agent.tracing.agent_trace import AgentTrace


def test_agent_trace_snapshot(
    agent_trace: AgentTrace, snapshot: SnapshotAssertion
) -> None:
    # Snapshot the dict representation (so you see changes in the schema)
    # If this assert fails and you decide that you're ok with the new schema,
    # you can easily update the snapshot by running:
    # pytest tests/snapshots --snapshot-update
    assert agent_trace.model_dump() == snapshot(
        name=agent_trace.spans[0].context.trace_id
    )
