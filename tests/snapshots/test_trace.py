from any_agent.tracing.trace import AgentTrace
import json
from pathlib import Path

def test_openai_agent_trace_snapshot(snapshot):
    # Load the JSON file
    trace_path = Path(__file__).parent.parent / "assets" / "OPENAI.json"
    with open(trace_path, encoding="utf-8") as f:
        trace_data = json.load(f)
    # Try to parse as AgentTrace (will fail if schema changes)
    agent_trace = AgentTrace.model_validate(trace_data)
    # Snapshot the dict representation (so you see changes in the schema)
    # If this assert fails and you decide that you're ok with the new schema,
    # you can easily update the snapshot by running:
    # pytest tests/snapshots --snapshot-update
    assert agent_trace.model_dump() == snapshot
