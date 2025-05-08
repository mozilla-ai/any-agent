from collections.abc import AsyncIterable
from typing import TYPE_CHECKING

from common.server.task_manager import InMemoryTaskManager
from common.server.utils import (
    are_modalities_compatible,
    new_incompatible_types_error,
)
from common.types import (
    Artifact,
    JSONRPCResponse,
    Message,
    SendTaskRequest,
    SendTaskResponse,
    SendTaskStreamingRequest,
    SendTaskStreamingResponse,
    Task,
    TaskSendParams,
    TaskState,
    TaskStatus,
    TextPart,
)

if TYPE_CHECKING:
    from any_agent import AnyAgent


class AnyAgentTaskManager(InMemoryTaskManager):  # type: ignore[misc]
    """Adapted from google/a2a/samples/python."""

    def __init__(self, agent: "AnyAgent"):  # noqa: D107
        super().__init__()
        self.agent = agent

    def _validate_request(self, request: SendTaskRequest) -> JSONRPCResponse | None:
        task_send_params: TaskSendParams = request.params
        if not are_modalities_compatible(
            task_send_params.acceptedOutputModes, ["text", "text/plain"]
        ):
            return new_incompatible_types_error(request.id)
        return None

    async def on_send_task(self, request: SendTaskRequest) -> SendTaskResponse:  # noqa: D102
        error = self._validate_request(request)
        if error:
            return error
        await self.upsert_task(request.params)

        query = self._get_user_query(request.params)

        agent_trace = await self.agent.run_async(query)

        parts = [{"type": "text", "text": agent_trace.final_output}]

        task = await self._update_store(
            request.params.id,
            TaskStatus(
                state=TaskState.COMPLETED, message=Message(role="agent", parts=parts)
            ),
            [Artifact(parts=parts)],
        )
        return SendTaskResponse(id=request.id, result=task)

    async def on_send_task_subscribe(  # noqa: D102
        self, request: SendTaskStreamingRequest
    ) -> AsyncIterable[SendTaskStreamingResponse] | JSONRPCResponse:
        raise NotImplementedError

    def _get_user_query(self, task_send_params: TaskSendParams) -> str:
        part = task_send_params.message.parts[0]
        if not isinstance(part, TextPart):
            msg = "Only text parts are supported"
            raise ValueError(msg)
        return str(part.text)

    async def _update_store(
        self, task_id: str, status: TaskStatus, artifacts: list[Artifact]
    ) -> Task:
        async with self.lock:
            try:
                task = self.tasks[task_id]
            except KeyError as e:
                msg = f"Task {task_id} not found"
                raise ValueError(msg) from e
            task.status = status
            if artifacts is not None:
                if task.artifacts is None:
                    task.artifacts = []
                task.artifacts.extend(artifacts)
            return task
