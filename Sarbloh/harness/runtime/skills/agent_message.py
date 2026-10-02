"""Direct agent-to-agent messaging, pre-imported in the REPL as ``agent_message`` (paper section 2.4).

    await agent_message.send("found the rule: ...", receiver_role="parent")
    await agent_message.send("also check level 2", receiver_role="child", receiver_name="explorer")

Messages are queued by the host and delivered at the recipient's next turn boundary.
"""

from __future__ import annotations

from rlm import host_request


async def send(message: str, receiver_role: str = "parent", receiver_name: str | None = None) -> dict:
    if not isinstance(message, str) or not message.strip():
        raise ValueError("message must be a non-empty str")
    if receiver_role not in ("parent", "child"):
        raise ValueError("receiver_role must be 'parent' or 'child'")
    if receiver_role == "child" and not receiver_name:
        raise ValueError("receiver_name is required for receiver_role='child'")
    return await host_request("agent_message.send", {"message": message, "receiver_role": receiver_role,
                                                     "receiver_name": receiver_name})
