"""Ingest raw frontier LLM traces into Sarbloh agentic tool-calling format (SFT JSONL).

Converts raw frontier reasoning + action traces (e.g. #1 frame 1 ..., ACTION4, ACTION1...)
into ChatML tool-calling format:
  - assistant: reasoning_content (the frontier thought) + tool_calls (act with actions list)
  - tool: execution result from environment
  - user: observation / perception briefing for next frame

Can run in two modes:
  1. Offline Synthesizer: converts text traces into tool-calling ChatML without running the game.
  2. Live Environment Replay (--replay): replays parsed actions against real ARC games in eval/real_games
     using Sarbloh's real perception engine (briefing, object tracking, delta views).

Usage:
    # Offline mode:
    python scripts/ingest_frontier_traces.py --trace-file my_trace.txt --game-id wa30 --output sft_traces.jsonl

    # Live replay mode (ground-truth perception injection):
    python scripts/ingest_frontier_traces.py --trace-file my_trace.txt --game-id wa30 --replay --output sft_traces.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

# Ensure Sarbloh-Experimentation is importable
REPO_ROOT = Path(__file__).resolve().parents[1]
SARBLOH_EXP = REPO_ROOT / "Sarbloh-Experimentation"
for p in (REPO_ROOT, SARBLOH_EXP):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

ACTION_MAP: dict[str, str] = {
    "ACTION1": "1", "A1": "1", "UP": "1",
    "ACTION2": "2", "A2": "2", "DOWN": "2",
    "ACTION3": "3", "A3": "3", "LEFT": "3",
    "ACTION4": "4", "A4": "4", "RIGHT": "4",
    "ACTION5": "5", "A5": "5", "SPACE": "5",
    "ACTION7": "7", "A7": "7", "UNDO": "7",
    "RESET": "reset", "A0": "reset", "ACTION0": "reset",
}

_ACTION_NAMES: dict[str, str] = {
    "1": "A1(up)", "2": "A2(down)", "3": "A3(left)", "4": "A4(right)", "5": "A5(space)",
    "7": "A7(undo)", "reset": "RESET"
}


def parse_raw_trace(text: str) -> list[dict[str, Any]]:
    """Parse text formatted with headers like '#1 frame 1' into turns."""
    pattern = re.compile(r"#\s*(\d+)\s+frame\s+(\d+)", re.IGNORECASE)
    matches = list(pattern.finditer(text))
    
    if not matches:
        return [{"step": 1, "frame": 1, "content": text.strip()}]
        
    entries = []
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk = text[start:end].strip()
        entries.append({
            "step": int(match.group(1)),
            "frame": int(match.group(2)),
            "content": chunk,
        })
    return entries


def extract_thought_and_actions(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group reasoning blocks with their corresponding actions."""
    turns: list[dict[str, Any]] = []
    i = 0
    while i < len(entries):
        entry = entries[i]
        content = entry["content"]
        
        # Check if the content is a standalone action keyword (e.g. 'ACTION4')
        lines = [l.strip() for l in content.splitlines() if l.strip()]
        pure_action = None
        for line in lines:
            normalized = line.strip().upper()
            if normalized in ACTION_MAP:
                pure_action = ACTION_MAP[normalized]
                break
                
        if pure_action and len(lines) == 1:
            if turns and not turns[-1]["actions"]:
                turns[-1]["actions"].append(pure_action)
            else:
                turns.append({
                    "step": entry["step"],
                    "frame": entry["frame"],
                    "reasoning": "",
                    "actions": [pure_action],
                })
        else:
            embedded_actions = []
            for word in re.findall(r"\b(?:ACTION[0-7]|A[1-7]|UP|DOWN|LEFT|RIGHT|RESET)\b", content, re.IGNORECASE):
                act_norm = ACTION_MAP.get(word.upper())
                if act_norm and act_norm not in embedded_actions:
                    embedded_actions.append(act_norm)
                    
            next_action = None
            if not embedded_actions and i + 1 < len(entries):
                next_content = entries[i + 1]["content"].strip().upper()
                if next_content in ACTION_MAP:
                    next_action = ACTION_MAP[next_content]
                    i += 1  # Consume next entry as the action
                    
            actions = embedded_actions if embedded_actions else ([next_action] if next_action else [])
            turns.append({
                "step": entry["step"],
                "frame": entry["frame"],
                "reasoning": content,
                "actions": actions,
            })
        i += 1
        
    return turns


def build_offline_messages(
    turns: list[dict[str, Any]],
    system_prompt: str = "You are an agent playing a multi-level grid puzzle game. Use tool 'act' to move.",
    initial_user_prompt: str = "Solve the puzzle in as few moves as possible.",
) -> list[dict[str, Any]]:
    """Synthesize ChatML tool-calling messages without a live environment."""
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": initial_user_prompt},
    ]
    
    for turn_idx, turn in enumerate(turns):
        actions = turn["actions"] or ["1"]
        reasoning = turn["reasoning"]
        call_id = f"call_act_{turn_idx + 1}"
        
        tool_call = {
            "id": call_id,
            "type": "function",
            "function": {
                "name": "act",
                "arguments": json.dumps({"actions": actions, "plan": reasoning[:150].strip()}),
            },
        }
        
        messages.append({
            "role": "assistant",
            "content": "",
            "reasoning_content": reasoning,
            "tool_calls": [tool_call],
        })
        
        action_labels = [_ACTION_NAMES.get(a, f"A{a}") for a in actions]
        tool_resp_text = "\n".join(
            f"#{turn['step'] + k} {lbl}: step executed"
            for k, lbl in enumerate(action_labels)
        )
        messages.append({
            "role": "tool",
            "tool_call_id": call_id,
            "name": "act",
            "content": tool_resp_text,
        })
        
        if turn_idx < len(turns) - 1:
            messages.append({
                "role": "user",
                "content": f"[Observation Frame {turn['frame'] + 1}]: State updated.",
            })
            
    return messages


def replay_with_environment(
    turns: list[dict[str, Any]],
    game_id: str,
    envs_dir: Path | str = REPO_ROOT / "eval" / "real_games",
) -> list[dict[str, Any]]:
    """Replay actions in the real ARC game environment, injecting Sarbloh's real perception engine."""
    import arcengine
    from harness.agent import perception
    from harness.game.games import ArcGame, make_arcade

    arcade = make_arcade("offline", environments_dir=str(envs_dir))
    game = ArcGame(arcade, game_id, None, record=False)
    initial_state = game.start()

    tracker = perception.Tracker()
    current_rows = [list(r) for r in initial_state.grid]
    objects, change = tracker.observe(current_rows)

    briefing = perception.brief(current_rows, objects, initial_state.available_actions)
    initial_scene = perception.Scene(step=0, rows=current_rows, objects=objects, briefing=briefing)
    initial_text = initial_scene.text(full=True)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "You are an agent playing ARC-AGI-3. Use tool act to move."},
        {"role": "user", "content": initial_text},
    ]

    step_count = 0
    prev_rows = None
    for turn_idx, turn in enumerate(turns):
        actions = turn["actions"] or ["1"]
        reasoning = turn["reasoning"]
        call_id = f"call_act_{turn_idx + 1}"

        tool_call = {
            "id": call_id,
            "type": "function",
            "function": {
                "name": "act",
                "arguments": json.dumps({"actions": actions, "plan": reasoning[:100].strip() if reasoning else ""}),
            },
        }
        messages.append({
            "role": "assistant",
            "content": "",
            "reasoning_content": reasoning,
            "tool_calls": [tool_call],
        })

        step_lines = []
        for a_str in actions:
            step_count += 1
            act_enum = arcengine.GameAction.RESET if a_str == "reset" else arcengine.GameAction["ACTION" + a_str]
            action_input = arcengine.ActionInput(id=act_enum)
            prev_rows = current_rows
            new_state = game.execute_action(action_input)
            current_rows = [list(r) for r in new_state.grid]
            objects, change = tracker.observe(current_rows)
            desc = change if change else "step executed"
            step_lines.append(f"#{step_count} A{a_str}: {desc}")

        messages.append({
            "role": "tool",
            "tool_call_id": call_id,
            "name": "act",
            "content": "\n".join(step_lines),
        })

        if not game.state.won and game.state.engine_state != arcengine.GameState.GAME_OVER:
            scene = perception.Scene(step=step_count, rows=current_rows, objects=objects, prev=prev_rows, change=change)
            messages.append({
                "role": "user",
                "content": scene.text(full=False),
            })

    game.finish()
    return messages


def convert_trace(
    trace_text: str,
    game_id: str = "wa30",
    level: int = 0,
    replay: bool = False,
    envs_dir: Path | str = REPO_ROOT / "eval" / "real_games",
) -> dict[str, Any]:
    """Convert raw trace text to an SFT record."""
    entries = parse_raw_trace(trace_text)
    turns = extract_thought_and_actions(entries)

    if replay:
        messages = replay_with_environment(turns, game_id, envs_dir=envs_dir)
    else:
        messages = build_offline_messages(turns)

    return {
        "game_id": game_id,
        "level": level,
        "level_cleared": True,
        "rhae": 1.0,
        "actions": sum(len(t["actions"]) for t in turns),
        "messages": messages,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert raw frontier traces to tool-calling SFT JSONL.")
    parser.add_argument("--trace-file", type=str, required=True, help="Path to raw trace text file")
    parser.add_argument("--game-id", type=str, default="wa30", help="ARC game ID (e.g. wa30, ls20)")
    parser.add_argument("--level", type=int, default=0, help="Level index")
    parser.add_argument("--replay", action="store_true", help="Replay in real environment with Sarbloh perception")
    parser.add_argument("--envs-dir", type=str, default=str(REPO_ROOT / "eval" / "real_games"), help="Path to offline games")
    parser.add_argument("--output", type=str, default="sft_converted.jsonl", help="Output JSONL path")
    args = parser.parse_args()

    text = Path(args.trace_file).read_text(encoding="utf-8")
    rec = convert_trace(text, game_id=args.game_id, level=args.level, replay=args.replay, envs_dir=args.envs_dir)

    out_path = Path(args.output)
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    mode_str = "Live Replay (perception injected)" if args.replay else "Offline"
    print(f"Successfully converted trace [{mode_str}] to {out_path} ({len(rec['messages'])} messages, {rec['actions']} actions)")


if __name__ == "__main__":
    main()
