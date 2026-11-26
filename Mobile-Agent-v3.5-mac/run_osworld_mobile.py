"""Run GUI-Owl 1.5 against the dedicated OSWorld VMware benchmark guest."""

import argparse
import json
import logging
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from uuid import UUID

from PIL import Image


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT / "UI-S1-mac"))
from run_osworld_baseline import EXAMPLES, OSWORLD, load_json, translate
from run_osworld_openai import (
    DEFAULT_SNAPSHOT, DEFAULT_VM, benchmark_desktop_env_class,
    save_screenshot, vmware_preflight, write_json,
)


MODEL = "mPLUG/GUI-Owl-1.5-4B-Instruct"
REVISION = "3f061c2c562cc860c42bf32542a70e07a7ff4840"

SYSTEM_PROMPT = (ROOT / "desktop_prompt.txt").read_text(encoding="utf-8")
POINTER_ACTIONS = {
    "left_click": {"action_type": "CLICK"},
    "click": {"action_type": "CLICK"},
    "right_click": {"action_type": "RIGHT_CLICK"},
    "middle_click": {"action_type": "CLICK", "button": "middle"},
    "double_click": {"action_type": "DOUBLE_CLICK"},
    "triple_click": {"action_type": "CLICK", "button": "left", "num_clicks": 3},
    "mouse_move": {"action_type": "MOVE_TO"},
    "left_click_drag": {"action_type": "DRAG_TO"},
    "drag": {"action_type": "DRAG_TO"},
}

def build_messages(screen, instruction, history, image_history, history_turns=None):
    # Follow the official PC demo's history layout. OSWorld supplies the screenshots.
    first_image = max(0, len(history) - image_history)
    earlier = []
    first_turn = 0 if history_turns is None else max(0, len(history) - history_turns)
    for index, (_, reply) in enumerate(history[first_turn:first_image], first_turn + 1):
        description = reply.split("<tool_call>", 1)[0].removeprefix("Action:").strip()
        earlier.append(f"Step {index}: {description}")
    task = ("Please generate the next move according to the UI screenshot, "
            "instruction and previous actions.\n\n"
            f"Instruction: {instruction}\n\nPrevious actions:\n" + ("\n".join(earlier) or "None"))
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    recent = history[max(first_turn, first_image):]
    for index, (image, reply) in enumerate([*recent, (screen, None)]):
        content = [{"type": "image", "image": str(image)}]
        if index == 0:
            content.insert(0, {"type": "text", "text": task})
        messages.append({"role": "user", "content": content})
        if reply is not None:
            messages.append({"role": "assistant", "content": reply})
    return messages


def parse_action(reply):
    calls = re.findall(r"<tool_call>(.*?)</tool_call>", reply, flags=re.DOTALL)
    if len(calls) != 1:
        raise ValueError("Expected one complete computer_use tool call.")
    call = json.loads(calls[0])
    if not isinstance(call, dict) or call.get("name") != "computer_use":
        raise ValueError("Expected the computer_use tool.")
    arguments = call.get("arguments")
    if not isinstance(arguments, dict):
        raise ValueError("Expected an arguments object.")
    return arguments


def position(coordinate, size):
    if not isinstance(coordinate, list) or len(coordinate) != 2:
        raise ValueError("Expected an x/y coordinate pair.")
    if any(type(value) not in (int, float) or not 0 <= value <= 1000 for value in coordinate):
        raise ValueError("The coordinate must lie within [0, 1000].")
    return dict(zip(("x", "y"), (min(edge - 1, int(value * edge / 1000))
                                 for value, edge in zip(coordinate, size))))


def to_desktop(action, size):
    kind = action["action"]
    if kind in POINTER_ACTIONS:
        return {**POINTER_ACTIONS[kind], **position(action["coordinate"], size)}, 1, None
    if kind in ("key", "hotkey"):
        keys = action["keys"]
        if isinstance(keys, str):
            keys = keys.split("+")
        aliases = {"arrow" + key: key for key in ("up", "down", "left", "right")}
        value = "+".join(aliases.get(key.lower(), key) for key in keys)
        return (*translate({"action": "key", "text": value}, size), None)
    if kind == "type":
        return (*translate(action, size), None)
    if kind in ("scroll", "hscroll"):
        ticks = action.get("pixels", 1)
        if type(ticks) not in (int, float) or not math.isfinite(ticks) or ticks == 0:
            raise ValueError("Scroll amount must be finite and nonzero.")
        # The official PC adapter passes the amount to the wheel. Bound a single event.
        amount = int(math.copysign(min(50, max(1, round(abs(ticks)))), ticks))
        move = {"action_type": "MOVE_TO", **position(action["coordinate"], size)} if "coordinate" in action else None
        return {"action_type": "SCROLL", "dx" if kind == "hscroll" else "dy": amount}, 1, move
    if kind == "wait":
        return (*translate({"action": "wait", "time": action.get("time", 2)}, size), None)
    if kind == "terminate":
        return {"success": "DONE", "failure": "FAIL"}[action["status"]], 0, None
    if kind == "answer":
        return "DONE", 0, None
    if kind == "interact":
        return "FAIL", 0, None
    raise ValueError(f"Unsupported computer_use action: {kind!r}.")


class ModelProcess:
    def __init__(self, args, log):
        self.process = subprocess.Popen(
            [str(args.model_python), str(ROOT / "infer.py"), args.device, str(args.max_tokens), str(args.weights)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True,
        )

    def receive(self):
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("GUI-Owl stopped. See inference.log for details.")
        response = json.loads(line)
        if "error" in response:
            raise RuntimeError(response["error"])
        return response

    def predict(self, messages):
        self.process.stdin.write(json.dumps(messages, ensure_ascii=False) + "\n")
        self.process.stdin.flush()
        response = self.receive()
        return response["text"], response["input_tokens"], response["output_tokens"]

    def close(self):
        self.process.stdin.close()
        try:
            self.process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self.process.stdout.close()



def run_task(env, model, task_path, args, output):
    task = load_json(task_path)
    output.mkdir()
    result = {
        "task": task["id"], "domain": task_path.parent.name, "instruction": task["instruction"],
        "model": MODEL, "revision": REVISION, "device": args.device, "snapshot": args.snapshot,
        "step_budget": args.steps, "image_history": args.image_history, "max_new_tokens": args.max_tokens,
        "history_turns": args.history if args.history is not None else "all",
        "score": None, "steps_executed": 0, "model_requests": 0, "operation_seconds": None,
    }
    history = []
    operation_start = None
    try:
        start = time.perf_counter()
        observation = env.reset(task_config=task)
        result["reset_seconds"] = round(time.perf_counter() - start, 3)
        result["stop"] = "step_limit"
        for turn in range(1, args.steps + 1):
            step = output / f"step-{turn:02}"
            step.mkdir()
            screen = step / "before.png"
            size = save_screenshot(observation, screen)
            messages = build_messages(screen, task["instruction"], history, args.image_history, args.history)
            write_json(step / "request.json", messages)

            started = time.perf_counter()
            if operation_start is None:
                operation_start = started
            result["model_requests"] += 1
            reply, input_tokens, output_tokens = model.predict(messages)
            write_json(step / "response.json", {
                "text": reply, "input_tokens": input_tokens, "output_tokens": output_tokens,
                "seconds": round(time.perf_counter() - started, 3),
            })
            try:
                action = parse_action(reply)
                desktop, pause, move = to_desktop(action, size)
            except (ValueError, KeyError, TypeError, AttributeError) as error:
                result.update(stop="invalid_action", action_error=str(error))
                print(f"{task['id']} turn {turn}: {error}", flush=True)
                break
            write_json(step / "action.json", {"model": action, "desktop": desktop, "move_before": move})
            started = time.perf_counter()
            if move is not None:
                env.controller.execute_action(move)
            observation, _, done, info = env.step(desktop, pause)
            operation_end = time.perf_counter()
            if desktop not in ("DONE", "FAIL"):
                result["steps_executed"] += 1
            save_screenshot(observation, step / "after.png")
            write_json(step / "execution.json", {
                "seconds": round(operation_end - started, 3), "done": done, "info": info,
            })
            history.append((screen, reply))
            print(f"{task['id']} turn {turn}: {desktop}", flush=True)
            if done:
                result["stop"] = "model_reported_failure" if desktop == "FAIL" else "model_reported_done"
                break

        if operation_start is not None:
            end = time.perf_counter() if result["stop"] == "invalid_action" else operation_end
            result["operation_seconds"] = round(end - operation_start, 3)
        started = time.perf_counter()
        try:
            result["score"] = float(env.evaluate())
        except Exception as error:
            result["evaluation_error"] = str(error)
            logging.exception("OSWorld evaluator failed for %s", task["id"])
        result["evaluation_seconds"] = round(time.perf_counter() - started, 3)
    except Exception as error:
        result.update(stop="execution_error", error=str(error))
        if operation_start is not None and result["operation_seconds"] is None:
            result["operation_seconds"] = round(time.perf_counter() - operation_start, 3)
        logging.exception("OSWorld task failed for %s", task["id"])
    finally:
        write_json(output / "result.json", result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--task", action="append", type=UUID, help="OSWorld task ID. Repeat for sequential trials.")
    mode.add_argument("--screen", type=Path, help="Run one offline screenshot without VMware.")
    parser.add_argument("--instruction", help="Instruction for --screen.")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--image-history", type=int, default=4)
    parser.add_argument("--history", type=int, metavar="TURNS", help="Keep this many prior turns (default: all).")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    local_python = PROJECT / "UI-S1-mac/.venv312/bin/python"
    parser.add_argument("--model-python", type=Path, default=local_python if local_python.is_file() else Path(sys.executable))
    parser.add_argument("--weights", type=Path, default=ROOT / "weights")
    parser.add_argument("--vmx", type=Path, default=DEFAULT_VM)
    parser.add_argument("--snapshot", default=DEFAULT_SNAPSHOT)
    args = parser.parse_args()
    if not 1 <= args.steps <= 100 or not 0 <= args.image_history <= 4 or args.max_tokens <= 0:
        parser.error("Use 1–100 steps, 0–4 historical images and a positive generation budget.")
    if args.history is not None and args.history < 0:
        parser.error("--history must be nonnegative.")
    args.model_python = args.model_python.expanduser().absolute()
    args.weights = args.weights.expanduser().resolve()
    args.vmx = args.vmx.expanduser().resolve()
    if not args.model_python.is_file() or not os.access(args.model_python, os.X_OK):
        parser.error("--model-python must point to an executable Python interpreter.")
    if not (args.weights / "config.json").is_file():
        parser.error("--weights must point to a local GUI-Owl model directory.")
    if args.screen and (not args.screen.is_file() or not args.instruction):
        parser.error("--screen requires an existing image and --instruction.")
    if args.task and len(set(args.task)) != len(args.task):
        parser.error("Each task ID must be unique within this run.")

    tasks = []
    for task_id in args.task or []:
        paths = list(EXAMPLES.glob(f"*/{task_id}.json"))
        if len(paths) != 1 or load_json(paths[0])["id"] != str(task_id):
            parser.error(f"No matching official task: {task_id}")
        tasks.append(paths[0])
    if tasks:
        if not args.vmx.is_file() or args.vmx.suffix != ".vmx" or args.vmx.is_relative_to(OSWORLD / "vmware_vm_data/Ubuntu-arm"):
            parser.error("--vmx must point to a dedicated benchmark VM that can be reverted.")
        try:
            vmware_preflight(args.vmx, args.snapshot)
        except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
            parser.error(str(error))

    results_root = ROOT / "results"
    results_root.mkdir(exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="pilot-", dir=results_root))
    print(f"Logs: {output}", flush=True)
    logging.basicConfig(filename=output / "runtime.log", level=logging.INFO)
    write_json(output / "settings.json", {
        "model": MODEL, "revision": REVISION,
        "device": args.device, "snapshot": args.snapshot, "vm": str(args.vmx),
        "model_python": str(args.model_python), "weights": str(args.weights),
        "steps": args.steps, "image_history": args.image_history, "max_new_tokens": args.max_tokens,
        "history_turns": args.history if args.history is not None else "all",
        "system_prompt": SYSTEM_PROMPT, "decoding": "greedy", "tasks": [p.stem for p in tasks],
        "dtype": "float16" if args.device == "mps" else "auto",
        "attention": "eager" if args.device == "mps" else "sdpa",
    })
    env = None
    launch_seconds = None
    results = []
    with (output / "inference.log").open("w", encoding="utf-8") as log:
        print(f"Loading {MODEL} on {args.device}...", flush=True)
        model = ModelProcess(args, log)
        try:
            model_load_seconds = model.receive()["model_load_seconds"]
            settings = load_json(output / "settings.json")
            settings["model_load_seconds"] = model_load_seconds
            write_json(output / "settings.json", settings)
            if args.screen:
                messages = build_messages(args.screen.resolve(), args.instruction, [], 0)
                started = time.perf_counter()
                reply, input_tokens, output_tokens = model.predict(messages)
                response = {"text": reply, "input_tokens": input_tokens, "output_tokens": output_tokens,
                            "seconds": round(time.perf_counter() - started, 3)}
                write_json(output / "response.json", response)
                print(json.dumps(response, ensure_ascii=False), flush=True)
                print(json.dumps(parse_action(reply), ensure_ascii=False), flush=True)
                return 0

            sys.path.insert(0, str(OSWORLD))
            os.chdir(OSWORLD)
            DesktopEnv = benchmark_desktop_env_class()

            started = time.perf_counter()
            env = DesktopEnv(
                provider_name="vmware", path_to_vm=str(args.vmx), snapshot_name=args.snapshot,
                cache_dir=str(output / "cache"), action_space="computer_13", headless=True,
                require_a11y_tree=False, recording_update="off",
            )
            launch_seconds = round(time.perf_counter() - started, 3)
            for task in tasks:
                results.append(run_task(env, model, task, args, output / task.stem))
        finally:
            try:
                if env is not None:
                    env.close()
            finally:
                model.close()
                if tasks:
                    write_json(output / "summary.json", {"vm_launch_seconds": launch_seconds, "results": results})
    return int(any(r["score"] is None or r["stop"] in ("invalid_action", "execution_error") for r in results))


if __name__ == "__main__":
    raise SystemExit(main())
