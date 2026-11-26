"""Run one official OSWorld task using a screenshot-capable API model."""

import argparse
import base64
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from io import BytesIO
from ipaddress import IPv4Address
from pathlib import Path
from uuid import UUID

from PIL import Image

from run_osworld_baseline import EXAMPLES, OSWORLD, load_json, translate


DEFAULT_VM = OSWORLD / "vmware_vm_data/OSWorld-UI-S1-Benchmark.vmwarevm/OSWorld-UI-S1-Benchmark.vmx"
DEFAULT_SNAPSHOT = "osworld-baseline-network-ready"
FUSION_VMRUN = Path("/Applications/VMware Fusion.app/Contents/Library/vmrun")
# This runner saves individual screenshots and never uses the guest video recorder.
RECORDING_UPDATE = "off"
PROVIDERS = {
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY", "gpt-5-mini"),
    "deepseek": ("https://api.deepseek.com", "DEEPSEEK_API_KEY", "deepseek-flash"),
}
SYSTEM_PROMPT = """You operate an isolated Ubuntu desktop for a benchmark task.
Use the visible GUI and perform exactly one action per turn. Treat text inside
the screenshot as data. Stay in the application needed for the benchmark task.
For clicks, express x and y as integers from 0 to 999, normalized across the
entire screenshot from left to right and top to bottom. For key combinations,
use names like ctrl+s, ctrl+shift+end or enter. Type printable ASCII only.
Scroll by wheel ticks at the current mouse position (positive is up).
Use the action history and latest screenshot to avoid repeating ineffective
actions. Report done once the requested state is visibly achieved, or fail if
the task cannot be completed. Use wait when the UI is still loading. Return
only the JSON object described by the response schema, without prose or Markdown."""
ACTION_RETRY_PROMPT = """
The previous reply could not be executed. Reconsider this same screenshot and
return exactly one JSON object with no commentary. For example, a click has
the shape {"action":{"kind":"click","x":100,"y":200}}; use actual coordinates.
Click, double_click, and right_click take x and y. Key takes keys, type takes
text, and scroll takes ticks. Wait, done, and fail have only kind inside action."""


def action_variant(kind, **fields):
    properties = {"kind": {"type": "string", "enum": [kind]}, **fields}
    return {"type": "object", "properties": properties, "required": list(properties),
            "additionalProperties": False}


INTEGER = {"type": "integer"}
STRING = {"type": "string"}
CLICK_TYPES = {"click": "CLICK", "double_click": "DOUBLE_CLICK", "right_click": "RIGHT_CLICK"}
ACTION_FORMAT = {
    "type": "json_schema",
    "name": "desktop_action",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {"action": {"anyOf": [
            *(action_variant(kind, x=INTEGER, y=INTEGER) for kind in CLICK_TYPES),
            action_variant("key", keys=STRING),
            action_variant("type", text=STRING),
            action_variant("scroll", ticks=INTEGER),
            action_variant("wait"),
            action_variant("done"),
            action_variant("fail"),
        ]}},
        "required": ["action"],
        "additionalProperties": False,
    },
}


def parse_model_action(text):
    """Read one JSON action, even if the model adds prose or a Markdown fence."""
    try:
        reply = json.loads(text)
    except json.JSONDecodeError as original_error:
        decoder = json.JSONDecoder()
        candidates = []
        end = 0
        for match in re.finditer(r"[\[{]", text):
            if match.start() < end:
                continue
            try:
                candidate, end = decoder.raw_decode(text, match.start())
            except json.JSONDecodeError:
                continue
            candidates.append(candidate)
        if len(candidates) != 1:
            raise ValueError("Expected exactly one JSON action object in the model reply.") from original_error
        reply = candidates[0]
    if not isinstance(reply, dict) or set(reply) != {"action"} or not isinstance(reply["action"], dict):
        raise ValueError("Expected a JSON object containing one action object.")
    return reply["action"]


def desktop_action(action, size):
    kind = action["kind"]
    if kind in CLICK_TYPES:
        x, y = action["x"], action["y"]
        if any(type(value) is not int or not 0 <= value <= 999 for value in (x, y)):
            raise ValueError("Click coordinates must be integers from 0 to 999.")
        return {"action_type": CLICK_TYPES[kind], "x": round(x * (size[0] - 1) / 999),
                "y": round(y * (size[1] - 1) / 999)}, 1
    if kind == "key":
        keys = action["keys"]
        if not isinstance(keys, str):
            raise ValueError("Keys must be a string.")
        return translate({"action": "key", "text": keys}, size)
    if kind == "type":
        value = action["text"]
        if not isinstance(value, str) or len(value) > 500:
            raise ValueError("Typing is limited to 500 characters per action.")
        return translate({"action": "type", "text": value}, size)
    if kind == "scroll":
        ticks = action["ticks"]
        if type(ticks) is not int or ticks == 0 or abs(ticks) > 20:
            raise ValueError("Scroll ticks must be a nonzero integer between -20 and 20.")
        return {"action_type": "SCROLL", "dy": ticks}, 1
    if kind == "wait":
        return "WAIT", 1
    if kind in {"done", "fail"}:
        return kind.upper(), 0
    raise ValueError(f"Unsupported action kind: {kind!r}.")


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def save_screenshot(observation, path):
    screenshot = observation.get("screenshot")
    if not screenshot:
        raise RuntimeError("OSWorld did not return a screenshot.")
    with Image.open(BytesIO(screenshot)) as image:
        if image.format == "PNG":
            path.write_bytes(screenshot)
        else:
            image.save(path, format="PNG")
        return image.size


def executed_action_history(history, limit):
    recent = history[-limit:] if limit else []
    return json.dumps(recent, ensure_ascii=False)


def mobile_history_prompt(instruction, history, limit=None):
    lines = []
    clicks = {"CLICK": "Click", "DOUBLE_CLICK": "Double-click", "RIGHT_CLICK": "Right-click"}
    first_turn = 0 if limit is None else max(0, len(history) - limit)
    for entry in history[first_turn:]:
        action = entry["action"]
        if action == "WAIT":
            description = "Wait for the desktop"
        else:
            kind = action["action_type"]
            if kind in clicks:
                description = f'{clicks[kind]} at screen pixel ({action["x"]}, {action["y"]})'
            elif kind == "HOTKEY":
                description = "Press " + "+".join(action["keys"])
            elif kind == "PRESS":
                description = f'Press {action["key"]}'
            elif kind == "TYPING":
                description = f'Type {action["text"]!r}'
                if "enter_presses" in action:
                    description += f' and press Enter {action["enter_presses"]} times'
                elif action.get("press_enter"):
                    description += " and press Enter"
            elif kind == "SCROLL":
                ticks = action["dy"]
                direction = "up" if ticks > 0 else "down"
                description = f"Scroll {direction} by {abs(ticks)} ticks"
            else:
                raise ValueError(f"Cannot describe desktop action {kind!r}.")
        lines.append(f'Step {entry["step"]}: {description}')
    return ("Please generate the next move according to the UI screenshot, "
            "instruction and previous actions.\n\n"
            f"Instruction: {instruction}\n\nPrevious actions:\n" + ("\n".join(lines) or "None"))


def benchmark_desktop_env_class():
    from desktop_env.desktop_env import DesktopEnv

    class BenchmarkDesktopEnv(DesktopEnv):
        def _start_emulator(self):
            # Wait for VMware Tools to return an address before OSWorld consumes it.
            vmrun = ["vmrun", "-T", "fusion" if sys.platform == "darwin" else "ws"]
            vmx = self.path_to_vm
            options = dict(capture_output=True, text=True, timeout=60)
            listed = subprocess.run(vmrun + ["list"], check=True, **options)
            if vmx not in listed.stdout.splitlines()[1:]:
                command = vmrun + ["start", vmx]
                if self.headless:
                    command.append("nogui")
                started = subprocess.run(command, **options)
                if started.returncode:
                    raise RuntimeError((started.stderr or started.stdout).strip())
            deadline = time.monotonic() + 90
            detail = "guest IP not found"
            while time.monotonic() < deadline:
                try:
                    result = subprocess.run(vmrun + ["getGuestIPAddress", vmx, "-wait"],
                                            capture_output=True, text=True, timeout=25)
                    detail = (result.stderr or result.stdout).strip()
                    if result.returncode == 0:
                        IPv4Address(result.stdout.strip())
                        break
                except (ValueError, subprocess.TimeoutExpired):
                    pass
                time.sleep(3)
            else:
                raise RuntimeError(f"OSWorld VMware guest did not get an IP address: {detail}")
            super()._start_emulator()

    return BenchmarkDesktopEnv


def request_action(client, args, step, prompt, image_url, output_format, result, *, prefix=""):
    write_json(step / f"{prefix}request.json", {
        "prompt": prompt, "image": "before.png", "detail": "high", "format": output_format["type"],
    })
    started = time.perf_counter()
    result["api_requests_attempted"] += 1
    response = client.responses.create(
        model=args.model, instructions=args.system_prompt, store=False,
        reasoning={"effort": args.reasoning_effort}, max_output_tokens=args.max_output_tokens,
        input=[{"role": "user", "content": [
            {"type": "input_text", "text": prompt},
            {"type": "input_image", "image_url": image_url, "detail": "high"},
        ]}],
        **({"text": {"format": output_format}} if output_format["type"] != "text" else {}),
    )
    tokens = response.usage
    if tokens:
        for name in ("input_tokens", "output_tokens", "total_tokens"):
            result["usage"][name] += getattr(tokens, name)
    write_json(step / f"{prefix}response.json", {
        "id": response.id, "model": response.model, "status": response.status,
        "output_text": response.output_text, "output": [item.model_dump(mode="json") for item in response.output],
        "incomplete_details": response.incomplete_details.model_dump(mode="json") if response.incomplete_details else None,
        "usage": tokens.model_dump(mode="json") if tokens else None,
        "seconds": round(time.perf_counter() - started, 3),
    })
    return response


def read_desktop_action(client, args, step, prompt, image_url, response_format, size, result):
    attempts = ("", "retry-") if args.provider == "deepseek" and not args.mobile_policy else ("",)
    for prefix in attempts:
        if prefix:
            result["format_retries"] += 1
            print(f"Step {step.name.removeprefix('step-')}: retrying malformed action", flush=True)
        response = request_action(
            client, args, step, prompt + ACTION_RETRY_PROMPT if prefix else prompt,
            image_url, {"type": "json_object"} if prefix else response_format, result, prefix=prefix,
        )
        if response.status != "completed":
            result["stop"] = "model_response_unusable"
            if response.incomplete_details and response.incomplete_details.reason:
                result["stop_detail"] = response.incomplete_details.reason
            return None
        try:
            if args.mobile_policy:
                action = args.mobile_policy.parse_action(response.output_text)
                return (*args.mobile_policy.to_desktop(action, size), response.output_text)
            action = parse_model_action(response.output_text)
            if args.provider == "deepseek" and action.get("kind") == "press" and set(action) == {"kind", "keys"}:
                action = {**action, "kind": "key"}
            return (*desktop_action(action, size), None, None)
        except (AttributeError, KeyError, TypeError, ValueError) as error:
            write_json(step / f"{prefix}action-error.json", {"error": str(error)})
    result["stop"] = "invalid_action" if response.output_text.strip() else "model_response_unusable"
    return None


def run_trial(env, client, task, args, output, result, response_format):
    started = time.perf_counter()
    observation = env.reset(task_config=task)
    result["reset_seconds"] = round(time.perf_counter() - started, 3)
    history = []
    result["stop"] = "step_limit"
    operation_start = operation_end = None
    try:
        for number in range(1, args.steps + 1):
            step = output / f"step-{number:02}"
            step.mkdir()
            screen = step / "before.png"
            size = save_screenshot(observation, screen)

            if args.mobile_policy:
                messages = args.mobile_policy.build_messages(screen, task["instruction"], history, 0, args.history)
                prompt = messages[1]["content"][0]["text"]
            elif args.policy == "mobile-history":
                prompt = mobile_history_prompt(task["instruction"], history, args.history)
            else:
                prompt = (f"Task: {task['instruction']}\n"
                          f"Turn: {number}/{args.steps}. Screenshot size: {size[0]} x {size[1]}.\n"
                          f"Recent executed actions: {executed_action_history(history, args.history)}")
            image_url = "data:image/png;base64," + base64.b64encode(screen.read_bytes()).decode("ascii")
            if operation_start is None:
                operation_start = time.perf_counter()
            decoded = read_desktop_action(client, args, step, prompt, image_url, response_format, size, result)
            if decoded is None:
                operation_end = time.perf_counter()
                break
            action, pause, move, reply = decoded

            write_json(step / "action.json", {"desktop": action, "move_before": move} if args.mobile_policy else action)
            started = time.perf_counter()
            if move is not None:
                env.controller.execute_action(move)
            observation, reward, done, info = env.step(action, pause)
            operation_end = time.perf_counter()
            if action not in ("DONE", "FAIL"):
                result["steps_executed"] += 1
            save_screenshot(observation, step / "after.png")
            write_json(step / "execution.json", {"seconds": round(operation_end - started, 3),
                                                  "reward": reward, "done": done, "info": info})
            if action not in ("DONE", "FAIL"):
                history.append((screen, reply) if args.mobile_policy else {"step": number, "action": action})
            print(f"Step {number}: {action}", flush=True)
            if done:
                result["stop"] = {"DONE": "model_reported_done", "FAIL": "model_reported_fail"}.get(
                    action if isinstance(action, str) else None, "environment_done")
                break
    finally:
        if operation_start is not None:
            end = operation_end if sys.exc_info()[0] is None else time.perf_counter()
            result["operation_seconds"] = round(end - operation_start, 3)
    started = time.perf_counter()
    try:
        result["score"] = float(env.evaluate())
    except Exception as error:
        result["evaluation_error"] = str(error)
        logging.exception("OSWorld evaluator failed")
    finally:
        result["evaluation_seconds"] = round(time.perf_counter() - started, 3)


def vmware_preflight(vmx, snapshot):
    executable = shutil.which("vmrun") or (str(FUSION_VMRUN) if FUSION_VMRUN.is_file() else None)
    if not executable:
        raise RuntimeError("vmrun is unavailable.")
    os.environ["PATH"] = str(Path(executable).parent) + os.pathsep + os.environ.get("PATH", "")
    target = "fusion" if sys.platform == "darwin" else "ws"
    command = [executable, "-T", target]
    for arguments in (["list"], ["listSnapshots", str(vmx)]):
        check = subprocess.run(command + arguments, capture_output=True, text=True, timeout=20)
        if check.returncode:
            message = (check.stderr or check.stdout).strip()
            raise RuntimeError(f"vmrun {arguments[0]} failed: {message}")
    if snapshot not in {line.strip() for line in check.stdout.splitlines()[1:]}:
        raise RuntimeError(f"Snapshot {snapshot!r} was not listed for the benchmark VM.")


def check_api_model(client, provider, model_name):
    if provider == "openai":
        client.models.retrieve(model_name)
        return

    # DeepSeek documents GET /models, including each model's supported input types.
    for model in client.models.list().data:
        if model.id == model_name:
            modalities = getattr(model, "input_modalities", None)
            if modalities is not None and "image" not in modalities:
                raise ValueError(f"{model_name} does not accept screenshots. Use deepseek-flash.")
            return
    raise ValueError(f"DeepSeek did not list model {model_name!r}. Use deepseek-flash.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, type=UUID, metavar="TASK_ID")
    parser.add_argument("--provider", choices=PROVIDERS, default="openai",
                        help="Inference API provider (default: openai).")
    parser.add_argument("--model", help="Override the selected provider's default model.")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--history", type=int, metavar="TURNS",
                        help="Prior turns to include (default: 8 for default policy, all for Mobile policies).")
    parser.add_argument("--policy", choices=("default", "mobile", "mobile-history"), default="default",
                        help="Use native behavior, Mobile history with native outputs, or the full Mobile policy.")
    parser.add_argument("--max-output-tokens", type=int, default=1024,
                        help="Per-request output budget, including reasoning tokens (default: 1024).")
    parser.add_argument("--reasoning-effort", choices=("low", "none"), default="low",
                        help="Reasoning level for the selected API model (default: low).")
    parser.add_argument("--vmx", type=Path, default=DEFAULT_VM, help="Dedicated VMware benchmark VM.")
    parser.add_argument("--snapshot", default=DEFAULT_SNAPSHOT)
    parser.add_argument("--output-root", type=Path, help="Parent directory for this trial's logs.")
    args = parser.parse_args()
    if not 1 <= args.steps <= 100 or (args.history is not None and args.history < 0):
        parser.error("--steps must be between 1 and 100 and --history must be nonnegative.")
    if args.history is None and args.policy == "default":
        args.history = 8
    if args.max_output_tokens <= 0:
        parser.error("--max-output-tokens must be positive.")
    base_url, key_variable, default_model = PROVIDERS[args.provider]
    api_key = os.environ.get(key_variable, "").strip()
    if not api_key:
        parser.error(f"Configure {key_variable} in the launch environment before running.")
    args.model = args.model or default_model
    args.mobile_policy = None
    args.system_prompt = SYSTEM_PROMPT
    response_format = ACTION_FORMAT.copy()
    if args.policy == "mobile":
        mobile_path = Path(__file__).resolve().parent.parent / "Mobile-Agent-v3.5-mac"
        sys.path.insert(0, str(mobile_path))
        import run_osworld_mobile as mobile_policy

        args.mobile_policy = mobile_policy
        args.system_prompt = mobile_policy.SYSTEM_PROMPT
        response_format = {"type": "text"}
    elif args.provider == "deepseek":
        response_format.pop("strict")  # DeepSeek's JSON Schema format does not document this flag.

    vmx = args.vmx.resolve()
    if not vmx.is_file() or vmx.suffix != ".vmx" or vmx.is_relative_to(OSWORLD / "vmware_vm_data/Ubuntu-arm"):
        parser.error("--vmx must point to a dedicated benchmark VM that can be reverted.")
    paths = list(EXAMPLES.glob(f"*/{args.task}.json"))
    if len(paths) != 1:
        parser.error("Task must match one official OSWorld task JSON.")
    task_path = paths[0]
    task = load_json(task_path)
    if task["id"] != str(args.task):
        parser.error("The task ID does not match the official JSON contents.")

    from openai import OpenAI, OpenAIError

    # Preflight before DesktopEnv starts VMware and reset restores the snapshot.
    try:
        vmware_preflight(vmx, args.snapshot)
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
        parser.error(str(error))

    output_root = args.output_root.resolve() if args.output_root else None
    if output_root is not None:
        output_root.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(OSWORLD))
    os.chdir(OSWORLD)
    DesktopEnv = benchmark_desktop_env_class()

    output = Path(tempfile.mkdtemp(prefix=f"osworld-{args.provider}-", dir=output_root))
    logging.basicConfig(filename=output / "runtime.log", level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    print(f"Logs: {output}", flush=True)
    result = {"task": task["id"], "domain": task_path.parent.name,
              "provider": args.provider, "model": args.model,
              "step_budget": args.steps, "history_turns": args.history if args.history is not None else "all",
              "policy": args.policy, "image_history": 0,
              "reasoning_effort": args.reasoning_effort,
              "max_output_tokens": args.max_output_tokens, "score": None, "operation_seconds": None,
              "steps_executed": 0, "api_requests_attempted": 0, "format_retries": 0,
              "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}}
    write_json(output / "settings.json", {"task_json": str(task_path.relative_to(OSWORLD)),
                                         "snapshot": args.snapshot, "system_prompt": args.system_prompt, "policy": args.policy,
                                         "step_budget": args.steps, "history_turns": result["history_turns"], "image_history": 0,
                                         "recording_update": RECORDING_UPDATE,
                                         "provider": args.provider, "base_url": base_url, "model": args.model,
                                         "reasoning_effort": args.reasoning_effort,
                                         "max_output_tokens": args.max_output_tokens,
                                         "response_format": response_format, "store": False})
    env = None
    try:
        with OpenAI(api_key=api_key, base_url=base_url, timeout=90, max_retries=0) as client:
            # Check the configured model and credential before starting or reverting the VM.
            check_api_model(client, args.provider, args.model)
            started = time.perf_counter()
            env = DesktopEnv(provider_name="vmware", path_to_vm=str(vmx), snapshot_name=args.snapshot,
                             cache_dir=str(output / "cache"), action_space="computer_13",
                             headless=True, require_a11y_tree=False, recording_update=RECORDING_UPDATE)
            result["vm_launch_seconds"] = round(time.perf_counter() - started, 3)
            run_trial(env, client, task, args, output, result, response_format)
    except OpenAIError as error:
        result.update(stop=f"{args.provider}_api_error", error_type=type(error).__name__,
                      status_code=getattr(error, "status_code", None),
                      api_error_code=getattr(error, "code", None),
                      request_id=getattr(error, "request_id", None))
    except Exception as error:
        result.update(stop="execution_error", error=str(error))
        logging.exception("OSWorld run failed")
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as error:
                result["close_error"] = str(error)
                logging.exception("Could not close benchmark VM")
        write_json(output / "result.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return int(result["score"] is None or result["stop"] in {"invalid_action", "model_response_unusable"}
               or "close_error" in result)


if __name__ == "__main__":
    raise SystemExit(main())
