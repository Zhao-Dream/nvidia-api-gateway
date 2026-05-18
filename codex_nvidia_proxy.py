
import sys
import os
import json
import uuid
import getpass

from flask import Flask, request, Response
from openai import OpenAI

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_dotenv():
    """加载 .env 文件到 os.environ（不覆盖已有的系统环境变量）"""
    env_file = os.path.join(BASE_DIR, ".nvidia_env")
    if not os.path.exists(env_file):
        return
    had_key = "NVIDIA_API_KEY" in os.environ
    with open(env_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip().strip("\"'")
            if key and key not in os.environ:
                os.environ[key] = val
    if "NVIDIA_API_KEY" in os.environ:
        os.environ["_NVIDIA_KEY_SOURCE"] = "sys" if had_key else "dotenv"


def _ensure_api_key():
    """确保 NVIDIA_API_KEY 已设置：系统环境变量 > .nvidia_env > 交互输入"""
    key = os.environ.get("NVIDIA_API_KEY", "").strip()
    if key:
        src = os.environ.get("_NVIDIA_KEY_SOURCE", "")
        if src == "sys":
            return key, "系统环境变量"
        return key, ".nvidia_env"

    print("=" * 60)
    print("  未检测到 NVIDIA_API_KEY")
    print("=" * 60)
    print()
    print("  从 https://build.nvidia.com/ 获取 API Key")
    print("  登录后点击任一模型 → Get API Key")
    print()
    print("  你也可以设置系统环境变量 NVIDIA_API_KEY 后重启")
    print()

    try:
        key = getpass.getpass("  请输入你的 NVIDIA API Key: ").strip()
    except (EOFError, KeyboardInterrupt):
        key = ""

    if not key:
        print()
        print("  ERROR: 未输入 API Key，程序退出。")
        print()
        input("  按 Enter 退出...")
        sys.exit(1)

    env_file = os.path.join(BASE_DIR, ".nvidia_env")
    existing = {}
    if os.path.exists(env_file):
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                existing[k.strip()] = f"{k.strip()}={v.strip()}"
    existing["NVIDIA_API_KEY"] = f"NVIDIA_API_KEY={key}"

    with open(env_file, "w", encoding="utf-8") as f:
        for line in existing.values():
            f.write(line + "\n")
        if "NVIDIA_MODEL" not in existing:
            f.write("NVIDIA_MODEL=nvidia/llama-3.1-nemotron-70b-instruct\n")

    os.environ["NVIDIA_API_KEY"] = key
    print()
    print(f"  API Key 已保存到: {env_file}")
    print()
    return key, ".nvidia_env (已保存)"


_load_dotenv()

DEBUG_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nvidia_proxy_debug.log")

app = Flask(__name__)

# ===================== 配置 =====================
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "").strip()
NVIDIA_MODEL = os.environ.get("NVIDIA_MODEL", "").strip()
NVIDIA_BASE_URL = os.environ.get("NVIDIA_BASE_URL", "").strip()
NVIDIA_DEBUG = os.environ.get("NVIDIA_DEBUG", "0").strip() in ("1", "true", "True", "yes")
# =================================================


def _clean_schema(obj):
    if not isinstance(obj, dict):
        return obj
    cleaned = {}
    for k, v in obj.items():
        if k in ("additionalProperties", "strict"):
            continue
        if isinstance(v, dict):
            cleaned[k] = _clean_schema(v)
        elif isinstance(v, list):
            cleaned[k] = [_clean_schema(i) if isinstance(i, dict) else i for i in v]
        else:
            cleaned[k] = v
    return cleaned


def _convert_tools(tools: list) -> list:
    result = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") != "function":
            continue
        func = {
            "name": tool.get("name", ""),
            "description": tool.get("description", ""),
        }
        if "parameters" in tool:
            func["parameters"] = _clean_schema(tool["parameters"])
        result.append({"type": "function", "function": func})
    return result


def _convert_tool_choice(tc):
    if tc is None:
        return "auto"
    if isinstance(tc, str):
        return tc
    if isinstance(tc, dict) and tc.get("type") == "function":
        return {"type": "function", "function": {"name": tc.get("name", "")}}
    return "auto"


def _estimate_tokens(text):
    return max(1, len(text) // 4)


def extract_messages(data: dict):
    """
    从 Responses API 请求中提取 messages 列表、tools 列表和 tool_choice。
    """
    ROLE_MAP = {"developer": "system"}
    raw_tools = data.get("tools", [])
    tools = _convert_tools(raw_tools)
    tool_choice = _convert_tool_choice(data.get("tool_choice"))

    if "input" not in data:
        if "messages" in data:
            return data["messages"], tools, tool_choice
        return [], tools, tool_choice

    inp = data["input"]
    if isinstance(inp, str):
        messages = []
        if "instructions" in data and data["instructions"]:
            messages.append({"role": "system", "content": data["instructions"]})
        messages.append({"role": "user", "content": inp})
        return messages, tools, tool_choice

    if not isinstance(inp, list):
        return [], tools, tool_choice

    messages = []
    if "instructions" in data and data["instructions"]:
        messages.append({"role": "system", "content": data["instructions"]})

    pending_tool_calls = []
    pending_reasoning = ""

    def _flush_tool_calls():
        nonlocal pending_tool_calls, pending_reasoning
        if pending_tool_calls:
            msg = {
                "role": "assistant",
                "content": "",
                "tool_calls": pending_tool_calls,
            }
            if pending_reasoning:
                msg["reasoning_content"] = pending_reasoning
            messages.append(msg)
            pending_tool_calls = []
            pending_reasoning = ""

    for item in inp:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")

        if item_type == "message":
            _flush_tool_calls()
            role = item.get("role", "user")
            role = ROLE_MAP.get(role, role)
            content = item.get("content", "")
            if isinstance(content, list):
                texts = []
                tool_calls = []
                for c in content:
                    if not isinstance(c, dict):
                        continue
                    c_type = c.get("type")
                    if c_type in ("text", "input_text", "output_text"):
                        t = c.get("text", "")
                        if t.strip():
                            texts.append(t)
                    elif c_type == "tool_call":
                        tool_calls.append({
                            "id": c.get("id", ""),
                            "type": "function",
                            "function": {
                                "name": c.get("name", ""),
                                "arguments": c.get("arguments", ""),
                            }
                        })
                text_content = "\n".join(texts)
                if tool_calls:
                    msg = {"role": role, "content": text_content or ""}
                    msg["tool_calls"] = tool_calls
                    messages.append(msg)
                elif text_content:
                    msg = {"role": role, "content": text_content}
                    messages.append(msg)
            elif isinstance(content, str) and content.strip():
                msg = {"role": role, "content": content.strip()}
                messages.append(msg)

        elif item_type == "function_call":
            pending_tool_calls.append({
                "id": item.get("call_id", ""),
                "type": "function",
                "function": {
                    "name": item.get("name", ""),
                    "arguments": item.get("arguments", ""),
                }
            })

        elif item_type == "function_call_output":
            _flush_tool_calls()
            messages.append({
                "role": "tool",
                "tool_call_id": item.get("call_id", ""),
                "content": item.get("output", ""),
            })

    _flush_tool_calls()

    # 重排消息
    reordered = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            expected_ids = {tc["id"] for tc in msg["tool_calls"]}
            tool_msgs = []
            non_tool_msgs = []
            j = i + 1
            while j < len(messages) and expected_ids:
                nxt = messages[j]
                if nxt.get("role") == "tool" and nxt.get("tool_call_id") in expected_ids:
                    expected_ids.remove(nxt["tool_call_id"])
                    tool_msgs.append(nxt)
                elif nxt.get("role") in ("system", "developer"):
                    non_tool_msgs.append(nxt)
                else:
                    break
                j += 1
            reordered.extend(non_tool_msgs)
            reordered.append(msg)
            reordered.extend(tool_msgs)
            i = j
        else:
            reordered.append(msg)
            i += 1
    messages = reordered

    return messages, tools, tool_choice


# ---- CORS ----
@app.after_request
def add_cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    return resp


# ---- 路由处理 ----
def _make_response():
    if request.method == "OPTIONS":
        return Response()

    req_data = request.get_json(silent=True) or {}
    messages, tools, tool_choice = extract_messages(req_data)
    effective_model = req_data.get("model") or NVIDIA_MODEL
    response_id = f"resp_{uuid.uuid4().hex[:12]}"

    if NVIDIA_DEBUG:
        with open(DEBUG_LOG, "a", encoding="utf-8") as f:
            f.write(f"\n--- [{__import__('datetime').datetime.now()}] ---\n")
            f.write(f"Messages:\n{json.dumps(messages, indent=2, ensure_ascii=False)}\n")
            if tools:
                f.write(f"Tools count: {len(tools)}\n")

    def generate():
        if not messages:
            yield "event: response.completed\n"
            yield "data: " + json.dumps({
                "type": "response.completed",
                "response": {
                    "id": response_id, "object": "response",
                    "status": "completed", "model": effective_model,
                    "output": [], "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                },
            }, ensure_ascii=False) + "\n\n"
            return

        # response.created
        yield "event: response.created\n"
        yield "data: " + json.dumps({
            "type": "response.created",
            "response": {
                "id": response_id, "object": "response",
                "status": "in_progress", "model": effective_model,
                "output": [], "usage": None,
            },
        }, ensure_ascii=False) + "\n\n"

        # response.in_progress
        yield "event: response.in_progress\n"
        yield "data: " + json.dumps({
            "type": "response.in_progress",
            "response": {
                "id": response_id, "object": "response",
                "status": "in_progress", "model": effective_model,
                "output": [], "usage": None,
            },
        }, ensure_ascii=False) + "\n\n"

        # 使用 OpenAI 客户端连接 NVIDIA API（已验证可用）
        client = OpenAI(
            base_url=NVIDIA_BASE_URL,
            api_key=NVIDIA_API_KEY,
        )

        # 构建请求参数
        kwargs = {
            "model": effective_model,
            "messages": messages,
            "stream": True,
        }
        if tools:
            kwargs["tools"] = tools
            if tool_choice != "auto":
                kwargs["tool_choice"] = tool_choice

        # 状态跟踪
        text_item_id = f"item_{uuid.uuid4().hex[:12]}"
        full_text = ""
        has_text = False
        text_started = False
        tool_calls_acc = {}
        input_tokens = 0
        output_tokens = 0
        seq = 0

        try:
            stream = client.chat.completions.create(**kwargs)

            for chunk in stream:
                if chunk.usage:
                    input_tokens = chunk.usage.prompt_tokens or 0
                    output_tokens = chunk.usage.completion_tokens or 0

                choice = chunk.choices[0] if chunk.choices else None
                if not choice:
                    continue

                delta = choice.delta

                # 文本内容
                content = delta.content
                if content:
                    if not text_started:
                        text_started = True
                        has_text = True
                        yield "event: response.output_item.added\n"
                        yield "data: " + json.dumps({
                            "type": "response.output_item.added",
                            "output_index": 0,
                            "item": {
                                "id": text_item_id, "type": "message",
                                "status": "in_progress", "role": "assistant",
                                "content": [],
                            },
                        }, ensure_ascii=False) + "\n\n"
                        yield "event: response.content_part.added\n"
                        yield "data: " + json.dumps({
                            "type": "response.content_part.added",
                            "item_id": text_item_id,
                            "output_index": 0, "content_index": 0,
                            "part": {"type": "text", "text": ""},
                        }, ensure_ascii=False) + "\n\n"

                    full_text += content
                    seq += 1
                    yield "event: response.output_text.delta\n"
                    yield "data: " + json.dumps({
                        "type": "response.output_text.delta",
                        "delta": content,
                        "item_id": text_item_id,
                        "output_index": 0, "content_index": 0,
                        "sequence_number": seq,
                    }, ensure_ascii=False) + "\n\n"

                # 工具调用
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in tool_calls_acc:
                            item_id = f"item_{uuid.uuid4().hex[:12]}"
                            tool_calls_acc[idx] = {
                                "id": tc.id or "",
                                "name": "",
                                "arguments": "",
                                "item_id": item_id,
                                "started": False,
                            }

                        acc = tool_calls_acc[idx]
                        if tc.function and tc.function.name:
                            acc["name"] = tc.function.name
                        if tc.id:
                            acc["id"] = tc.id
                        if tc.function and tc.function.arguments:
                            acc["arguments"] += tc.function.arguments
                            out_idx = (1 if has_text else 0) + sorted(tool_calls_acc.keys()).index(idx)

                            if not acc["started"]:
                                acc["started"] = True
                                yield "event: response.output_item.added\n"
                                yield "data: " + json.dumps({
                                    "type": "response.output_item.added",
                                    "output_index": out_idx,
                                    "item": {
                                        "id": acc["item_id"],
                                        "type": "function_call",
                                        "status": "in_progress",
                                        "call_id": acc["id"],
                                        "name": acc["name"],
                                        "arguments": "",
                                    },
                                }, ensure_ascii=False) + "\n\n"

                            yield "event: response.function_call_arguments.delta\n"
                            yield "data: " + json.dumps({
                                "type": "response.function_call_arguments.delta",
                                "item_id": acc["item_id"],
                                "output_index": out_idx,
                                "delta": tc.function.arguments,
                            }, ensure_ascii=False) + "\n\n"

            # 文本完成
            if has_text:
                yield "event: response.output_text.done\n"
                yield "data: " + json.dumps({
                    "type": "response.output_text.done",
                    "text": full_text, "item_id": text_item_id,
                    "output_index": 0, "content_index": 0,
                }, ensure_ascii=False) + "\n\n"
                yield "event: response.content_part.done\n"
                yield "data: " + json.dumps({
                    "type": "response.content_part.done",
                    "item_id": text_item_id,
                    "output_index": 0, "content_index": 0,
                    "part": {"type": "text", "text": full_text},
                }, ensure_ascii=False) + "\n\n"
                output_item_text = {
                    "id": text_item_id, "type": "message",
                    "status": "completed", "role": "assistant",
                    "content": [{"type": "text", "text": full_text}],
                }
                yield "event: response.output_item.done\n"
                yield "data: " + json.dumps({
                    "type": "response.output_item.done",
                    "output_index": 0,
                    "item": output_item_text,
                }, ensure_ascii=False) + "\n\n"

            # 工具调用完成
            output_items = []
            if has_text:
                output_items.append(output_item_text)

            for idx in sorted(tool_calls_acc.keys()):
                acc = tool_calls_acc[idx]
                out_idx = (1 if has_text else 0) + sorted(tool_calls_acc.keys()).index(idx)

                yield "event: response.function_call_arguments.done\n"
                yield "data: " + json.dumps({
                    "type": "response.function_call_arguments.done",
                    "item_id": acc["item_id"],
                    "output_index": out_idx,
                    "arguments": acc["arguments"],
                }, ensure_ascii=False) + "\n\n"

                func_item = {
                    "id": acc["item_id"],
                    "type": "function_call",
                    "status": "completed",
                    "call_id": acc["id"],
                    "name": acc["name"],
                    "arguments": acc["arguments"],
                }
                yield "event: response.output_item.done\n"
                yield "data: " + json.dumps({
                    "type": "response.output_item.done",
                    "output_index": out_idx,
                    "item": func_item,
                }, ensure_ascii=False) + "\n\n"

                output_items.append(func_item)

            # response.completed
            yield "event: response.completed\n"
            yield "data: " + json.dumps({
                "type": "response.completed",
                "response": {
                    "id": response_id, "object": "response",
                    "status": "completed", "model": effective_model,
                    "output": output_items,
                    "usage": {
                        "input_tokens": input_tokens or _estimate_tokens(json.dumps(messages)),
                        "output_tokens": output_tokens or _estimate_tokens(full_text),
                        "total_tokens": (input_tokens or _estimate_tokens(json.dumps(messages)))
                                        + (output_tokens or _estimate_tokens(full_text)),
                    },
                },
            }, ensure_ascii=False) + "\n\n"

        except Exception as e:
            err_msg = f"NVIDIA API error: {type(e).__name__}: {e}"
            if NVIDIA_DEBUG:
                with open(DEBUG_LOG, "a", encoding="utf-8") as f:
                    f.write(f"ERROR: {err_msg}\n")
            yield "event: response.failed\n"
            yield "data: " + json.dumps({
                "type": "response.failed",
                "response": {
                    "id": response_id, "object": "response",
                    "status": "failed", "model": effective_model,
                    "error": {"message": err_msg, "type": "upstream_error"},
                    "output": [], "usage": None,
                },
            }, ensure_ascii=False) + "\n\n"

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---- 注册路由 ----
app.add_url_rule("/responses", "responses", _make_response, methods=["POST", "OPTIONS"])
app.add_url_rule("/v1/responses", "v1_responses", _make_response, methods=["POST", "OPTIONS"])
app.add_url_rule("/v1/chat/completions", "v1_chat", _make_response, methods=["POST", "OPTIONS"])


if __name__ == "__main__":
    key, source = _ensure_api_key()
    if not key:
        sys.exit(1)
    globals()["NVIDIA_API_KEY"] = key

    from waitress import serve
    print("codex_nvidia_proxy starting ...")
    print(f"   Endpoint: http://127.0.0.1:5000")
    print(f"   Model:    {NVIDIA_MODEL}")
    print(f"   Key:      {source}")
    print(f"   Debug:    {'ON' if NVIDIA_DEBUG else 'OFF'}")
    print(f"   Routes:   /responses, /v1/responses, /v1/chat/completions")
    serve(app, host="127.0.0.1", port=5000, threads=4)
