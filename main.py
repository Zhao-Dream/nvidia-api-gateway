
import sys
import os
import json
import uuid
import getpass

from dataclasses import dataclass, field
import threading

from flask import Flask, request, Response
from openai import OpenAI

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")  # 日志统一存放目录


def _load_dotenv():
    """加载 .env 文件到 os.environ（不覆盖已有的系统环境变量）"""
    env_file = os.path.join(BASE_DIR, ".nvidia_env")
    if not os.path.exists(env_file):
        return
    had_key = "NVIDIA_API_KEYS" in os.environ
    with open(env_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip().strip("\"'")
            if key and key not in os.environ:
                os.environ[key] = val
    if "NVIDIA_API_KEYS" in os.environ:
        os.environ["_NVIDIA_KEY_SOURCE"] = "sys" if had_key else "dotenv"




# ============================================================
# 从 .nvidia_env 解析所有 Key（支持多个 Key）
# 格式:
# NVIDIA_API_KEY_1=nvapi-******
# NVIDIA_API_KEY_2=nvapi-******
# ... ...
# NVIDIA_API_KEY_20=nvapi-******
# ============================================================
def _parse_all_keys() -> list:
    """从环境变量中解析所有 NVIDIA API Key"""
    keys = []
    seen = set()
    for i in range(1, 21):
        k = os.environ.get(f"NVIDIA_API_KEY_{i}", "").strip()
        if k and k not in seen:
            keys.append(k)
            seen.add(k)
        else:
            break
    return keys


def _ensure_api_keys():
    """确保至少有一个 API Key，若一个都没有则交互输入"""
    all_keys = _parse_all_keys()
    if all_keys:
        src = os.environ.get("_NVIDIA_KEY_SOURCE", "")
        if src == "sys":
            return all_keys, "系统环境变量"
        return all_keys, ".nvidia_env"

    print("=" * 60)
    print("  未检测到任何 NVIDIA API Key")
    print("=" * 60)
    print()
    print("  从 https://build.nvidia.com/ 获取 API Key")
    print("  登录后点击任一模型 → Get API Key")
    print()
    print("  你也可以设置系统环境变量 NVIDIA_API_KEYS 后重试")
    print()

    try:
        key = getpass.getpass("  请输入你的 NVIDIA API Key: ").strip()
    except (EOFError, KeyboardInterrupt):
        key = ""

    if not keys:
        print()
        print("  ERROR: 未输入 API Key，程序退出。")
        print()
        input("  按 Enter 退出 ...")
        sys.exit(1)

    # 保存到 .nvidia_env
    env_file = os.path.join(BASE_DIR, ".nvidia_env")
    all_keys = [key]
    existing = {}
    if os.path.exists(env_file):
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                existing[k.strip()] = f"{k.strip()}={v.strip()}"
    existing["NVIDIA_API_KEYS"] = f"NVIDIA_API_KEYS={key}"

    with open(env_file, "w", encoding="utf-8") as f:
        for line in existing.values():
            f.write(line + "\\n")
        if "NVIDIA_MODEL" not in existing:
            f.write("NVIDIA_MODEL=deepseek-ai/deepseek-v4-pro\\n")

    os.environ["NVIDIA_API_KEYS"] = key
    print()
    print(f"  API Key 已保存到: {env_file} (支持多个 Key 用逗号分隔)")
    print()
    return all_keys, ".nvidia_env (已保存)"

_load_dotenv()

app = Flask(__name__)


# ============================================================
# 配置类 ---- 所有配置项统一封装到 dataclass
# ============================================================
@dataclass
class AppConfig:
    """应用全局配置，所有配置项统一管理"""
    # --- API Key 列表（多 Key 支持）---
    api_keys: list = field(default_factory=list)
    # --- 当前使用的 Key 索引 ---
    current_key_index: int = 0
    # --- 线程锁，保证 Key 切换的线程安全 ---
    key_lock: threading.Lock = field(default_factory=threading.Lock)
    # --- 模型名称 ---
    model: str = ""
    # --- 基础 URL ---
    base_url: str = ""
    # --- 调试模式 ---
    debug: bool = False
    # --- 日志目录 ---
    log_dir: str = ""
    # --- 调试日志文件 ---
    debug_log: str = ""

    @classmethod
    def create(cls) -> "AppConfig":
        """从环境变量和文件创建配置实例"""
        # 解析所有 API Key
        api_keys = _parse_all_keys()

        # 模型名称
        model = os.environ.get("NVIDIA_MODEL", "").strip()
        if not model:
            model = "deepseek-ai/deepseek-v4-pro"

        # 基础 URL
        base_url = os.environ.get("NVIDIA_BASE_URL", "").strip()
        if not base_url:
            base_url = "https://integrate.api.nvidia.com/v1"

        # 调试模式
        debug_val = os.environ.get("NVIDIA_DEBUG", "0").strip()
        debug = debug_val in ("1", "true", "True", "yes")

        # 日志目录
        log_dir = os.environ.get("NVIDIA_LOG_DIR", "").strip()
        if not log_dir:
            log_dir = LOG_DIR

        return cls(
            api_keys=api_keys,
            model=model,
            base_url=base_url,
            debug=debug,
            log_dir=log_dir,
            debug_log=os.path.join(log_dir, "nvidia_proxy_debug.log"),
        )

    @property
    def current_key(self) -> str:
        """获取当前使用的 API Key"""
        if not self.api_keys:
            return ""
        return self.api_keys[self.current_key_index]

    def rotate_key(self) -> bool:
        """切换到下一个 Key，返回 True 表示切换成功"""
        with self.key_lock:
            if len(self.api_keys) <= 1:
                return False
            old_idx = self.current_key_index
            self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
            new_preview = self.current_key[:20] + "..." if len(self.current_key) > 20 else self.current_key
            print(f"  [Key切换] 索引 {old_idx} -> {self.current_key_index}, Key: {new_preview}")
            return True

    def get_openai_client(self, key_index: int | None = None) -> OpenAI:
        """根据指定索引创建 OpenAI 客户端"""
        idx = key_index if key_index is not None else self.current_key_index
        if idx >= len(self.api_keys):
            idx = 0
        return OpenAI(
            base_url=self.base_url,
            api_key=self.api_keys[idx],
        )


# ============================================================
# 全局配置实例
# ============================================================
config = AppConfig.create()


# ============================================================
# CORS 头注入 ---- 为所有响应添加跨域头
# ============================================================
@app.after_request
def _add_cors_headers(response: Response) -> Response:
    """为所有响应添加 CORS 跨域头"""
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, PATCH, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, x-request-id"
    response.headers["Access-Control-Max-Age"] = "86400"
    return response


# ============================================================
# GET /health 健康检查端点
# ============================================================
@app.route("/health", methods=["GET"])
def health_check():
    """返回服务状态，确保服务可用"""
    return {"status": "ok"}, 200, {"Content-Type": "application/json"}


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
    effective_model = req_data.get("model") or config.model
    response_id = f"resp_{uuid.uuid4().hex[:12]}"

    if config.debug:
        with open(config.debug_log, "a", encoding="utf-8") as f:
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
            base_url=config.base_url,
            api_key=config.current_key,
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
            # 检测 HTTP 429（超出速率限制），自动切换到下一个 Key 并重试
            is_429 = False
            if hasattr(e, "status_code") and e.status_code == 429:
                is_429 = True
            elif "429" in str(e) or "rate" in str(e).lower() or "quota" in str(e).lower():
                is_429 = True

            if is_429 and config.rotate_key():
                err_msg = f"NVIDIA API 429 (速率限制)，已切换 Key 索引 -> {config.current_key_index}，请重试"
            else:
                err_msg = f"NVIDIA API error: {type(e).__name__}: {e}"

            if config.debug:
                with open(config.debug_log, "a", encoding="utf-8") as f:
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
    import requests as _requests

    # ========================================================
    # 第1步：创建日志目录 + 清空之前的日志
    # ========================================================
    os.makedirs(config.log_dir, exist_ok=True)
    # 清空调试日志
    with open(config.debug_log, "w", encoding="utf-8") as _f:
        _f.write(f"=== 启动时间: {__import__('datetime').datetime.now().isoformat()} ===\n")
    # print(f"[日志] 已清空并初始化: {config.debug_log}")

    # ========================================================
    # 第2步：确保至少有一个 Key
    # ========================================================
    keys, source = _ensure_api_keys()
    if not keys:
        sys.exit(1)
    # 重新更新 config（因为可能交互输入了新 Key）
    config.api_keys = keys
    config.current_key_index = 0

    # ========================================================
    # 第3步：检测所有 Key 的健康状态（GET https://.../health 或 /v1/models）
    # ========================================================
    print()
    print("=" * 60)
    print("  正在检测所有 API Key 的健康状态...")
    print("=" * 60)
    key_check_results = []
    for idx, key in enumerate(config.api_keys):
        preview = key[:10] + "...." + key[-8:] if len(key) > 18 else key
        try:
            # NVIDIA API 健康检测：尝试获取 models 列表
            resp = _requests.get(
                f"{config.base_url}/models",
                headers={"Authorization": f"Bearer {key}"},
                timeout=10,
            )
            if resp.status_code == 200:
                status = "✔ 可用"
                key_check_results.append((idx, preview, True, f"HTTP {resp.status_code}"))
            elif resp.status_code == 429:
                status = "✘ 429 限流"
                key_check_results.append((idx, preview, False, f"HTTP 429 (速率限制)"))
            elif resp.status_code in (401, 403):
                status = "✘ 认证失败"
                key_check_results.append((idx, preview, False, f"HTTP {resp.status_code} (认证失败)"))
            else:
                status = f"⚠ HTTP {resp.status_code}"
                key_check_results.append((idx, preview, True, f"HTTP {resp.status_code}"))
        except Exception as ex:
            status = f"✘ 连接失败"
            key_check_results.append((idx, preview, False, f"{type(ex).__name__}"))

        print(f"  [{idx}] {preview} -> {status}")

    # 统计可用 Key
    available_keys = [r for r in key_check_results if r[2]]
    print()
    print(f"  共检测 {len(config.api_keys)} 个 Key，可用 {len(available_keys)} 个")
    if not available_keys:
        print()
        print("  ERROR: 所有 Key 均不可用，无法启动服务！")
        print()
        input("  按 Enter 退出 ...")
        sys.exit(1)

    # 将当前索引对准第一个可用 Key
    if not key_check_results[config.current_key_index][2]:
        for r in key_check_results:
            if r[2]:
                config.current_key_index = r[0]
                print(f"  [自动] 当前 Key 切换至索引 {r[0]}: {r[1]}")
                break

    # 保存检测结果日志
    check_log_path = os.path.join(config.log_dir, "key_check.log")
    with open(check_log_path, "w", encoding="utf-8") as _f:
        _f.write(f"=== Key 健康检测 ({__import__('datetime').datetime.now().isoformat()}) ===\n")
        for idx, preview, ok, detail in key_check_results:
            _f.write(f"[{idx}] {preview} -> {'OK' if ok else 'FAIL'}: {detail}\n")
        _f.write(f"可用: {len(available_keys)}/{len(config.api_keys)}\n")

    # ========================================================
    # 第4步：启动服务
    # ========================================================
    print()
    from waitress import serve
    print("codex_nvidia_proxy 启动中 ...")
    print(f"   Endpoint:  http://127.0.0.1:5000")
    print(f"   Health:    http://127.0.0.1:5000/health")
    print(f"   Model:     {config.model}")
    print(f"   Key总数:   {len(config.api_keys)} (来源: {source})")
    print(f"   Key索引:   {config.current_key_index}")
    print(f"   Debug:     {'ON' if config.debug else 'OFF'}")
    print()
    serve(app, host="127.0.0.1", port=5000, threads=4)