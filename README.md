# NVIDIA API 网关

一个轻量级的 **NVIDIA NIM API 代理网关**，将 OpenAI 兼容的 [NVIDIA NIM API](https://build.nvidia.com/) 封装为本地 HTTP 服务，提供多 Key 负载均衡、流式对话、工具调用等功能。

## 核心功能

- **多 Key 自动轮换**: 支持配置最多 20 个 API Key，遇到速率限制 (HTTP 429) 自动切换
- **流式响应转发**: 完整的 Server-Sent Events (SSE) 流式代理，实时推送生成内容
- **OpenAI 兼容接口**: 提供 `/v1/chat/completions` 和 `/v1/responses` 端点，无缝对接 OpenAI SDK
- **工具调用支持**: 转发 function calling 请求，支持工具定义与调用结果回传
- **启动健康检查**: 启动时自动检测所有 Key 的可用性，确保服务正常
- **请求日志记录**: 支持调试模式，记录所有请求与错误日志

## 项目结构

```
nvidia-api-gateway/
├── main.py              # 主程序入口，包含所有路由与代理逻辑
├── requirements.txt     # Python 依赖清单
├── .nvidia_env          # API Key 与配置模板
└── logs/                # 运行时日志输出目录
```

## 环境要求

| 依赖 | 最低版本 | 说明 |
|------|----------|------|
| Python | 3.10+ | 需支持 `dataclass` 类型注解 |
| Flask | 2.3.0+ | Web 框架 |
| OpenAI | 1.0.0+ | NVIDIA API 客户端 |
| Waitress | 2.1.0+ | 生产级 WSGI 服务器 |
| Requests | 2.28.0+ | HTTP 健康检查 |

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/Zhao-Dream/nvidia-api-gateway.git
cd nvidia-api-gateway
```

### 2. 安装依赖

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # Linux / macOS
pip install -r requirements.txt
```

### 3. 配置 API Key

编辑 `.nvidia_env` 文件，填入你的 NVIDIA API Key：

```env
# NVIDIA_API_KEY
NVIDIA_API_KEY_1=nvapi-你的第一个Key
NVIDIA_API_KEY_2=nvapi-你的第二个Key
NVIDIA_API_KEY_3=nvapi-你的第三个Key

# 使用的模型（可选，默认为 deepseek-ai/deepseek-v4-pro）
NVIDIA_MODEL=deepseek-ai/deepseek-v4-pro

# API 基础 URL（可选，通常无需修改）
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1

# 调试模式：1 启用 / 0 禁用（可选）
NVIDIA_DEBUG=0
```

> **获取 API Key**: 访问 [build.nvidia.com](https://build.nvidia.com/)，登录后点击任意模型 → "Get API Key"。

### 4. 启动服务

```bash
python main.py
```

启动后访问：
- **网关地址**: `http://127.0.0.1:5000`
- **健康检查**: `http://127.0.0.1:5000/health`

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/health` | 服务健康检查 |
| `POST` | `/v1/chat/completions` | OpenAI Chat Completions 兼容接口 |
| `POST` | `/v1/responses` | OpenAI Responses API 兼容接口 |
| `POST` | `/responses` | 同上（别名路由） |
| `OPTIONS` | `/v1/chat/completions` | CORS 预检请求 |

### 使用示例

```python
from openai import OpenAI

# 连接到本地网关
client = OpenAI(
    base_url="http://127.0.0.1:5000/v1",
    api_key="任意值即可（本地网关不使用此字段）"
)

# 流式对话
stream = client.chat.completions.create(
    model="deepseek-ai/deepseek-v4-pro",
    messages=[{"role": "user", "content": "你好，请介绍一下自己"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

## 多 Key 机制

程序支持配置多个 API Key，启动时自动检测每个 Key 的健康状态：

```
============================================================
  正在检测所有 API Key 的健康状态...
============================================================
  [0] nvapi-xxxx....xxxxxx -> ✅ 可用
  [1] nvapi-yyyy....yyyyyy -> ✅ 可用
  [2] nvapi-zzzz....zzzzzz -> ⚠️ 认证失败

  共检测 3 个 Key，可用 2 个
============================================================
```

**自动切换规则**:
- 当 NVIDIA API 返回 HTTP 429 (速率限制) 时，自动轮换到下一个可用 Key
- Key 切换通过线程锁保证并发安全
- 检测结果日志保存在 `logs/key_check.log`

## 启动流程

项目启动时按以下步骤执行：

1. **加载配置**: 从 `.nvidia_env` 或系统环境变量读取 Key 和模型参数
2. **创建日志目录**: 清空旧的调试日志文件
3. **Key 有效性验证**: 并行检测所有 Key 的健康状态
4. **选择默认 Key**: 自动定位第一个可用 Key 作为当前 Key
5. **启动 HTTP 服务**: 以 Waitress 多线程模式监听 `127.0.0.1:5000`

## 配置项

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `NVIDIA_API_KEY_1` ~ `NVIDIA_API_KEY_20` | 无 | 可配置最多 20 个 API Key |
| `NVIDIA_MODEL` | `deepseek-ai/deepseek-v4-pro` | 默认模型名称 |
| `NVIDIA_BASE_URL` | `https://integrate.api.nvidia.com/v1` | NVIDIA API 基础地址 |
| `NVIDIA_DEBUG` | `0` | 调试模式开关 (1/0) |
| `NVIDIA_LOG_DIR` | `./logs` | 日志输出目录 |

## 日志管理

调试模式下，所有日志统一存储在 `D:\Codex` 或项目 `logs/` 目录下：

| 文件 | 说明 |
|------|------|
| `nvidia_proxy_debug.log` | 调试日志：记录完整请求/响应内容 |
| `key_check.log` | Key 健康检测结果 |

## 技术栈

- **Web 框架**: Flask (开发) + Waitress (生产)
- **AI 客户端**: OpenAI Python SDK
- **流式传输**: Server-Sent Events (SSE)
- **并发控制**: Python threading.Lock

## 常见问题

**Q: 如何确认服务正常运行？**
访问 `http://127.0.0.1:5000/health`，返回 `{"status": "ok"}` 即表示正常。

**Q: 启动时提示 "所有 Key 均不可用"？**
检查网络连接是否可达 `https://integrate.api.nvidia.com`，并确认 `.nvidia_env` 中的 Key 格式正确。

**Q: 如何切换模型？**
修改 `.nvidia_env` 中的 `NVIDIA_MODEL` 值，或设置系统环境变量后重启服务。

## 许可证

本项目仅供学习与个人使用，NVIDIA API Key 请从官方渠道获取。
