# Codex NVIDIA Proxy

一个轻量级的 NVIDIA API 代理服务，将 NVIDIA NIM API 转换为兼容 OpenAI Chat Completions 和 Responses API 格式，方便接入 Claude Code、Cursor 等支持 OpenAI 协议的开发工具。

## 功能特性

- 支持 `/responses` 接口（OpenAI Responses API）
- 支持 `/v1/responses` 和 `/v1/chat/completions` 接口
- 自动处理函数调用（Function Calling）
- 支持工具调用（Tools）
- CORS 跨域支持
- SSE 流式响应
- 灵活的 API Key 配置方式
- 调试日志功能

## 环境要求

- Python 3.8+
- NVIDIA API Key（从 [build.nvidia.com](https://build.nvidia.com/) 获取）

## 安装

1. 克隆项目
```bash
git clone https://github.com/yourusername/codex-nvidia-proxy.git
cd codex-nvidia-proxy
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

3. 配置 API Key

方式一：创建 `.nvidia_env` 文件
```bash
echo NVIDIA_API_KEY=your_api_key_here > .nvidia_env
echo NVIDIA_MODEL=nvidia/llama-3.1-nemotron-70b-instruct >> .nvidia_env
```

方式二：设置系统环境变量
```bash
# Linux/macOS
export NVIDIA_API_KEY=your_api_key_here
export NVIDIA_MODEL=nvidia/llama-3.1-nemotron-70b-instruct

# Windows PowerShell
$env:NVIDIA_API_KEY="your_api_key_here"
$env:NVIDIA_MODEL="nvidia/llama-3.1-nemotron-70b-instruct"
```

## 使用方法

### 启动服务

```bash
python codex_nvidia_proxy.py
```

服务启动后显示：
```
codex_nvidia_proxy starting ...
   Endpoint: http://127.0.0.1:5000
   Model:    nvidia/llama-3.1-nemotron-70b-instruct
   Key:      .nvidia_env (已保存)
   Debug:    OFF
   Routes:   /responses, /v1/responses, /v1/chat/completions
```

### 配置开发工具

#### Claude Code / Claude CLI
```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:5000
export ANTHROPIC_API_KEY=any_key  # 任意值即可
claude
```

#### Cursor
在 Cursor 设置中配置：
- API Base URL: `http://127.0.0.1:5000`
- API Key: `any_key`

### API 端点

| 端点 | 说明 |
|------|------|
| `POST /responses` | OpenAI Responses API |
| `POST /v1/responses` | OpenAI Responses API (v1) |
| `POST /v1/chat/completions` | OpenAI Chat Completions API |

## 配置选项

| 环境变量 | 说明 | 默认值 |
|----------|------|--------|
| `NVIDIA_API_KEY` | NVIDIA API Key | 必填 |
| `NVIDIA_MODEL` | 使用的模型 | `nvidia/llama-3.1-nemotron-70b-instruct` |
| `NVIDIA_BASE_URL` | API Base URL | `https://integrate.api.nvidia.com/v1` |
| `NVIDIA_DEBUG` | 开启调试日志 | `0` |

### 调试模式

开启调试模式后会生成 `nvidia_proxy_debug.log` 日志文件：

```bash
# Linux/macOS
export NVIDIA_DEBUG=1

# Windows PowerShell
$env:NVIDIA_DEBUG="1"
```

## 快速测试

使用 curl 测试服务：

```bash
curl -X POST http://127.0.0.1:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer any_key" \
  -d '{
    "model": "nvidia/llama-3.1-nemotron-70b-instruct",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## 项目结构

```
codex-nvidia-proxy/
├── codex_nvidia_proxy.py   # 主程序
├── requirements.txt        # 依赖列表
├── .nvidia_env.example     # 配置文件示例
├── README.md               # 本文件
└── LICENSE                # MIT 许可证
```

## 可用模型

推荐在 [build.nvidia.com](https://build.nvidia.com/) 查看可用模型，热门模型包括：

- `nvidia/llama-3.1-nemotron-70b-instruct`
- `nvidia/llama-3.3-nemotron-70b-instruct`
- `nvidia/nemotron-4-340b-instruct`
- `mistralai/mixtral-8x7b-instruct-v0.1`
- `google/gemma-2-27b-it`

## 常见问题

### Q: 获取 NVIDIA API Key？
访问 [build.nvidia.com](https://build.nvidia.com/)，登录后点击任意模型，选择 "Get API Key"。

### Q: 支持流式响应吗？
支持，所有接口都默认使用 SSE 流式响应。

### Q: 如何处理函数调用？
代理会自动转换工具调用格式，支持 OpenAI Tools 协议。

## 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件。

## 贡献

欢迎提交 Issue 和 Pull Request！
