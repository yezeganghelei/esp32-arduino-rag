# MCP 客户端配置指南

> ESP32 RAG 的 MCP Server 可以让支持 MCP 协议的 AI 客户端（Claude Desktop、Cursor、Trae 等）
> 直接调用 ESP32 知识库检索。本文给出各客户端的具体配置示例。

---

## 一、可用工具

MCP Server 名：`esp-rag`，暴露 3 个检索工具（调用统一内核 `rag_api`，与网页/命令行一致）：

| 工具 | 参数 | 说明 |
|---|---|---|
| `search_esp_docs` | `query`, `top_n=10`, `doc_type=None` | 检索**文档库**（datasheet / TRM / guide 等）|
| `search_esp_code` | `query`, `top_n=10` | 检索**代码库**（C/C++/Python 示例）|
| `search_esp` | `query`, `top_n=10` | **双库检索**（文档 + 代码合并）|

- `doc_type` 可选值：`datasheet`、`trm`、`guide`、`api_reference`、`safety`、`release_notes`、`specification`、`docs`。
- 检索词建议用**英文技术词**（如 `clock source PLL XTAL`）召回更好；代码检索建议用函数名/标识符（如 `esp_camera_init camera_config_t`）。
- MCP 只做**原始检索**（返回带来源的文本片段），不调用 LLM，因此**不需要 LLM_API_KEY**。

---

## 二、两种运行模式

### 模式 1：SSE（远程，推荐 Docker 部署）

MCP Server 常驻运行，客户端通过网络连接。

```bash
# Docker 里（docker-compose 已内置，all 模式会同时启动）
docker-compose up -d --build
# MCP SSE 地址
http://<host>:8001/sse
```

### 模式 2：stdio（本地）

MCP Server 由 AI 客户端以子进程方式拉起，通过标准输入/输出通信。

```powershell
# 需要本地 Python 环境 + 依赖 + 模型/向量库
cd D:\09.WorkSpace\esp-rag
.\.venv\Scripts\python.exe scripts\mcp\server.py
```

> 本地 stdio 需要先 `pip install mcp`（已加入 `requirements.txt`），且模型目录 `models/`、
> 向量库 `.chroma_esp32_all/`、`.chroma_esp32_code/` 都在本机。

---

## 三、客户端配置示例

### 1. Claude Desktop

配置文件路径：
- Windows：`%APPDATA%\Claude\claude_desktop_config.json`
- macOS：`~/Library/Application Support/Claude/claude_desktop_config.json`

**A. SSE 模式（连接 Docker 容器，推荐）**

```json
{
  "mcpServers": {
    "esp-rag": {
      "url": "http://localhost:8001/sse"
    }
  }
}
```

**B. stdio 模式（本地 Python）**

```json
{
  "mcpServers": {
    "esp-rag": {
      "command": "python",
      "args": ["D:\\09.WorkSpace\\esp-rag\\scripts\\mcp\\server.py"],
      "env": {
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_OFFLINE": "1"
      }
    }
  }
}
```

### 2. Cursor

**A. 项目级配置**：在项目根目录创建 `.cursor/mcp.json`

```json
{
  "mcpServers": {
    "esp-rag": {
      "url": "http://localhost:8001/sse"
    }
  }
}
```

**B. 全局配置**：Cursor → Settings → MCP → Add new MCP server，或编辑 `~/.cursor/mcp.json`

```json
{
  "mcpServers": {
    "esp-rag": {
      "command": "python",
      "args": ["D:\\09.WorkSpace\\esp-rag\\scripts\\mcp\\server.py"]
    }
  }
}
```

### 3. 其他支持 MCP 的客户端（Trae / Windsurf 等）

大部分客户端的 `mcpServers` 格式与上面一致，SSE 用 `url`，stdio 用 `command` + `args`：

```json
{
  "mcpServers": {
    "esp-rag": {
      "url": "http://<服务器IP>:8001/sse"
    }
  }
}
```

---

## 四、Docker stdio 方式（进阶）

如果本地不想装 Python 环境，也可以用 Docker 以 stdio 方式运行（需挂载模型和向量库）：

```powershell
docker run -i --rm `
  -v D:\09.WorkSpace\esp-rag\models:/app/models:ro `
  -v D:\09.WorkSpace\esp-rag\source:/app/source:ro `
  -v D:\09.WorkSpace\esp-rag\.chroma_esp32_all:/app/.chroma_esp32_all `
  -v D:\09.WorkSpace\esp-rag\.chroma_esp32_code:/app/.chroma_esp32_code `
  -v D:\09.WorkSpace\esp-rag\source_code:/app/source_code:ro `
  -v D:\09.WorkSpace\esp-rag\config.yaml:/app/config.yaml:ro `
  esp-rag:latest python scripts/mcp/server.py
```

对应客户端 stdio 配置：

```json
{
  "mcpServers": {
    "esp-rag": {
      "command": "docker",
      "args": [
        "run", "-i", "--rm",
        "-v", "D:\\09.WorkSpace\\esp-rag\\models:/app/models:ro",
        "-v", "D:\\09.WorkSpace\\esp-rag\\source:/app/source:ro",
        "-v", "D:\\09.WorkSpace\\esp-rag\\.chroma_esp32_all:/app/.chroma_esp32_all",
        "-v", "D:\\09.WorkSpace\\esp-rag\\.chroma_esp32_code:/app/.chroma_esp32_code",
        "-v", "D:\\09.WorkSpace\\esp-rag\\source_code:/app/source_code:ro",
        "-v", "D:\\09.WorkSpace\\esp-rag\\config.yaml:/app/config.yaml:ro",
        "esp-rag:latest", "python", "scripts/mcp/server.py"
      ]
    }
  }
}
```

---

## 五、验证是否连接成功

1. 启动 Docker（SSE 模式）：`docker-compose up -d --build`
2. 检查端口：`curl http://localhost:8001/sse`，应返回 `event: endpoint` + `data: /messages/?session_id=...`
3. 在 AI 客户端里问一句测试（如 "ESP32-S3 有哪些 SPI 控制器？"），客户端应自动调用 `search_esp_docs` / `search_esp` 工具并返回带来源的检索结果。

---

## 六、常见问题

| 问题 | 解决 |
|---|---|
| 客户端提示 "Could not connect to server" | 确认 Docker 容器在跑（`docker ps`），端口 8001 已映射 |
| stdio 模式报 `No module named 'mcp'` | 本地 venv 里执行 `pip install -r requirements.txt`（或 `pip install "mcp>=1.6.0"`）|
| 检索结果为空 | 确认向量库目录已挂载，且 `config.yaml` 路径正确 |
| 想要自定义 top_n / doc_type | 在工具调用参数里传 `top_n`、`doc_type` 即可 |
