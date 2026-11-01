#!/usr/bin/env python3
"""
MCP 端到端测试脚本（stdio 模式）。

通过标准输入/输出与 MCP Server 完成完整的 JSON-RPC 握手：
  initialize → initialized → tools/list → tools/call

用法：
    python scripts/mcp/test_client.py              # 调用 search_esp_docs
    python scripts/mcp/test_client.py --tool search_esp_code --query "esp_camera_init"

（在 Docker 里：docker exec -it esp-rag python scripts/mcp/test_client.py）
"""

import json
import subprocess
import sys
import argparse

SERVER = [sys.executable, "scripts/mcp/server.py"]


def rpc(proc, method, params, req_id):
    msg = {"jsonrpc": "2.0", "id": req_id, "method": method}
    if params is not None:
        msg["params"] = params
    proc.stdin.write(json.dumps(msg) + "\n")
    proc.stdin.flush()
    while True:
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError("server closed stdout")
        line = line.strip()
        if not line:
            continue
        try:
            resp = json.loads(line)
        except json.JSONDecodeError:
            # 忽略非 JSON 行（例如日志），继续读下一行
            continue
        if resp.get("id") == req_id:
            return resp


def main():
    ap = argparse.ArgumentParser(description="MCP stdio test client")
    ap.add_argument("--tool", default="search_esp_docs",
                    help="tool to call (search_esp_docs / search_esp_code / search_esp)")
    ap.add_argument("--query", default="ESP32-S3 clock source PLL XTAL",
                    help="query text")
    ap.add_argument("--top", type=int, default=3, help="top_n")
    args = ap.parse_args()

    proc = subprocess.Popen(
        SERVER, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, text=True, cwd=".",
    )

    try:
        # 1. initialize
        r = rpc(proc, "initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test-client", "version": "1.0"},
        }, 1)
        print("=== initialize ===")
        print(json.dumps(r.get("result", {}).get("serverInfo", {}), ensure_ascii=False))

        # 2. notifications/initialized
        proc.stdin.write(json.dumps(
            {"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n")
        proc.stdin.flush()

        # 3. tools/list
        r = rpc(proc, "tools/list", None, 2)
        tools = r.get("result", {}).get("tools", [])
        print("\n=== tools/list ===")
        for t in tools:
            print(f"  - {t['name']}: {t.get('description', '')[:60]}")

        # 4. tools/call
        call_args = {"query": args.query, "top_n": args.top}
        if args.tool == "search_esp_docs":
            call_args["doc_type"] = None
        r = rpc(proc, "tools/call", {
            "name": args.tool,
            "arguments": call_args,
        }, 3)
        print(f"\n=== tools/call {args.tool} ===")
        content = r.get("result", {}).get("content", [])
        if r.get("error"):
            print("ERROR:", json.dumps(r["error"], ensure_ascii=False))
        for c in content:
            if c.get("type") == "text":
                print(c.get("text", ""))
                break
    finally:
        proc.terminate()
        proc.wait()


if __name__ == "__main__":
    main()
