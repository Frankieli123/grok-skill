import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from typing import Any


def _compact_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"), sort_keys=False)


def _default_user_config_path() -> str:
    home = os.path.expanduser("~")
    return os.path.join(home, ".codex", "config", "grok-search.json")


def _skill_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def _default_skill_config_paths() -> list[str]:
    root = _skill_root()
    return [
        os.path.join(root, "config.json"),
        os.path.join(root, "config.local.json"),
    ]


def _normalize_api_key(api_key: str) -> str:
    api_key = api_key.strip()
    if not api_key:
        return ""
    placeholder = {"YOUR_API_KEY", "API_KEY", "CHANGE_ME", "REPLACE_ME"}
    if api_key.upper() in placeholder:
        return ""
    return api_key


def _normalize_base_url_value(base_url: str) -> str:
    base_url = base_url.strip()
    if not base_url:
        return ""
    placeholder = {
        "https://your-grok-endpoint.example",
        "YOUR_BASE_URL",
        "BASE_URL",
        "CHANGE_ME",
        "REPLACE_ME",
    }
    if base_url.upper() in placeholder:
        return ""
    return base_url


def _load_json_file(path: str) -> dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            value = json.load(f)
    except FileNotFoundError:
        return {}
    if not isinstance(value, dict):
        raise ValueError("config must be a JSON object")
    return value


def _normalize_base_url(base_url: str) -> str:
    base_url = base_url.strip().rstrip("/")
    if base_url.endswith("/v1"):
        return base_url[: -len("/v1")]
    return base_url


def _coerce_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None

    # 尝试直接解析 JSON
    if text.startswith("{") and text.endswith("}"):
        try:
            value = json.loads(text)
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            pass

    # 尝试从 Markdown 代码块中提取 JSON
    json_match = re.search(r'```json\s*\n(.*?)\n```', text, re.DOTALL)
    if json_match:
        try:
            value = json.loads(json_match.group(1))
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            pass

    # 尝试查找包含 content 和 sources 的 JSON 对象
    # 使用更宽松的正则表达式
    json_match = re.search(r'\{"content":', text)
    if json_match:
        try:
            # 找到最外层的完整 JSON 对象
            start = json_match.start()
            brace_count = 0
            in_string = False
            escape_next = False

            for i in range(start, len(text)):
                char = text[i]

                if escape_next:
                    escape_next = False
                    continue

                if char == '\\':
                    escape_next = True
                    continue

                if char == '"' and not escape_next:
                    in_string = not in_string
                    continue

                if not in_string:
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_str = text[start:i+1]
                            value = json.loads(json_str)
                            return value if isinstance(value, dict) else None
        except (json.JSONDecodeError, ValueError):
            pass

    return None


def _extract_urls(text: str) -> list[str]:
    urls = re.findall(r"https?://[^\s)\]}>\"']+", text)
    seen: set[str] = set()
    out: list[str] = []
    for url in urls:
        url = url.rstrip(".,;:!?'\"")
        if url and url not in seen:
            seen.add(url)
            out.append(url)
    return out


def _load_json_env(var_name: str) -> dict[str, Any]:
    raw = os.environ.get(var_name, "").strip()
    if not raw:
        return {}
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{var_name} must be a JSON object")
    return value


def _parse_json_object(raw: str, *, label: str) -> dict[str, Any]:
    raw = raw.strip()
    if not raw:
        return {}
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _parse_sse_response(raw: str) -> dict[str, Any]:
    """解析 SSE 流式响应，合并所有 delta.content"""
    lines = raw.strip().split("\n")
    content_parts = []
    model_name = ""

    for line in lines:
        line = line.strip()
        if not line or line == "data: [DONE]":
            continue

        if line.startswith("data: "):
            json_str = line[6:]  # 移除 "data: " 前缀
            try:
                chunk = json.loads(json_str)

                # 提取模型名称
                if not model_name and "model" in chunk:
                    model_name = chunk["model"]

                # 提取 content
                choices = chunk.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        content_parts.append(content)
            except json.JSONDecodeError:
                continue

    # 合并所有内容
    full_content = "".join(content_parts)

    # 构造标准响应格式
    return {
        "choices": [
            {
                "message": {
                    "content": full_content
                }
            }
        ],
        "model": model_name
    }


def _request_chat_completions(
    *,
    base_url: str,
    api_key: str,
    model: str,
    query: str,
    timeout_seconds: float,
    extra_headers: dict[str, Any],
    extra_body: dict[str, Any],
) -> dict[str, Any]:
    url = f"{_normalize_base_url(base_url)}/v1/chat/completions"

    system = (
        "You are a web research assistant. Use live web search/browsing when answering. "
        "Return ONLY a single JSON object with keys: "
        "content (string), sources (array of objects with url/title/snippet when possible). "
        "Keep content concise and evidence-backed."
    )

    body: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": query},
        ],
        "temperature": 0.2,
        "stream": False,
    }
    body.update(extra_body)

    headers: dict[str, str] = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    for key, value in extra_headers.items():
        headers[str(key)] = str(value)

    req = urllib.request.Request(
        url=url,
        data=_compact_json(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    # 优先使用 requests 库（处理代理和 SSL 重协商更可靠）
    try:
        import requests as _requests
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        body_dict = json.loads(_compact_json(body))
        r = _requests.post(
            url,
            json=body_dict,
            headers=headers,
            timeout=timeout_seconds,
            verify=False,
        )
        if r.status_code != 200:
            raise urllib.error.HTTPError(url, r.status_code, r.text, {}, None)
        raw = r.content.decode("utf-8", errors="replace")
        if raw.strip().startswith("data:"):
            return _parse_sse_response(raw)
        return json.loads(raw)
    except ImportError:
        pass

    with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
        raw = resp.read().decode("utf-8", errors="replace")

        # 检测是否为 SSE 流式响应
        if raw.strip().startswith("data:"):
            return _parse_sse_response(raw)
        else:
            # 标准 JSON 响应
            return json.loads(raw)


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggressive web research via OpenAI-compatible Grok endpoint.")
    parser.add_argument("--query", required=True, help="Search query / research task.")
    parser.add_argument("--config", default="", help="Path to config JSON file.")
    parser.add_argument("--base-url", default="", help="Override base URL.")
    parser.add_argument("--api-key", default="", help="Override API key.")
    parser.add_argument("--model", default="", help="Override model.")
    parser.add_argument("--timeout-seconds", type=float, default=0.0, help="Override timeout (seconds).")
    parser.add_argument(
        "--extra-body-json",
        default="",
        help="Extra JSON object merged into request body.",
    )
    parser.add_argument(
        "--extra-headers-json",
        default="",
        help="Extra JSON object merged into request headers.",
    )
    args = parser.parse_args()

    env_config_path = os.environ.get("GROK_CONFIG_PATH", "").strip()
    explicit_config_path = args.config.strip() or env_config_path

    config_path = ""
    config: dict[str, Any] = {}

    if explicit_config_path:
        config_path = explicit_config_path
        try:
            config = _load_json_file(config_path)
        except Exception as e:
            sys.stderr.write(f"Invalid config ({config_path}): {e}\n")
            return 2
    else:
        fallback_path = ""
        fallback_config: dict[str, Any] = {}
        for candidate in [*_default_skill_config_paths(), _default_user_config_path()]:
            if not os.path.exists(candidate):
                continue
            try:
                candidate_config = _load_json_file(candidate)
            except Exception as e:
                sys.stderr.write(f"Invalid config ({candidate}): {e}\n")
                return 2

            if not fallback_path:
                fallback_path = candidate
                fallback_config = candidate_config

            candidate_key = _normalize_api_key(str(candidate_config.get("api_key") or ""))
            if candidate_key:
                config_path = candidate
                config = candidate_config
                break

        if not config_path and fallback_path:
            config_path = fallback_path
            config = fallback_config

        if not config_path:
            config_path = _default_skill_config_paths()[0]

    base_url = _normalize_base_url_value(
        args.base_url.strip() or os.environ.get("GROK_BASE_URL", "").strip() or str(config.get("base_url") or "").strip()
    )
    api_key = _normalize_api_key(
        args.api_key.strip() or os.environ.get("GROK_API_KEY", "").strip() or str(config.get("api_key") or "").strip()
    )
    model = args.model.strip() or os.environ.get("GROK_MODEL", "").strip() or str(config.get("model") or "").strip() or "grok-2-latest"

    timeout_seconds = args.timeout_seconds
    if not timeout_seconds:
        timeout_seconds = float(os.environ.get("GROK_TIMEOUT_SECONDS", "0") or "0")
    if not timeout_seconds:
        timeout_seconds = float(config.get("timeout_seconds") or 0) or 60.0

    if not base_url:
        sys.stderr.write(
            "Missing base URL: set GROK_BASE_URL, write it to config, or pass --base-url\n"
            f"Config path: {config_path}\n"
        )
        return 2

    if not api_key:
        sys.stderr.write(
            "Missing API key: set GROK_API_KEY, write it to config, or pass --api-key\n"
            f"Config path: {config_path}\n"
        )
        return 2

    try:
        extra_body: dict[str, Any] = {}
        cfg_extra_body = config.get("extra_body")
        if isinstance(cfg_extra_body, dict):
            extra_body.update(cfg_extra_body)
        extra_body.update(_load_json_env("GROK_EXTRA_BODY_JSON"))
        extra_body.update(_parse_json_object(args.extra_body_json, label="--extra-body-json"))

        extra_headers: dict[str, Any] = {}
        cfg_extra_headers = config.get("extra_headers")
        if isinstance(cfg_extra_headers, dict):
            extra_headers.update(cfg_extra_headers)
        extra_headers.update(_load_json_env("GROK_EXTRA_HEADERS_JSON"))
        extra_headers.update(_parse_json_object(args.extra_headers_json, label="--extra-headers-json"))
    except Exception as e:
        sys.stderr.write(f"Invalid JSON: {e}\n")
        return 2

    started = time.time()
    try:
        resp = _request_chat_completions(
            base_url=base_url,
            api_key=api_key,
            model=model,
            query=args.query,
            timeout_seconds=timeout_seconds,
            extra_headers=extra_headers,
            extra_body=extra_body,
        )
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else ""
        out = {
            "ok": False,
            "error": f"HTTP {getattr(e, 'code', None)}",
            "detail": raw or str(e),
            "config_path": config_path,
            "base_url": base_url,
            "model": model,
            "elapsed_ms": int((time.time() - started) * 1000),
        }
        sys.stdout.write(_compact_json(out))
        return 1
    except Exception as e:
        out = {
            "ok": False,
            "error": "request_failed",
            "detail": str(e),
            "config_path": config_path,
            "base_url": base_url,
            "model": model,
            "elapsed_ms": int((time.time() - started) * 1000),
        }
        sys.stdout.write(_compact_json(out))
        return 1

    message = ""
    try:
        choice0 = (resp.get("choices") or [{}])[0]
        msg = choice0.get("message") or {}
        message = msg.get("content") or ""
    except Exception:
        message = ""

    # 尝试解析 JSON
    parsed = _coerce_json_object(message)
    sources: list[dict[str, Any]] = []
    content = ""
    raw = ""

    if parsed is not None:
        # 成功解析 JSON
        content = str(parsed.get("content") or "")
        src = parsed.get("sources")
        if isinstance(src, list):
            for item in src:
                if isinstance(item, dict) and item.get("url"):
                    sources.append(
                        {
                            "url": str(item.get("url")),
                            "title": str(item.get("title") or ""),
                            "snippet": str(item.get("snippet") or ""),
                        }
                    )
        if not sources:
            for url in _extract_urls(content):
                sources.append({"url": url, "title": "", "snippet": ""})
    else:
        # 无法解析为 JSON，将完整消息放入 raw
        raw = message
        # 尝试从 raw 中提取 URL
        for url in _extract_urls(message):
            sources.append({"url": url, "title": "", "snippet": ""})

    out = {
        "ok": True,
        "query": args.query,
        "config_path": config_path,
        "base_url": base_url,
        "model": resp.get("model") or model,
        "content": content,
        "sources": sources,
        "raw": raw,
        "usage": resp.get("usage") or {},
        "elapsed_ms": int((time.time() - started) * 1000),
    }
    # 确保正确输出 UTF-8 编码
    output = _compact_json(out)
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout.buffer.write(output.encode('utf-8'))
    else:
        sys.stdout.write(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
