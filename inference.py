#!/usr/bin/env python3
"""
Inference client for NanoSLG server.
Usage:
    python inf.py "Your prompt here"
    python inf.py --max-tokens 200 "Explain quantum computing"
    python inf.py --no-stream "Hello"
    echo "What is AI?" | python inf.py
"""

import argparse
import sys
import json
import requests
from typing import Generator, Optional


def stream_chat(
    prompt: str,
    max_tokens: int = 100,
    server_url: str = "http://localhost:8000",
    system_prompt: Optional[str] = None,
) -> Generator[str, None, None]:
    """Stream tokens from the server."""
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    response = requests.post(
        f"{server_url}/v1/chat/completions",
        json={
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": True,
        },
        stream=True,
    )
    
    for line in response.iter_lines():
        if not line:
            continue
        
        line = line.decode("utf-8")
        
        # Skip non-data lines
        if not line.startswith("data: "):
            continue
        
        data = line[6:]
        
        # Check for end signal
        if data == "[DONE]":
            break
        
        try:
            chunk = json.loads(data)
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            content = delta.get("content", "")
            if content:
                yield content
        except json.JSONDecodeError:
            continue


def chat(
    prompt: str,
    max_tokens: int = 100,
    server_url: str = "http://localhost:8000",
    system_prompt: Optional[str] = None,
) -> str:
    """Get complete response (non-streaming)."""
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    response = requests.post(
        f"{server_url}/v1/chat/completions",
        json={
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": False,
        },
    )
    
    data = response.json()
    return data["choices"][0]["message"]["content"]


def get_benchmark(server_url: str = "http://localhost:8000") -> dict:
    """Get benchmark results from last generation."""
    response = requests.get(f"{server_url}/v1/benchmark/last")
    return response.json()


def main():
    parser = argparse.ArgumentParser(description="NanoSLG Inference Client")
    parser.add_argument("prompt", nargs="?", default=None, help="Prompt text")
    parser.add_argument("--max-tokens", "-n", type=int, default=100, help="Max tokens to generate")
    parser.add_argument("--server", "-s", type=str, default="http://localhost:8000", help="Server URL")
    parser.add_argument("--system", type=str, default=None, help="System prompt")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming")
    parser.add_argument("--benchmark", "-b", action="store_true", help="Show benchmark after generation")
    args = parser.parse_args()
    
    prompt = args.prompt
    if prompt is None:
        if not sys.stdin.isatty():
            prompt = sys.stdin.read().strip()
        else:
            print("Enter your prompt (Ctrl+D to submit):")
            prompt = sys.stdin.read().strip()
    
    if not prompt:
        print("Error: No prompt provided", file=sys.stderr)
        sys.exit(1)
    
    try:
        if args.no_stream:
            # Non-streaming mode
            response = chat(
                prompt=prompt,
                max_tokens=args.max_tokens,
                server_url=args.server,
                system_prompt=args.system,
            )
            print(response)
        else:
            # Streaming mode
            for token in stream_chat(
                prompt=prompt,
                max_tokens=args.max_tokens,
                server_url=args.server,
                system_prompt=args.system,
            ):
                print(token, end="", flush=True)
            print()  # Final newline
        
        if args.benchmark:
            print("\n" + "="*50)
            bench = get_benchmark(args.server)
            if "prompt_tokens" in bench:
                print(f"Benchmark Results:")
                print(f"  TTFT:           {bench.get('ttft_ms', 0):.2f} ms")
                print(f"  Tokens/sec:     {bench.get('tokens_per_second', 0):.2f}")
                print(f"  Total time:     {bench.get('total_time_ms', 0):.2f} ms")
                print(f"  Peak memory:    {bench.get('peak_memory_mb', 0):.2f} MB")
            else:
                print(bench)
    
    except requests.ConnectionError:
        print(f"Error: Cannot connect to server at {args.server}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[Interrupted]")
        sys.exit(0)


if __name__ == "__main__":
    main()