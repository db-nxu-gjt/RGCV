"""断点续传下载 sqlcoder-7b-2 GGUF (绕过 Windows 系统代理直连 hf-mirror)。"""
import sys
import time
from pathlib import Path

import requests

URL = "https://modelscope.cn/models/MaziyarPanahi/sqlcoder-7b-2-GGUF/resolve/master/sqlcoder-7b-2.Q4_K_M.gguf"
OUT = Path("models") / "sqlcoder-7b-2.Q4_K_M.gguf"


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    # 断点续传
    pos = OUT.stat().st_size if OUT.exists() else 0
    headers = {"Range": f"bytes={pos}-"} if pos else {}

    s = requests.Session()
    s.trust_env = False  # 绕过 Windows 系统代理

    for attempt in range(20):
        try:
            with s.get(URL, headers=headers, stream=True, timeout=60, allow_redirects=True) as r:
                if r.status_code == 416:  # 已下完
                    print("416: 已完成")
                    break
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0)) + pos
                mode = "ab" if pos else "wb"
                with open(OUT, mode) as f:
                    for chunk in r.iter_content(chunk_size=1 << 22):  # 4MB
                        f.write(chunk)
                        pos += len(chunk)
                        pct = pos / total * 100 if total else 0
                        print(f"{pos/1e9:.2f}/{total/1e9:.2f} GB ({pct:.1f}%)", flush=True)
                if total and pos >= total:
                    print("下载完成")
                    break
                headers = {"Range": f"bytes={pos}-"}
        except Exception as e:
            print(f"[{attempt}] 异常: {e}, 5s 后续传 from {pos/1e9:.2f}GB", flush=True)
            time.sleep(5)

    final = OUT.stat().st_size / 1e9
    print(f"最终大小: {final:.2f} GB")
    sys.exit(0 if final > 4.0 else 1)


if __name__ == "__main__":
    main()
