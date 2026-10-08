"""手动下载 Stanford GloVe glove.6B.zip 到 DAIL-SQL vector_cache。

torchtext 默认下载到 cache/ 目录,手动放好 zip 后它会跳过下载。
"""
import urllib.request, time, os, sys

url = "http://nlp.stanford.edu/data/glove.6B.zip"
dest = os.path.join("baselines", "DAIL-SQL", "vector_cache")
os.makedirs(dest, exist_ok=True)
out = os.path.join(dest, "glove.6B.zip")

if os.path.exists(out) and os.path.getsize(out) > 100000000:
    print(f"已存在:{out} ({os.path.getsize(out)/1024/1024:.0f}MB)")
    sys.exit(0)

print(f"下载 {url} → {out}")

def report(block, block_size, total_size):
    pct = block * block_size / total_size * 100 if total_size > 0 else 0
    mb = block * block_size / 1024 / 1024
    print(f"\r进度: {pct:.1f}% ({mb:.0f}MB)", end="", flush=True)

for attempt in range(3):
    try:
        urllib.request.urlretrieve(url, out, reporthook=report)
        print(f"\n完成:{os.path.getsize(out)/1024/1024:.0f}MB")
        break
    except Exception as e:
        print(f"\n尝试 {attempt+1} 失败:{e}")
        if os.path.exists(out):
            os.remove(out)
        time.sleep(5)
else:
    print("全部失败,试镜像...")
    for mirror in [
        "https://mirror.ghproxy.com/https://nlp.stanford.edu/data/glove.6B.zip",
        "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip",
    ]:
        print(f"尝试 {mirror}")
        try:
            urllib.request.urlretrieve(mirror, out, reporthook=report)
            print(f"\n完成:{os.path.getsize(out)/1024/1024:.0f}MB")
            break
        except Exception as e2:
            print(f"  镜像也失败:{e2}")
