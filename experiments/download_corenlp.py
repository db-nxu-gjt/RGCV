"""下载并解压 Stanford CoreNLP 2018-10-05 到 DAIL-SQL/third_party(仅安装,不启动)。"""
import os
import time
import urllib.request
import zipfile

URL = "https://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip"
DEST_DIR = os.path.join("baselines", "DAIL-SQL", "third_party")
ZIP_PATH = os.path.join(DEST_DIR, "stanford-corenlp-full-2018-10-05.zip")
TARGET = os.path.join(DEST_DIR, "stanford-corenlp-full-2018-10-05")
os.makedirs(DEST_DIR, exist_ok=True)

if os.path.isdir(TARGET):
    print("ALREADY EXTRACTED")
else:
    req = urllib.request.Request(URL, method="HEAD",
                                 headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        total = int(r.headers["Content-Length"])
    print(f"total {total / 1e6:.1f} MB", flush=True)

    while True:
        have = os.path.getsize(ZIP_PATH) if os.path.exists(ZIP_PATH) else 0
        if have >= total:
            break
        headers = {"User-Agent": "Mozilla/5.0",
                   "Range": f"bytes={have}-{total - 1}"}
        try:
            req = urllib.request.Request(URL, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as r, \
                    open(ZIP_PATH, "ab") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    have += len(chunk)
                    if have % (40 << 20) < (1 << 20):
                        print(f"  {have / 1e6:.0f}/{total / 1e6:.0f} MB",
                              flush=True)
        except Exception as e:
            print(f"  retry@{os.path.getsize(ZIP_PATH) / 1e6:.0f}MB "
                  f"{type(e).__name__}", flush=True)
            time.sleep(2)

    print("download complete, extracting...", flush=True)
    with zipfile.ZipFile(ZIP_PATH) as z:
        z.extractall(DEST_DIR)
    os.remove(ZIP_PATH)
    print("EXTRACTED:", TARGET, flush=True)
print("DONE")
