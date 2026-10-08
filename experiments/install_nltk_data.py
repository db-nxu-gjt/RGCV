"""从 GitHub 直链安装 NLTK punkt / stopwords(默认 raw 源在当前网络超时)。"""
import os
import sys
import zipfile
import urllib.request

import nltk

# nltk_data 默认可写目录(Windows:%APPDATA%\nltk_data)
target_dir = os.path.join(os.environ["APPDATA"], "nltk_data")
os.makedirs(target_dir, exist_ok=True)

BASE = "https://github.com/nltk/nltk_data/raw/gh-pages/packages"
PACKAGES = {
    "punkt": f"{BASE}/tokenizers/punkt.zip",
    "stopwords": f"{BASE}/corpora/stopwords.zip",
}

for name, url in PACKAGES.items():
    zip_path = os.path.join(target_dir, f"{name}.zip")
    print(f"downloading {name} <- {url}", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(zip_path, "wb") as f:
        f.write(r.read())
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(target_dir)
    os.remove(zip_path)
    print(f"  installed -> {target_dir}", flush=True)

# 验证
from nltk.tokenize import word_tokenize, sent_tokenize  # noqa
from nltk.corpus import stopwords  # noqa
print("punkt ok:", sent_tokenize("This is a test. It works.")[-1])
print("stopwords ok:", len(stopwords.words("english")) > 0)
