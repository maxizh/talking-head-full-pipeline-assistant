"""Local corpus: fetch article text to markdown, then retrieve over it.

Default search is character n-gram TF-IDF, which works for mixed Chinese and
English. `retrieval_experiment` also scores keyword overlap, local LSA, and
an LLM query-translation pass. LSA here is TruncatedSVD on this corpus only,
not a trained embedding model.
"""

import html
import json
import os
import re
import time
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CORPUS_DIR = Path(os.environ.get(
    "CORPUS_DIR", Path(__file__).resolve().parents[2] / "corpus"))
MIN_ARTICLE_CHARS = 500
CHUNK_CHARS = 600
UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36 TalkingHeadAssistant/1.0"
)

_BLOCK_END = re.compile(
    r"</(p|div|section|article|li|tr|h[1-6]|blockquote|pre)>|<br\s*/?>", re.I)
_index = {}   # 语料指纹 -> (vectorizer, matrix, chunks)


# 这些章节标题本身不是正文。注意不能从这里截断全文：
# 维基页面顶部的目录里就有 References / See also，截断会把整篇文章砍掉。
_TAIL_SECTION = re.compile(
    r"^(references|external links|see also|further reading|notes|bibliography|"
    r"sources|citations|参考文献|外部链接|参见|注释|延伸阅读)\s*$", re.I)
# 引用条目的典型特征
_CITATION = re.compile(
    r"(archived from the original|retrieved\s+\d|doi:|isbn\b|issn\b|arxiv:|"
    r"^[\^↑]|^\d+(\s+\d+)+\s|"
    # 转义过的 wiki 源码标记：剥标签在前、反转义在后，它们会变回文本留下来
    r"\{\{cite|<ref[\s>/])", re.I)
# 页面框架元素（目录开关、站点导航），够长但不是正文
_CHROME = re.compile(
    r"^(toggle\b.*|jump to \w+|from wikipedia, the free encyclopedia|"
    r"this (section|article) needs.*|please help improve.*|"
    r"read edit.*|main page|contents|current events|random article|"
    r"跳转到内容|维基百科，自由的百科全书)$", re.I)
_SENTENCE_END = re.compile(r"[.!?。！？：:]\s*$")
MIN_PROSE_CHARS = 25   # 比这短又不成句的，基本是导航项/列表项


def _denoise(text):
    """剥掉维基这类页面的引用条目和导航列表，只留正文。

    不去噪的话，检索「ffmpeg 支持哪些格式」头几条命中的全是参考文献行。
    """
    kept = []
    for line in text.split("\n"):
        s = line.strip()
        if not s:
            continue
        if _TAIL_SECTION.match(s) or _CHROME.match(s) or _CITATION.search(s):
            continue
        if len(s) < MIN_PROSE_CHARS and not _SENTENCE_END.search(s):
            continue
        kept.append(s)
    return "\n\n".join(kept)


def _clean_html(raw):
    raw = re.sub(r"(?is)<(script|style|noscript|svg|head)[^>]*>.*?</\1>", " ", raw)
    raw = re.sub(r"(?is)<!--.*?-->", " ", raw)
    text = _BLOCK_END.sub("\n", raw)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = html.unescape(text)
    lines = [re.sub(r"[ \t\xa0]+", " ", ln).strip() for ln in text.splitlines()]
    return "\n\n".join(ln for ln in lines if len(ln) > 1)


def _slugify(text, fallback):
    text = unicodedata.normalize("NFKC", text).strip().lower()
    text = re.sub(r"[^\w一-鿿]+", "-", text).strip("-")
    return (text or fallback)[:60]


def fetch_article(url, name=None):
    """抓页面正文存进语料库。正文太短会报错（多半是 JS 渲染页）。"""
    if not url.startswith(("http://", "https://")):
        raise ValueError(f"不是合法的 http(s) 地址: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": UA,
                                               "Accept-Language": "zh-CN,zh,en"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            charset = resp.headers.get_content_charset()
            raw = resp.read()
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"抓取失败 HTTP {e.code}: {url}") from None
    except urllib.error.URLError as e:
        raise RuntimeError(f"抓取失败（网络或域名问题）: {e.reason}") from None

    if not charset:
        m = re.search(rb'charset=["\']?([\w-]+)', raw[:4096], re.I)
        charset = m.group(1).decode() if m else "utf-8"
    page = raw.decode(charset, "replace")

    m = re.search(r"(?is)<title[^>]*>(.*?)</title>", page)
    title = html.unescape(m.group(1)).strip() if m else url
    body = _denoise(_clean_html(page))

    if len(body) < MIN_ARTICLE_CHARS:
        raise RuntimeError(
            f"只抽到 {len(body)} 字正文（少于 {MIN_ARTICLE_CHARS}），"
            "这页多半是 JS 渲染的，urllib 拿不到内容；换一个静态页面，或手工把正文存进 corpus/。")

    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    slug = _slugify(name or title, _slugify(url, "article"))
    path = CORPUS_DIR / f"{slug}.md"
    path.write_text(
        f"<!-- source: {url} | fetched: {time.strftime('%Y-%m-%d %H:%M')} -->\n"
        f"# {title}\n\n{body}\n")
    _index.clear()
    return {"file": path.name, "path": str(path), "title": title,
            "source": url, "chars": len(body)}


def _docs():
    if not CORPUS_DIR.exists():
        return []
    out = []
    for p in sorted(CORPUS_DIR.glob("*.md")):
        text = p.read_text(errors="replace")
        m = re.match(r"<!-- source: (\S+) \| fetched: ([^>]*?) -->", text)
        out.append({"file": p.name, "path": str(p),
                    "source": m.group(1) if m else "（未记录）",
                    "fetched": m.group(2).strip() if m else "",
                    "text": text, "mtime": p.stat().st_mtime})
    return out


def corpus_list():
    """列出语料库里有哪些资料，以及各自的来源 URL。"""
    docs = _docs()
    return {"corpus_dir": str(CORPUS_DIR), "count": len(docs),
            "documents": [{"file": d["file"], "source": d["source"],
                           "fetched": d["fetched"], "chars": len(d["text"])}
                          for d in docs]}


def _chunks(docs):
    out = []
    for d in docs:
        # 对已抓好的旧语料同样去噪，不用重抓
        body = _denoise(re.sub(r"^<!--.*?-->\n", "", d["text"], count=1))
        buf = ""
        for para in [p.strip() for p in body.split("\n\n") if p.strip()]:
            if len(buf) + len(para) > CHUNK_CHARS and buf:
                out.append({"file": d["file"], "source": d["source"], "text": buf})
                buf = para
            else:
                buf = f"{buf}\n{para}" if buf else para
        if buf:
            out.append({"file": d["file"], "source": d["source"], "text": buf})
    return out


def _get_index():
    docs = _docs()
    if not docs:
        raise RuntimeError(f"语料库是空的（{CORPUS_DIR}）；先用 fetch_article 抓几篇资料。")
    fingerprint = tuple((d["file"], d["mtime"]) for d in docs)
    if fingerprint not in _index:
        chunks = _chunks(docs)
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3))
        matrix = vec.fit_transform([c["text"] for c in chunks])
        _index.clear()
        _index[fingerprint] = (vec, matrix, chunks)
    return _index[fingerprint]


def corpus_search(query, top_k=5):
    """在语料库里按相似度检索段落。写稿只能用这里返回的事实。"""
    vec, matrix, chunks = _get_index()
    scores = cosine_similarity(vec.transform([query]), matrix)[0]
    order = scores.argsort()[::-1][:max(1, int(top_k))]
    hits = [{"file": chunks[i]["file"], "source": chunks[i]["source"],
             "score": round(float(scores[i]), 4), "text": chunks[i]["text"]}
            for i in order if scores[i] > 0]
    return {"query": query, "hit_count": len(hits), "hits": hits}


# Script anchors: did a planned cut drop a point that is not still in the keep set?
# Tuned on real transcripts against the written script:
# verbatim delivery 0.97–1.0; a mistaken take that was re-delivered ~0.56;
# intro/outro lines absent from the script 0.36–0.41; filler ~0.14.
POINT_HIT = 0.50       # removed speech touches this script point
POINT_COVERED = 0.75   # kept speech still covers the point
_script = {"text": "", "points": []}


def set_script(text):
    """记下当前口播稿（teleprompter_load 时写入），段落即要点。"""
    points = [re.sub(r"\[来源[:：][^\]]*\]", "", p).strip()
              for p in (text or "").split("\n\n")]
    _script.update(text=text or "", points=[p for p in points if len(p) >= 10])
    return len(_script["points"])


def get_script():
    return {"chars": len(_script["text"]), "points": list(_script["points"])}


def get_script_text():
    return _script["text"]


def lost_script_points(kept_texts, removed_texts):
    """哪些口播稿要点在这次剪辑后丢了。

    判断的是「要点还在不在成片里」，不是「删掉的文字像不像要点」——
    所以删掉讲错的那一次不算丢（重讲的还留着），删掉唯一讲对的那次才算。
    稿子是写稿时定下的外部参照，不靠模型剪辑时自己回忆哪句重要。
    """
    if not _script["points"] or not removed_texts:
        return []
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3))
    points_m = vec.fit_transform(_script["points"])
    removed_sim = cosine_similarity(vec.transform(removed_texts), points_m)
    kept_sim = (cosine_similarity(vec.transform(kept_texts), points_m)
                if kept_texts else None)

    lost = []
    for j, point in enumerate(_script["points"]):
        in_removed = float(removed_sim[:, j].max())
        still_kept = float(kept_sim[:, j].max()) if kept_sim is not None else 0.0
        if in_removed >= POINT_HIT and still_kept < POINT_COVERED:
            lost.append({"point": point[:100],
                         "removed_score": round(in_removed, 3),
                         "kept_score": round(still_kept, 3),
                         "removed_index": int(removed_sim[:, j].argmax())})
    return lost


# Retrieval routes compared by retrieval_experiment. They share one chunking.
# keyword: literal token overlap.
# tfidf: character n-gram TF-IDF.
# lsa: TruncatedSVD over that TF-IDF matrix. Local dimensionality reduction
# only — it sees co-occurrence inside this corpus and does not know synonyms
# from outside it. It is not a substitute for a trained embedding index.
# llm_tfidf: translate/expand the query, then take the max TF-IDF score.
ROUTES = ("keyword", "tfidf", "lsa", "llm_tfidf")
_route_cache = {}
_variants_cache = {}


def query_variants(query, n=3):
    """把检索问题翻成英文并扩出等价术语（结果缓存，同一问题只花一次 token）。"""
    if query in _variants_cache:
        return _variants_cache[query]
    from agent import llm
    reply = llm.call(
        [{"role": "user", "content":
          f"检索问题：{query}\n\n"
          f"把它翻译成英文，并给出最多 {n} 个英文技术文档里可能出现的等价说法或术语。"
          "只输出这些说法，每行一个，不要编号、不要解释。"}],
        system="你在帮一个英文技术语料库做检索词扩展。只输出检索词本身。",
        max_tokens=200)
    text = "".join(b.get("text", "") for b in reply["content"] if b["type"] == "text")
    variants = [ln.strip(" -•\t") for ln in text.splitlines() if ln.strip()][:n + 1]
    _variants_cache[query] = variants or [query]
    return _variants_cache[query]


def _tokens(text):
    text = text.lower()
    latin = re.findall(r"[a-z0-9]{2,}", text)
    cjk = re.findall(r"[一-鿿]", text)
    return latin + ["".join(p) for p in zip(cjk, cjk[1:])]


def _get_route_cache():
    """Shared chunks and vectorizers so routes differ by method, not by data."""
    vec, matrix, chunks = _get_index()
    key = id(chunks)
    if key not in _route_cache:
        from sklearn.decomposition import TruncatedSVD
        n = min(100, max(2, matrix.shape[0] - 1))
        svd = TruncatedSVD(n_components=n, random_state=0)
        reduced = svd.fit_transform(matrix)
        norms = (reduced ** 2).sum(axis=1) ** 0.5
        norms[norms == 0] = 1e-9
        _route_cache.clear()
        _route_cache[key] = {"vec": vec, "matrix": matrix, "chunks": chunks, "svd": svd,
                     "reduced": reduced / norms[:, None],
                     "chunk_tokens": [set(_tokens(c["text"])) for c in chunks]}
    return _route_cache[key]


def _route_scores(route, query, index):
    if route == "tfidf":
        return cosine_similarity(index["vec"].transform([query]), index["matrix"])[0]
    if route == "llm_tfidf":
        # 每个变体各查一遍，按段落取最高分（并集召回）
        sims = cosine_similarity(index["vec"].transform(query_variants(query)), index["matrix"])
        return sims.max(axis=0)
    if route == "lsa":
        q = index["svd"].transform(index["vec"].transform([query]))[0]
        norm = (q ** 2).sum() ** 0.5 or 1e-9
        return index["reduced"] @ (q / norm)
    terms = set(_tokens(query))
    if not terms:
        return [0.0] * len(index["chunks"])
    return [len(terms & toks) / len(terms) for toks in index["chunk_tokens"]]


def search_route(query, route="tfidf", top_k=5):
    """Search with one route: keyword, tfidf, lsa, or llm_tfidf."""
    if route not in ROUTES:
        raise ValueError(f"route 只能是 {ROUTES} 之一，收到 {route!r}")
    index = _get_route_cache()
    scores = list(_route_scores(route, query, index))
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:int(top_k)]
    return {"route": route, "query": query,
            "hits": [{"file": index["chunks"][i]["file"], "score": round(float(scores[i]), 4),
                      "text": index["chunks"][i]["text"][:300]}
                     for i in order if scores[i] > 0]}


def _rank_of(expect, scores, chunks, top_k):
    """期望内容出现在第几位（1 开始），没进 top_k 就是 None。"""
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    for rank, i in enumerate(order, 1):
        if scores[i] > 0 and expect.lower() in chunks[i]["text"].lower():
            return rank
    return None


def retrieval_experiment(cases, top_k=3):
    """Compare every route on exact wording and on a paraphrase.

    cases: [{"exact": wording close to the source, "paraphrase": a rephrase,
             "expect": a short snippet that must appear in a correct hit}]
    A hit is objective: `expect` occurs inside a top-k chunk.
    """
    index = _get_route_cache()
    if not cases:
        raise ValueError("至少给一个问题；每个问题要有 exact / paraphrase / expect 三项")
    rows, summary = [], {r: {"exact_hit": 0, "para_hit": 0} for r in ROUTES}
    for c in cases:
        for f in ("exact", "paraphrase", "expect"):
            if not c.get(f):
                raise ValueError(f"问题里缺少 {f}: {c}")
        row = {"expect": c["expect"][:60], "exact": c["exact"], "paraphrase": c["paraphrase"],
               "ranks": {}}
        for route in ROUTES:
            e = _rank_of(c["expect"], _route_scores(route, c["exact"], index),
                         index["chunks"], top_k)
            p = _rank_of(c["expect"], _route_scores(route, c["paraphrase"], index),
                         index["chunks"], top_k)
            row["ranks"][route] = {"exact_rank": e, "paraphrase_rank": p}
            summary[route]["exact_hit"] += e is not None
            summary[route]["para_hit"] += p is not None
        rows.append(row)

    n = len(cases)
    for route in ROUTES:
        s = summary[route]
        s["exact_hit_rate"] = round(s["exact_hit"] / n, 2)
        s["paraphrase_hit_rate"] = round(s["para_hit"] / n, 2)
        s["drop"] = round((s["exact_hit"] - s["para_hit"]) / n, 2)
    return {"corpus_chunks": len(index["chunks"]), "cases": n, "top_k": top_k,
            "rows": rows, "summary": summary,
            "note": "lsa 是对本语料 TF-IDF 做的 SVD 降维，不是训练过的向量模型："
                    "它只能在本语料内找共现关系，学不到语料外的同义词。"
                    "若 tfidf 在换说法上掉得厉害而 lsa 明显更好，才有理由上真向量库。"}


def _tool(name, fn, description, properties, required):
    return name, {
        "fn": lambda **kw: json.dumps(fn(**kw), ensure_ascii=False),
        "schema": {"name": name, "description": description,
                   "input_schema": {"type": "object", "properties": properties,
                                    "required": required}},
    }


TOOLS = dict([
    _tool("fetch_article", fetch_article,
          "抓取一个网页的正文，存进语料库 corpus/，首行记来源 URL。"
          "正文太短会报错（多半是 JS 渲染页，换静态页面）。",
          {"url": {"type": "string", "description": "文章的 http(s) 地址"},
           "name": {"type": "string", "description": "可选，自定义文件名（不含扩展名）"}},
          ["url"]),
    _tool("corpus_list", corpus_list,
          "列出语料库里已有的资料及来源 URL。", {}, []),
    _tool("search_route", search_route,
          "用指定的一路检索查语料："
          "keyword=字面命中、tfidf=字符级 TF-IDF、lsa=对本语料做 SVD 降维后匹配、"
          "llm_tfidf=先把问题译成英文并扩同义术语再走 TF-IDF。"
          "lsa 不是训练过的向量模型。",
          {"query": {"type": "string"},
           "route": {"type": "string", "enum": list(ROUTES), "description": "默认 tfidf"},
           "top_k": {"type": "integer", "description": "默认 5"}},
          ["query"]),
    _tool("retrieval_experiment", retrieval_experiment,
          "并排比较 keyword / tfidf / lsa / llm_tfidf。每个问题给三项："
          "exact（用语料里原本的措辞问）、paraphrase（同一件事换个说法、换同义词问）、"
          "expect（正确答案段落里必然出现的一小段原文，用来客观判命中）。"
          "返回每路在两类问法下的命中排名、命中率，以及换说法相对精确词掉了多少。"
          "向用户解释结果时要说明：lsa 是本地 SVD，不是向量模型。",
          {"cases": {"type": "array",
                     "description": "问题列表，建议 4~6 个，覆盖精确词和换说法两类",
                     "items": {"type": "object",
                               "properties": {"exact": {"type": "string"},
                                              "paraphrase": {"type": "string"},
                                              "expect": {"type": "string"}},
                               "required": ["exact", "paraphrase", "expect"]}},
           "top_k": {"type": "integer", "description": "看前几条，默认 3"}},
          ["cases"]),
    _tool("corpus_search", corpus_search,
          "在语料库里检索相关段落（字符级 TF-IDF，中英混排都好使）。"
          "写口播稿只能用这里返回的内容作为事实依据。",
          {"query": {"type": "string", "description": "检索词或问题"},
           "top_k": {"type": "integer", "description": "返回条数，默认 5"}},
          ["query"]),
])
