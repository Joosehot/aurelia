"""Export the knowledge Aurelia brands, once, from what is already on the machine.

    .venv\\Scripts\\python.exe tools\\export_knowledge.py

Reads (never writes) Devil's Clockwork:
  state/base/nodes.sqlite   every proposition Devil read out of Simple English Wikipedia
                            (one per sentence clause, with its article and paragraph) and
                            Primo's word network said as sentences
  corpus/books.tsv, corpus/raw/<id>.txt   the 39 philosophy books (Project Gutenberg)

Writes knowledge/world.tsv (Wikipedia and Primo) and knowledge/books.tsv (every sentence
of every book, with its paragraph). Aurelia brands both into its block the first time it
starts (aurelia's brand/), and never again.
"""
import json
import os
import re
import sqlite3
import sys

DEVIL = r"C:\Users\joose\DevilsClockwork"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "knowledge")


def clean(s):
    s = s or ""
    if "\u00e2\u20ac" in s or "\u00c3" in s:  # UTF-8 read as cp1252 somewhere on the way
        try:
            s = s.encode("cp1252").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    return re.sub(r"\s+", " ", s.replace("\t", " ")).strip()


def world():
    db = sqlite3.connect(os.path.join(DEVIL, "state", "base", "nodes.sqlite"))
    n = 0
    with open(os.path.join(OUT, "world.tsv"), "w", encoding="utf-8", newline="\n") as f:
        f.write("id\tsource\twork\tlocator\tkind\tsubj\ttext\tverb\trel\tneg\tpred\n")
        rows = db.execute("select id, thinker, work, locator, kind, prop, text from nodes "
                          "where thinker in ('Wikipedia', 'Primo') order by id")
        for id_, thinker, work, loc, kind, prop, text in rows:
            try:
                p = json.loads(prop or "{}")
            except ValueError:
                p = {}
            f.write("\t".join([str(id_), thinker, clean(work), clean(loc), kind, clean(p.get("subj", "")), clean(text),
                               clean(p.get("verb", "")), clean(p.get("rel", "")), "1" if p.get("neg") else "0",
                               clean(p.get("pred", ""))]) + "\n")
            n += 1
    return n


def book_body(text):
    s = re.search(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", text)
    e = re.search(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG", text)
    return text[s.end() if s else 0: e.start() if e else len(text)]


SPLIT = re.compile(r"(?<=[.!?])[\"')\]]?\s+(?=[\"'(\[]?[A-Z])")


def books():
    n = 0
    with open(os.path.join(DEVIL, "corpus", "books.tsv"), encoding="utf-8") as f:
        lines = f.read().splitlines()[1:]
    with open(os.path.join(OUT, "books.tsv"), "w", encoding="utf-8", newline="\n") as out:
        out.write("gid\tthinker\twork\tpara\tsentence\n")
        for line in lines:
            gid, thinker, work = line.split("\t")
            path = os.path.join(DEVIL, "corpus", "raw", f"{gid}.txt")
            if not os.path.exists(path):
                continue
            with open(path, encoding="utf-8", errors="replace") as f:
                body = book_body(f.read())
            for p, para in enumerate(re.split(r"\n\s*\n", body)):
                para = clean(para)
                if len(para) < 30 or para.upper() == para:
                    continue
                for s in SPLIT.split(para):
                    s = s.strip()
                    words = len(s.split())
                    if 5 <= words <= 70 and re.search(r"[a-z]", s):
                        out.write("\t".join([gid, thinker, work, f"par. {p}", s]) + "\n")
                        n += 1
    return n


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    print("world:", world(), "propositions")
    print("books:", books(), "sentences")
    sys.exit(0)
