//! The word network: WordNet 3.1, the same files Prim's word network reads
//! (Primogenitus/motor/data/wordnet/dict), read here as they are.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Pos {
    N,
    V,
    A,
    R,
}

pub fn pos_of(c: &str) -> Option<Pos> {
    match c {
        "n" => Some(Pos::N),
        "v" => Some(Pos::V),
        "a" | "s" => Some(Pos::A),
        "r" => Some(Pos::R),
        _ => None,
    }
}

pub struct Ptr {
    pub sym: String,
    pub target: (Pos, u32),
    /// word number in the target synset (0 = the whole synset)
    pub tgt: u8,
}

pub struct Synset {
    pub lexfile: u8,
    /// words as written ("Albert_Einstein", "dog")
    pub words: Vec<String>,
    pub ptrs: Vec<Ptr>,
    pub gloss: String,
}

pub struct WordNet {
    pub synsets: HashMap<(Pos, u32), Synset>,
    /// (pos, lemma) -> senses, most common first
    pub index: HashMap<(Pos, String), Vec<u32>>,
    /// (pos, inflected) -> base
    pub exc: HashMap<(Pos, String), String>,
    /// (pos, base) -> irregular inflection (mouse -> mice)
    pub inflect: HashMap<(Pos, String), String>,
}

pub const LEX_LOCATION: u8 = 15;
pub const LEX_PERSON: u8 = 18;

const FILES: [(&str, Pos); 4] = [("noun", Pos::N), ("verb", Pos::V), ("adj", Pos::A), ("adv", Pos::R)];

fn read_text(path: &Path) -> Result<String> {
    let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

impl WordNet {
    pub fn read(dir: &Path) -> Result<WordNet> {
        let mut synsets = HashMap::new();
        let mut index = HashMap::new();
        let mut exc = HashMap::new();
        let mut inflect = HashMap::new();
        for (name, pos) in FILES {
            let text = read_text(&dir.join(format!("data.{name}")))?;
            for line in text.lines() {
                if line.starts_with("  ") || line.trim().is_empty() {
                    continue;
                }
                let (head, gloss) = line.split_once("| ").unwrap_or((line, ""));
                let f: Vec<&str> = head.split_whitespace().collect();
                let offset: u32 = f[0].parse()?;
                let lexfile: u8 = f[1].parse()?;
                let w_cnt = usize::from_str_radix(f[3], 16)?;
                let mut p = 4;
                let mut words = Vec::with_capacity(w_cnt);
                for _ in 0..w_cnt {
                    let w = f[p];
                    p += 2;
                    words.push(match w.find('(') {
                        Some(k) => w[..k].to_string(),
                        None => w.to_string(),
                    });
                }
                let p_cnt: usize = f[p].parse()?;
                p += 1;
                let mut ptrs = Vec::with_capacity(p_cnt);
                for _ in 0..p_cnt {
                    if let Some(tp) = pos_of(f[p + 2]) {
                        let st = f[p + 3];
                        ptrs.push(Ptr {
                            sym: f[p].to_string(),
                            target: (tp, f[p + 1].parse()?),
                            tgt: u8::from_str_radix(&st[2..4], 16)?,
                        });
                    }
                    p += 4;
                }
                synsets.insert((pos, offset), Synset { lexfile, words, ptrs, gloss: gloss.trim().to_string() });
            }
            let text = read_text(&dir.join(format!("index.{name}")))?;
            for line in text.lines() {
                if line.starts_with("  ") || line.trim().is_empty() {
                    continue;
                }
                let f: Vec<&str> = line.split_whitespace().collect();
                let synset_cnt: usize = f[2].parse()?;
                let p_cnt: usize = f[3].parse()?;
                let at = 4 + p_cnt + 2;
                let offsets = f[at..at + synset_cnt].iter().map(|s| s.parse()).collect::<Result<Vec<u32>, _>>()?;
                index.insert((pos, f[0].to_string()), offsets);
            }
            let path = dir.join(format!("{name}.exc"));
            if path.exists() {
                for line in read_text(&path)?.lines() {
                    let f: Vec<&str> = line.split_whitespace().collect();
                    if f.len() >= 2 {
                        exc.entry((pos, f[0].to_string())).or_insert_with(|| f[1].to_string());
                        inflect.entry((pos, f[1].to_string())).or_insert_with(|| f[0].to_string());
                    }
                }
            }
        }
        Ok(WordNet { synsets, index, exc, inflect })
    }

    /// The base form of a word as `pos`, if the word network has it (WordNet's morphy).
    pub fn base(&self, word: &str, pos: Pos) -> Option<String> {
        let w = word.to_lowercase().replace(' ', "_");
        if self.index.contains_key(&(pos, w.clone())) {
            return Some(w);
        }
        if let Some(b) = self.exc.get(&(pos, w.clone())) {
            if self.index.contains_key(&(pos, b.clone())) {
                return Some(b.clone());
            }
        }
        let rules: &[(&str, &str)] = match pos {
            Pos::N => &[("ses", "s"), ("xes", "x"), ("zes", "z"), ("ches", "ch"), ("shes", "sh"), ("men", "man"), ("ies", "y"), ("s", "")],
            Pos::V => &[("ies", "y"), ("es", "e"), ("es", ""), ("ed", "e"), ("ed", ""), ("ing", "e"), ("ing", ""), ("s", "")],
            Pos::A => &[("er", ""), ("est", ""), ("er", "e"), ("est", "e")],
            Pos::R => &[],
        };
        for (suf, rep) in rules {
            if let Some(stem) = w.strip_suffix(suf) {
                if stem.is_empty() {
                    continue;
                }
                let cand = format!("{stem}{rep}");
                if self.index.contains_key(&(pos, cand.clone())) {
                    return Some(cand);
                }
            }
        }
        None
    }

    pub fn senses(&self, lemma: &str, pos: Pos) -> &[u32] {
        self.index.get(&(pos, lemma.to_string())).map(|v| v.as_slice()).unwrap_or(&[])
    }

    pub fn get(&self, key: (Pos, u32)) -> Option<&Synset> {
        self.synsets.get(&key)
    }
}

/// "Albert_Einstein" -> "Albert Einstein"
pub fn show(word: &str) -> String {
    word.replace('_', " ")
}
