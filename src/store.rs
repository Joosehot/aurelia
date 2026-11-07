//! World and philosophical knowledge, as tools/export_knowledge.py took it from
//! Devil's Clockwork: every proposition of Simple English Wikipedia and of Primo's
//! word network (knowledge/world.tsv), and every sentence of the 39 philosophy
//! books (knowledge/books.tsv). Each item is branded onto a particle of its own.
//!
//! The index says which items are about a word: the article about it, items whose
//! subject is it, book sentences that say what it is, book sentences that mention it.

use crate::wordnet::{Pos, WordNet};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Src {
    Wikipedia,
    Primo,
    Book,
    /// a human reply from Topical-Chat (CDLA-Sharing 1.0), with the turn it answered as `subj`
    Dialog,
}

pub struct Item {
    pub src: Src,
    /// the thinker for a book; "Wikipedia" or "Primo" otherwise
    pub who: String,
    /// the article or the book
    pub work: String,
    pub locator: String,
    /// Devil's kind: claim, argument, definition (empty for books)
    pub kind: String,
    pub subj: String,
    pub text: String,
    /// the proposition's verb, whether it is denied, and what it says of the subject
    pub verb: String,
    pub neg: bool,
    pub pred: String,
}

pub struct Store {
    pub items: Vec<Item>,
}

/// What an item is to a word.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Role {
    /// in the article about the word, and about the word itself
    Article,
    /// in the article about the word
    InArticle,
    /// its subject is the word
    Subject,
    /// a book sentence saying what the word is ("Justice is ...")
    BookSays,
    /// a book sentence that uses the word
    BookMentions,
    /// a dialog reply whose prompt uses the word
    DialogPrompt,
}

pub type Index = HashMap<String, Vec<(u32, Role)>>;

impl Store {
    pub fn load(dir: &Path) -> Result<Store> {
        let mut items = Vec::new();
        let world = std::fs::read_to_string(dir.join("world.tsv"))
            .with_context(|| format!("no {}; run tools/export_knowledge.py once", dir.join("world.tsv").display()))?;
        for line in world.lines().skip(1) {
            let f: Vec<&str> = line.split('\t').collect();
            if f.len() < 7 {
                continue;
            }
            items.push(Item {
                src: if f[1] == "Primo" { Src::Primo } else { Src::Wikipedia },
                who: f[1].to_string(),
                work: f[2].to_string(),
                locator: f[3].to_string(),
                kind: f[4].to_string(),
                subj: f[5].to_string(),
                text: f[6].to_string(),
                verb: f.get(7).unwrap_or(&"").to_string(),
                neg: f.get(9).map(|x| *x == "1").unwrap_or(false),
                pred: f.get(10).unwrap_or(&"").to_string(),
            });
        }
        let books = std::fs::read_to_string(dir.join("books.tsv")).context("reading knowledge/books.tsv")?;
        for line in books.lines().skip(1) {
            let f: Vec<&str> = line.split('\t').collect();
            if f.len() < 5 {
                continue;
            }
            items.push(Item {
                src: Src::Book,
                who: f[1].to_string(),
                work: f[2].to_string(),
                locator: f[3].to_string(),
                kind: String::new(),
                subj: String::new(),
                text: f[4].to_string(),
                verb: String::new(),
                neg: false,
                pred: String::new(),
            });
        }
        Ok(Store { items })
    }
}

/// Words a social reply may carry without making a claim about the world.
pub const SOCIAL_NOUNS: &[&str] = &[
    "thing", "things", "way", "idea", "point", "time", "chat", "talk", "day", "night", "fun", "kiss", "hug", "lot", "bit", "sense",
    "question", "answer", "pleasure", "luck", "care", "joke", "deal", "guess", "chance", "morning", "evening", "kind", "sort", "stuff",
    "favorite", "problem", "worries", "worry", "story", "thought", "thoughts", "friend", "wow", "lol", "haha", "yes", "yeah", "hello",
    "hi", "hey", "thanks", "thank", "bye", "goodbye", "conversation", "word", "one", "today", "too", "fact", "facts", "news",
];
/// First-person claims Aurelia cannot make: she has no life, no likes, no past.
const CLAIMS: &[&str] = &[
    "my", "mine", "we", "our", "i have", "i had", "i was", "i went", "i live", "i've", "i work", "i love", "i like", "i hate",
    "i feel", "i watched", "i read", "i saw", "i heard", "i remember", "i grew", "i used", "i play", "i own", "i'm", "i am",
    "i did", "i do like", "i enjoy", "i miss", "i want", "i wish", "i would", "i'd", "i got", "i get", "i need", "i listen",
    "i watch", "i think i", "me too", "i also", "i never", "i always", "i usually", "i sometimes",
];

/// The human replies from Topical-Chat that Aurelia may say: social moves only.
pub fn load_dialog(dir: &Path, wn: &WordNet) -> Result<Vec<Item>> {
    let mut out = Vec::new();
    for name in ["train", "valid_freq", "valid_rare", "test_freq", "test_rare"] {
        let path = dir.join(format!("{name}.json"));
        if !path.exists() {
            continue;
        }
        let v: serde_json::Value = serde_json::from_slice(&std::fs::read(&path)?)?;
        let Some(convs) = v.as_object() else { continue };
        let mut ids: Vec<&String> = convs.keys().collect();
        ids.sort();
        for id in ids {
            let Some(turns) = convs[id].get("content").and_then(|c| c.as_array()) else { continue };
            for k in 1..turns.len() {
                let prompt = turns[k - 1].get("message").and_then(|m| m.as_str()).unwrap_or("").trim().to_string();
                let reply = turns[k].get("message").and_then(|m| m.as_str()).unwrap_or("").trim().to_string();
                if social(wn, &reply) && !prompt.is_empty() && prompt.split_whitespace().count() <= 25 {
                    out.push(Item {
                        src: Src::Dialog,
                        who: "Topical-Chat".into(),
                        work: format!("Topical-Chat {name}"),
                        locator: format!("{id} turn {k}"),
                        kind: String::new(),
                        subj: prompt,
                        text: reply,
                        verb: String::new(),
                        neg: false,
                        pred: String::new(),
                    });
                }
            }
        }
    }
    Ok(out)
}

/// A reply that does something in the talk (reacts, thanks, asks back, opens, closes)
/// without claiming anything about the world or about a life Aurelia doesn't have.
fn social(wn: &WordNet, reply: &str) -> bool {
    let n = reply.split_whitespace().count();
    if !(2..=14).contains(&n) || reply.chars().any(|c| c.is_ascii_digit()) || reply.contains("http") {
        return false;
    }
    // no names: a capital only at the start of a sentence, or "I"
    let mut start = true;
    for w in reply.split_whitespace() {
        let c = w.chars().next().unwrap_or('a');
        if c.is_uppercase() && !start && w.trim_matches(|x: char| !x.is_alphabetic()) != "I" {
            return false;
        }
        start = w.ends_with('.') || w.ends_with('!') || w.ends_with('?');
    }
    let low = format!(" {} ", reply.to_lowercase().replace(['.', ',', '!', '?', ';', ':'], " "));
    if CLAIMS.iter().any(|c| low.contains(&format!(" {c} "))) {
        return false;
    }
    // no thing named that the talk would have to know about
    let ws = words(reply);
    !ws.iter().any(|w| {
        w.len() > 2 && !SOCIAL_NOUNS.contains(&w.as_str()) && !CUT.contains(&w.as_str()) && !ARTICLES.contains(&w.as_str())
            && wn.base(w, Pos::N).is_some() && wn.base(w, Pos::V).is_none() && wn.base(w, Pos::A).is_none()
    })
}

const CUT: &[&str] = &["of", "in", "on", "at", "for", "with", "from", "by", "to", "that", "which", "who", "whose", "and", "or", "as"];
const ARTICLES: &[&str] = &["the", "a", "an", "this", "that", "these", "those", "his", "her", "its", "their", "our", "my", "your", "some", "many", "most", "all", "every", "each", "one"];

pub fn words(s: &str) -> Vec<String> {
    s.split(|c: char| !(c.is_alphanumeric() || c == '-' || c == '\''))
        .filter(|w| !w.is_empty())
        .map(|w| w.to_lowercase())
        .collect()
}

/// The word a noun phrase is about: the whole phrase if the network has it, else
/// its last noun before the first preposition ("the capital of France" -> capital).
pub fn head(wn: &WordNet, phrase: &str) -> Option<String> {
    // a name ("Benjamin Justice Soghie") is about itself, never about a word inside it
    let raw: Vec<&str> = phrase.split_whitespace().collect();
    if raw.len() > 1 && raw.iter().skip(1).filter(|x| x.chars().next().map(|c| c.is_uppercase()).unwrap_or(false)).count() > 0 {
        let name = words(phrase).join("_");
        return Some(wn.base(&name, Pos::N).unwrap_or(name));
    }
    let w = words(phrase);
    let w: Vec<String> = w.into_iter().skip_while(|x| ARTICLES.contains(&x.as_str())).collect();
    let cut = w.iter().position(|x| CUT.contains(&x.as_str())).unwrap_or(w.len());
    let core = &w[..cut];
    if core.is_empty() {
        return None;
    }
    if core.len() > 1 {
        if let Some(b) = wn.base(&core.join("_"), Pos::N) {
            return Some(b);
        }
    }
    core.iter().rev().find_map(|x| wn.base(x, Pos::N))
}

/// The word an article is about. "Paris (mythology)" is another Paris than the word
/// "Paris", so a title with a parenthesis is only itself.
fn title_lemma(wn: &WordNet, title: &str) -> Option<String> {
    let t = title.trim().to_lowercase().replace(' ', "_");
    if title.contains('(') {
        return Some(t);
    }
    wn.base(&t, Pos::N).or(Some(t))
}

pub fn build_index(store: &Store, wn: &WordNet) -> Index {
    let mut index: Index = HashMap::new();
    let mut titles: HashMap<String, Option<String>> = HashMap::new();
    for (i, it) in store.items.iter().enumerate() {
        let id = i as u32;
        match it.src {
            Src::Wikipedia | Src::Primo => {
                let mut h = head(wn, &it.subj);
                // in the article "Jordan Love", "Love was selected ..." is about Jordan Love
                let tw: Vec<&str> = it.work.split_whitespace().collect();
                let sw: Vec<&str> = it.subj.split_whitespace().collect();
                if it.src == Src::Wikipedia && tw.len() > 1 && sw.len() == 1 && tw.iter().any(|x| x.eq_ignore_ascii_case(sw[0])) {
                    h = title_lemma(wn, &it.work);
                }
                let title = if it.src == Src::Wikipedia {
                    titles.entry(it.work.clone()).or_insert_with(|| title_lemma(wn, &it.work)).clone()
                } else {
                    None
                };
                match (&h, &title) {
                    (Some(h), Some(t)) if h == t => index.entry(t.clone()).or_default().push((id, Role::Article)),
                    _ => {
                        if let Some(t) = &title {
                            index.entry(t.clone()).or_default().push((id, Role::InArticle));
                        }
                        if let Some(h) = &h {
                            index.entry(h.clone()).or_default().push((id, Role::Subject));
                        }
                    }
                }
            }
            Src::Dialog => {
                let mut seen = std::collections::HashSet::new();
                for x in words(&it.subj) {
                    let key = [Pos::N, Pos::V, Pos::A].iter().find_map(|&p| wn.base(&x, p)).unwrap_or(x.clone());
                    if key.len() >= 2 && seen.insert(key.clone()) {
                        index.entry(format!("dlg:{key}")).or_default().push((id, Role::DialogPrompt));
                    }
                }
            }
            Src::Book => {
                let w = words(&it.text);
                // "Justice is ...", "The soul is ...", "Men are ..."
                let start = w.iter().position(|x| !ARTICLES.contains(&x.as_str())).unwrap_or(0);
                let mut said = None;
                for len in [2usize, 1] {
                    if start + len < w.len() && matches!(w[start + len].as_str(), "is" | "are" | "was" | "were") {
                        if let Some(b) = wn.base(&w[start..start + len].join("_"), Pos::N) {
                            said = Some(b);
                            break;
                        }
                    }
                }
                if let Some(s) = &said {
                    index.entry(s.clone()).or_default().push((id, Role::BookSays));
                }
                let mut seen = std::collections::HashSet::new();
                for x in &w {
                    if x.len() < 3 || ARTICLES.contains(&x.as_str()) || CUT.contains(&x.as_str()) {
                        continue;
                    }
                    if let Some(b) = wn.base(x, Pos::N) {
                        if Some(&b) != said.as_ref() && seen.insert(b.clone()) {
                            index.entry(b).or_default().push((id, Role::BookMentions));
                        }
                    }
                }
            }
        }
    }
    index
}
