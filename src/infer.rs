//! Aurelia thinks for itself: the branded knowledge is premises, and the reply is
//! what follows from them, with the steps shown.
//!
//! - **Inheritance.** What is said of a kind holds for what is of that kind:
//!   "an animal can fossilize" and "a dog is an animal" (WordNet's chain) give
//!   "a dog can fossilize". A premise denied of the thing itself ("penguins cannot
//!   fly") blocks the conclusion ("birds can fly" does not make a penguin fly).
//! - **Transitivity.** "a dog is a canine", "a canine is a carnivore", ... give
//!   "a dog is an animal"; "Helsinki is in Finland" and "Finland is in Europe"
//!   give "Helsinki is in Europe".
//! - **Comparison.** Two things meet at their nearest common kind: a cat and a
//!   dog are both carnivores, a cat as a feline and a dog as a canine.
//!
//! A conclusion is surprising when it comes from far up the chain and says
//! something the thing's own kinds don't hint at; the surprising ones are the
//! funny ones, and they are still valid.

use crate::store::{Index, Item, Role, Src, Store};
use crate::wordnet::{show, Pos, WordNet};
use std::collections::{HashMap, HashSet, VecDeque};

pub type Key = (Pos, u32);

/// Every kind above a sense, with the path to it (the sense first).
pub fn ancestors(wn: &WordNet, from: Key, max_depth: usize) -> Vec<Vec<Key>> {
    let mut out = Vec::new();
    let mut seen = HashSet::from([from]);
    let mut q = VecDeque::from([vec![from]]);
    while let Some(path) = q.pop_front() {
        if path.len() > max_depth {
            continue;
        }
        let last = *path.last().unwrap();
        if let Some(s) = wn.get(last) {
            for p in &s.ptrs {
                if (p.sym == "@" || p.sym == "@i") && seen.insert(p.target) {
                    let mut np = path.clone();
                    np.push(p.target);
                    out.push(np.clone());
                    q.push_back(np);
                }
            }
        }
    }
    out
}

/// "Helsinki" -> "Finland" -> "Europe": the places a sense is part of, step by step.
pub fn part_chain(wn: &WordNet, from: Key, max: usize) -> Vec<Key> {
    let mut chain = vec![from];
    let mut seen = HashSet::from([from]);
    while chain.len() <= max {
        let last = *chain.last().unwrap();
        let next = wn.get(last).and_then(|s| s.ptrs.iter().find(|p| p.sym == "#p").map(|p| p.target));
        match next {
            Some(n) if seen.insert(n) => chain.push(n),
            _ => break,
        }
    }
    chain
}

/// The nearest kind two senses share, with each one's path up to it.
pub fn meet(wn: &WordNet, a: Key, b: Key) -> Option<(Vec<Key>, Vec<Key>)> {
    let up_a = ancestors(wn, a, 14);
    let up_b = ancestors(wn, b, 14);
    let mut best: Option<(Vec<Key>, Vec<Key>)> = None;
    for pa in &up_a {
        let top = *pa.last().unwrap();
        if let Some(pb) = up_b.iter().find(|p| *p.last().unwrap() == top) {
            let cost = pa.len() + pb.len();
            if best.as_ref().map(|(x, y)| cost < x.len() + y.len()).unwrap_or(true) {
                best = Some((pa.clone(), pb.clone()));
            }
        }
    }
    best
}

#[derive(Clone, Debug)]
pub struct Premise {
    pub item: u32,
    /// the kind it is said of (lemma)
    pub head: String,
    pub verb: String,
    pub pred: String,
    pub plural: bool,
}

const QUANT: &[&str] = &[
    "the", "this", "that", "these", "those", "his", "her", "their", "its", "our", "my", "your", "most", "some", "many", "few",
    "several", "no", "one", "two", "three", "other", "such", "each", "both", "only", "male", "female", "young", "adult", "wild",
];
const PRONOUNS: &[&str] = &["it", "its", "they", "them", "their", "this", "these", "those", "he", "she", "his", "her", "him", "itself", "themselves"];

/// A proposition fit to be a premise: said of a whole kind, in the present, plain.
pub fn premise(wn: &WordNet, it: &Item, item: u32) -> Option<Premise> {
    if it.src == Src::Book || it.neg || it.pred.is_empty() {
        return None;
    }
    // "a carnivore is an animal" restates the chain of kinds; that is not a premise
    let pl = it.pred.to_lowercase();
    if matches!(it.verb.to_lowercase().as_str(), "is" | "are") && (pl.starts_with("a ") || pl.starts_with("an ")) {
        return None;
    }
    let subj: Vec<String> = it.subj.split_whitespace().map(|w| w.to_lowercase()).collect();
    let (article, rest) = match subj.first().map(|s| s.as_str()) {
        Some("a") | Some("an") | Some("all") | Some("every") => (true, &subj[1..]),
        Some(w) if QUANT.contains(&w) => return None,
        _ => (false, &subj[..]),
    };
    if rest.is_empty() || rest.len() > 3 {
        return None;
    }
    let written = rest.join("_");
    let head = wn.base(&written, Pos::N)?;
    let plural = !article && written != head;
    if !article && !plural && subj.len() == 1 {
        // a bare singular ("Love is ...") is a mass word or a name: not a kind's premise
        return None;
    }
    let verb = it.verb.to_lowercase();
    let ok_verb = match verb.as_str() {
        "is" | "has" | "does" => !plural,
        "are" | "have" | "do" => plural,
        "can" => true,
        "was" | "were" | "had" | "did" | "could" | "would" | "should" | "might" | "may" | "must" | "will" => false,
        v => {
            let base = wn.base(v, Pos::V);
            match base {
                Some(b) if plural => b == v,
                Some(b) => b != v && !v.ends_with("ed"),
                None => false,
            }
        }
    };
    if !ok_verb {
        return None;
    }
    let pw: Vec<String> = it.pred.split_whitespace().map(|w| w.to_lowercase()).collect();
    if pw.is_empty() || pw.len() > 10 || pw.iter().any(|w| PRONOUNS.contains(&w.as_str()) || w.chars().any(|c| c.is_ascii_digit())) {
        return None;
    }
    // Wikipedia speaks of kinds in passing ("animals are exempt from taxes under Section"):
    // only what is plainly said of the kind itself, short, with be, have or can
    if it.src == Src::Wikipedia {
        let plain = matches!(verb.as_str(), "is" | "are" | "has" | "have" | "can");
        let odd = pw.iter().any(|w| matches!(w.as_str(), "being" | "used" | "under" | "only" | "called" | "named" | "known" | "also" | "often" | "usually" | "sometimes" | "mostly" | "mainly" | "very"));
        // said of the kind as a whole ("instruments can be divided by type"), not of each one
        let whole = pw.iter().any(|w| matches!(w.as_str(), "divided" | "classified" | "grouped" | "sorted" | "categorized" | "categorised" | "split" | "organized" | "organised" | "many" | "different" | "various" | "kinds" | "types" | "species" | "common" | "rare" | "extinct" | "numerous" | "widespread" | "everywhere"));
        let odd = odd || whole;
        if !plain || odd || pw.len() > 7 {
            return None;
        }
    }
    // "birds are the living descendants of dinosaurs" does not say a penguin "is the descendants"
    if plural && (verb == "are") {
        let last = pw.last().unwrap();
        let first = pw.first().unwrap();
        if first == "the" || (last.ends_with('s') && wn.base(last, Pos::N).map(|b| &b != last).unwrap_or(false)) {
            return None;
        }
    }
    // the sentence must say it plainly: subject, verb, the rest
    let lt = it.text.to_lowercase();
    if !lt.starts_with(&it.subj.to_lowercase()) {
        return None;
    }
    Some(Premise { item, head, verb, pred: it.pred.clone(), plural })
}

/// Kinds too wide to say anything of a thing by.
pub const TOO_WIDE: &[&str] = &[
    "entity", "physical_entity", "abstraction", "abstract_entity", "object", "physical_object", "whole", "unit", "living_thing",
    "organism", "being", "causal_agent", "cause", "matter", "psychological_feature", "attribute", "relation", "group", "grouping",
    "measure", "thing", "artifact", "artefact", "event", "act", "human_action", "state", "location", "region", "person",
];

/// The premises said of a kind.
pub fn premises_of(wn: &WordNet, store: &Store, index: &Index, lemma: &str, max: usize) -> Vec<Premise> {
    let Some(entries) = index.get(lemma) else { return vec![] };
    let mut out = Vec::new();
    for &(i, role) in entries {
        if !matches!(role, Role::Article | Role::Subject) {
            continue;
        }
        let it = &store.items[i as usize];
        // Primo's sentences carry no sense: "a canine is a tooth" is another canine than the dog's.
        // From Primo only words of one sense; and never "is a ..." (the chain of kinds says that)
        if it.src == Src::Primo && (wn.senses(lemma, Pos::N).len() != 1 || matches!(it.verb.as_str(), "is" | "are")) {
            continue;
        }
        if let Some(p) = premise(wn, it, i) {
            if p.head == lemma && (it.src == Src::Primo || role == Role::Article) {
                out.push(p);
                if out.len() >= max {
                    break;
                }
            }
        }
    }
    out
}

/// The verb as said of one thing: "have" -> "has", "eat" -> "eats", "can" -> "can".
pub fn singular(verb: &str, plural: bool) -> String {
    if !plural {
        return verb.to_string();
    }
    match verb {
        "are" => "is".into(),
        "have" => "has".into(),
        "do" => "does".into(),
        "can" => "can".into(),
        v if v.ends_with('s') || v.ends_with('x') || v.ends_with("ch") || v.ends_with("sh") || v.ends_with('o') => format!("{v}es"),
        v if v.ends_with('y') && !v.ends_with("ay") && !v.ends_with("ey") && !v.ends_with("oy") => format!("{}ies", &v[..v.len() - 1]),
        v => format!("{v}s"),
    }
}

/// Is the conclusion denied of the thing itself, or already said of it? (denied, known)
pub fn checked(wn: &WordNet, store: &Store, index: &Index, thing: &str, p: &Premise) -> (bool, bool) {
    let Some(entries) = index.get(thing) else { return (false, false) };
    let first = |s: &str| s.split_whitespace().next().unwrap_or("").to_lowercase();
    let pv = wn.base(&p.verb, Pos::V).unwrap_or(p.verb.clone());
    let pp = first(&p.pred);
    let (mut denied, mut known) = (false, false);
    for &(i, role) in entries {
        if !matches!(role, Role::Article | Role::Subject) {
            continue;
        }
        let it = &store.items[i as usize];
        let v = wn.base(&it.verb.to_lowercase(), Pos::V).unwrap_or(it.verb.to_lowercase());
        if v == pv && first(&it.pred) == pp {
            if it.neg {
                denied = true;
            } else {
                known = true;
            }
        }
    }
    (denied, known)
}

pub fn word(wn: &WordNet, k: Key) -> String {
    wn.get(k).map(|s| show(&s.words[0])).unwrap_or_default()
}

pub fn np(w: &str) -> String {
    if w.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) {
        return w.to_string();
    }
    let lw = w.to_lowercase();
    let y = ["one", "uni", "use", "usu", "eu"].iter().any(|p| lw.starts_with(p));
    if "aeiou".contains(lw.chars().next().unwrap_or('x')) && !y {
        format!("an {w}")
    } else {
        format!("a {w}")
    }
}

/// How surprising a conclusion is: from how far up it comes, and whether its words
/// are strangers to the thing's own kinds.
pub fn surprise(wn: &WordNet, path: &[Key], pred: &str) -> f32 {
    let depth = (path.len() - 1) as f32;
    let kinds: HashSet<String> = path.iter().flat_map(|k| wn.get(*k).map(|s| s.words.iter().map(|w| w.to_lowercase()).collect::<Vec<_>>()).unwrap_or_default()).collect();
    let near = pred.split_whitespace().any(|w| kinds.contains(&w.to_lowercase()) || wn.base(w, Pos::N).map(|b| kinds.contains(&b)).unwrap_or(false));
    (depth.min(8.0) / 8.0) * 0.6 + if near { 0.0 } else { 0.4 }
}

/// How familiar a kind is: how much the world knowledge says of it.
pub fn familiar(index: &Index, lemma: &str) -> usize {
    index.get(lemma).map(|e| e.iter().filter(|(_, r)| matches!(r, Role::Article | Role::Subject)).count()).unwrap_or(0)
}

/// "a dog is a canine, a canine is a carnivore, ..." shortened to the kinds on the way.
pub fn by_way(wn: &WordNet, path: &[Key]) -> String {
    let mids: Vec<String> = path[1..path.len().saturating_sub(1)].iter().map(|k| word(wn, *k)).collect();
    match mids.len() {
        0 => String::new(),
        1 => format!(" (by way of {})", mids[0]),
        2 => format!(" (by way of {} and {})", mids[0], mids[1]),
        3 => format!(" (by way of {}, {} and one more kind)", mids[0], mids[1]),
        n => format!(" (by way of {}, {} and {} more kinds)", mids[0], mids[1], n - 2),
    }
}

pub fn cap(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
        None => String::new(),
    }
}

/// The lemma most often used of a set of kinds (for "a dog is an animal" rather than "a dog is a chordate").
pub fn best_kind(wn: &WordNet, index: &Index, paths: &[Vec<Key>]) -> Option<Vec<Key>> {
    let mut scored: HashMap<usize, usize> = HashMap::new();
    for (i, p) in paths.iter().enumerate() {
        if p.len() < 3 || p.len() > 7 {
            continue;
        }
        let w = wn.get(*p.last().unwrap()).map(|s| s.words[0].to_lowercase()).unwrap_or_default();
        if TOO_WIDE.contains(&w.as_str()) {
            continue;
        }
        // familiar, and not far: "a violin is a musical instrument", not "an object"
        scored.insert(i, familiar(index, &w) * 8 / (p.len() + 4));
    }
    scored.into_iter().filter(|(_, f)| *f > 0).max_by_key(|(i, f)| (*f, usize::MAX - *i)).map(|(i, _)| paths[i].clone())
}
