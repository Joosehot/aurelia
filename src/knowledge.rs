//! Knowledge and memory, branded with a lightning strike.
//!
//! A fact is never stored alone: it carries its strike, the link to where it
//! came from (a WordNet synset, a chain of synsets, a turn of the
//! conversation) and to the moment it was branded (the turn and the word that
//! brought it). Before a fact may be said, `verify` derives it again from its
//! strike; a fact that does not come out the same is not said.

use crate::wordnet::{show, Pos, WordNet, LEX_LOCATION};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Rel {
    Def,
    IsA,
    InstanceOf,
    PartOf,
    HasPart,
    MemberOf,
    Example,
    Synonym,
    Antonym,
    /// derived: X is a Y through a chain of hypernyms
    Chain,
    /// derived: the R of Y is X
    RelOf,
    /// something the user told about themselves
    Memory,
    /// a whole line the user said
    Said,
    /// Aurelia about itself
    SelfNote,
    /// a proposition of Simple English Wikipedia
    World,
    /// a sentence of Primo's word network
    Primo,
    /// a sentence of a philosophy book
    Book,
    /// a principle of the moral core
    Ethic,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Source {
    WordNet { pos: Pos, offset: u32 },
    Chain { path: Vec<(Pos, u32)> },
    RelOf { of: (Pos, u32), found: (Pos, u32), class: String },
    Turn { said: String },
    SelfNote { line: usize },
    /// row of the knowledge store (knowledge/world.tsv, then knowledge/books.tsv)
    Store { row: u32 },
    Ethic { line: usize },
}

/// The memory strike: where the fact came from, and the turn and word that brought it.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Strike {
    pub source: Source,
    pub turn: u32,
    pub cue: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Fact {
    pub rel: Rel,
    /// the lookup key the fact is about (lowercase lemma, "you:name", "self:who")
    pub topic: String,
    pub subject: String,
    pub object: Vec<String>,
    pub proper: bool,
    pub lexfile: u8,
    pub pos: Option<Pos>,
    pub strike: Strike,
    /// the WordNet node this fact is said from: the synset it links to, or its own
    #[serde(default)]
    pub via: Option<(Pos, u32)>,
}

impl Fact {
    pub fn key(&self) -> String {
        format!("{:?}|{}|{}|{}", self.rel, self.topic, self.subject, self.object.join("+"))
    }
}

pub const SELF_NOTES: [(&str, &str); 6] = [
    ("self:who", "I am Aurelia, a block of ring accelerators. A wave rolls through my rings, and I answer from the rings that peak last."),
    ("self:name", "My name is Aurelia."),
    ("self:how", "I have no feelings to report. My rings are turning and the wave is rolling."),
    ("self:can", "I can tell you what the word network holds about a word, check whether one thing is a kind of another, and remember what you tell me about yourself."),
    ("self:know", "Everything I say comes from WordNet or from what you told me. When I hold nothing, I say I don't know."),
    ("self:chat", "Yes. Ask me about anything, tell me about yourself, ask me why, or say tell me more."),
];

pub fn self_facts() -> Vec<Fact> {
    SELF_NOTES
        .iter()
        .enumerate()
        .map(|(i, (topic, text))| Fact {
            rel: Rel::SelfNote,
            topic: topic.to_string(),
            subject: "Aurelia".into(),
            object: vec![text.to_string()],
            proper: true,
            lexfile: 0,
            pos: None,
            strike: Strike { source: Source::SelfNote { line: i }, turn: 0, cue: topic.to_string() },
            via: None,
        })
        .collect()
}

fn split_gloss(gloss: &str) -> (String, Vec<String>) {
    let mut def = String::new();
    let mut examples = Vec::new();
    for (i, part) in gloss.split('"').enumerate() {
        if i % 2 == 0 {
            if i == 0 {
                def = part.trim().trim_end_matches(';').trim().to_string();
            }
        } else if !part.trim().is_empty() {
            examples.push(part.trim().to_string());
        }
    }
    (def, examples)
}

fn first_word(wn: &WordNet, key: (Pos, u32), tgt: u8) -> Option<String> {
    let s = wn.get(key)?;
    let i = if tgt == 0 { 0 } else { (tgt as usize - 1).min(s.words.len() - 1) };
    Some(show(&s.words[i]))
}

/// Every fact one synset holds about `topic`.
pub fn synset_facts(wn: &WordNet, pos: Pos, offset: u32, topic: &str, turn: u32, cue: &str) -> Vec<Fact> {
    let Some(s) = wn.get((pos, offset)) else { return vec![] };
    let subject_raw = s
        .words
        .iter()
        .find(|w| w.to_lowercase() == topic)
        .unwrap_or(&s.words[0])
        .clone();
    let subject = show(&subject_raw);
    let proper = subject.chars().next().map(|c| c.is_uppercase()).unwrap_or(false);
    let strike = Strike { source: Source::WordNet { pos, offset }, turn, cue: cue.to_string() };
    let mk_via = |rel: Rel, object: Vec<String>, via: (Pos, u32)| Fact {
        rel,
        topic: topic.to_string(),
        subject: subject.clone(),
        object,
        proper,
        lexfile: s.lexfile,
        pos: Some(pos),
        strike: strike.clone(),
        via: Some(via),
    };
    let mk = |rel: Rel, object: Vec<String>| mk_via(rel, object, (pos, offset));
    let mut out = Vec::new();
    let (def, examples) = split_gloss(&s.gloss);
    if !def.is_empty() {
        out.push(mk(Rel::Def, vec![def]));
    }
    let syn: Vec<String> = s
        .words
        .iter()
        .filter(|w| w.to_lowercase() != subject_raw.to_lowercase())
        .take(3)
        .map(|w| show(w))
        .collect();
    if !syn.is_empty() {
        out.push(mk(Rel::Synonym, syn));
    }
    let (mut part_of, mut member_of, mut parts) = (0, 0, 0);
    for p in &s.ptrs {
        let Some(w) = first_word(wn, p.target, 0) else { continue };
        let at = p.target;
        match p.sym.as_str() {
            "@" => out.push(mk_via(Rel::IsA, vec![w], at)),
            "@i" => out.push(mk_via(Rel::InstanceOf, vec![w], at)),
            "#p" if part_of < 2 => {
                part_of += 1;
                out.push(mk_via(Rel::PartOf, vec![w], at));
            }
            "#m" if member_of < 2 => {
                member_of += 1;
                out.push(mk_via(Rel::MemberOf, vec![w], at));
            }
            "%p" if parts < 4 => {
                parts += 1;
                out.push(mk_via(Rel::HasPart, vec![w], at));
            }
            "!" => {
                if let Some(a) = first_word(wn, p.target, p.tgt) {
                    out.push(mk_via(Rel::Antonym, vec![a], at));
                }
            }
            _ => {}
        }
    }
    for e in examples.into_iter().take(2) {
        out.push(mk(Rel::Example, vec![e]));
    }
    out
}

fn hypernyms(wn: &WordNet, key: (Pos, u32)) -> Vec<(Pos, u32)> {
    wn.get(key)
        .map(|s| s.ptrs.iter().filter(|p| p.sym == "@" || p.sym == "@i").map(|p| p.target).collect())
        .unwrap_or_default()
}

fn has_lemma(wn: &WordNet, key: (Pos, u32), lemma: &str) -> bool {
    wn.get(key).map(|s| s.words.iter().any(|w| w.to_lowercase() == lemma)).unwrap_or(false)
}

/// The shortest hypernym chain from `from` up to a synset that holds `lemma`.
pub fn chain(wn: &WordNet, from: (Pos, u32), lemma: &str) -> Option<Vec<(Pos, u32)>> {
    let mut seen = std::collections::HashSet::new();
    let mut frontier = vec![vec![from]];
    seen.insert(from);
    for _ in 0..20 {
        let mut next = Vec::new();
        for path in frontier {
            let last = *path.last().unwrap();
            for h in hypernyms(wn, last) {
                if !seen.insert(h) {
                    continue;
                }
                let mut p = path.clone();
                p.push(h);
                if has_lemma(wn, h, lemma) {
                    return Some(p);
                }
                next.push(p);
            }
        }
        if next.is_empty() {
            return None;
        }
        frontier = next;
    }
    None
}

pub fn chain_fact(wn: &WordNet, path: &[(Pos, u32)], x: &str, y: &str, turn: u32, cue: &str) -> Fact {
    let s0 = wn.get(path[0]).unwrap();
    let subject = s0.words.iter().find(|w| w.to_lowercase() == x).map(|w| show(w)).unwrap_or(show(&s0.words[0]));
    let mut object: Vec<String> = path[1..path.len() - 1].iter().map(|k| show(&wn.get(*k).unwrap().words[0])).collect();
    let last = wn.get(*path.last().unwrap()).unwrap();
    object.push(last.words.iter().find(|w| w.to_lowercase() == y).map(|w| show(w)).unwrap_or(show(&last.words[0])));
    Fact {
        rel: Rel::Chain,
        topic: x.to_string(),
        proper: subject.chars().next().map(|c| c.is_uppercase()).unwrap_or(false),
        subject,
        object,
        lexfile: s0.lexfile,
        pos: Some(path[0].0),
        strike: Strike { source: Source::Chain { path: path.to_vec() }, turn, cue: cue.to_string() },
        via: path.last().cloned(),
    }
}

/// The R of Y: a part or member of Y that is a kind of R.
pub fn rel_of(wn: &WordNet, of: (Pos, u32), class: &str) -> Option<(Pos, u32)> {
    let s = wn.get(of)?;
    for p in &s.ptrs {
        if p.sym == "%p" || p.sym == "%m" {
            if has_lemma(wn, p.target, class) || chain(wn, p.target, class).is_some() {
                return Some(p.target);
            }
        }
    }
    None
}

pub fn rel_of_fact(wn: &WordNet, of: (Pos, u32), found: (Pos, u32), class: &str, y: &str, turn: u32, cue: &str) -> Fact {
    let ys = wn.get(of).unwrap();
    let fs = wn.get(found).unwrap();
    Fact {
        rel: Rel::RelOf,
        topic: y.to_string(),
        subject: show(class),
        object: vec![show(&fs.words[0]), show(ys.words.iter().find(|w| w.to_lowercase() == y).unwrap_or(&ys.words[0]))],
        proper: true,
        lexfile: fs.lexfile,
        pos: Some(Pos::N),
        strike: Strike { source: Source::RelOf { of, found, class: class.to_string() }, turn, cue: cue.to_string() },
        via: Some(found),
    }
}

/// Derive the fact again from its strike. Only a fact that comes out the same may be said.
pub fn verify(wn: &WordNet, f: &Fact) -> bool {
    match &f.strike.source {
        Source::WordNet { pos, offset } => {
            let key = f.key();
            synset_facts(wn, *pos, *offset, &f.topic, 0, "").iter().any(|g| g.key() == key)
        }
        Source::Chain { path } => {
            if path.len() < 2 || !has_lemma(wn, path[0], &f.topic) {
                return false;
            }
            let linked = path.windows(2).all(|w| hypernyms(wn, w[0]).contains(&w[1]));
            let y = f.object.last().map(|s| s.to_lowercase().replace(' ', "_")).unwrap_or_default();
            linked && has_lemma(wn, *path.last().unwrap(), &y)
        }
        Source::RelOf { of, found, class } => {
            let linked = wn
                .get(*of)
                .map(|s| s.ptrs.iter().any(|p| (p.sym == "%p" || p.sym == "%m") && p.target == *found))
                .unwrap_or(false);
            linked && (has_lemma(wn, *found, class) || chain(wn, *found, class).is_some())
        }
        Source::Turn { said } => {
            let said = said.to_lowercase();
            match f.rel {
                Rel::Said => f.object.first().map(|o| o.to_lowercase() == said).unwrap_or(false),
                _ => f.object.iter().all(|o| said.contains(&o.to_lowercase())),
            }
        }
        Source::SelfNote { line } => SELF_NOTES.get(*line).map(|(_, t)| f.object.first().map(|o| o == t).unwrap_or(false)).unwrap_or(false),
        Source::Ethic { line } => crate::ethics::PRINCIPLES.get(*line).map(|(t, _)| f.object.first().map(|o| o == t).unwrap_or(false)).unwrap_or(false),
        // checked against the store by the engine, which holds it
        Source::Store { .. } => false,
    }
}

// ---------------------------------------------------------------- phrasing

fn article(w: &str) -> String {
    let first = w.chars().next().unwrap_or('x').to_ascii_lowercase();
    let lw = w.to_lowercase();
    let sounds_y = ["one", "once", "uni", "use", "usu", "eu", "ewe"].iter().any(|p| lw.starts_with(p));
    if "aeiou".contains(first) && !sounds_y {
        format!("an {w}")
    } else {
        format!("a {w}")
    }
}

pub fn plural(wn: &WordNet, w: &str) -> String {
    let (head, last) = match w.rsplit_once(' ') {
        Some((h, l)) => (format!("{h} "), l.to_string()),
        None => (String::new(), w.to_string()),
    };
    if let Some(p) = wn.inflect.get(&(Pos::N, last.to_lowercase())) {
        return format!("{head}{p}");
    }
    let l = last.as_str();
    let p = if l.ends_with('s') || l.ends_with('x') || l.ends_with('z') || l.ends_with("ch") || l.ends_with("sh") {
        format!("{l}es")
    } else if l.ends_with('y') && !l.ends_with("ay") && !l.ends_with("ey") && !l.ends_with("oy") && !l.ends_with("uy") {
        format!("{}ies", &l[..l.len() - 1])
    } else {
        format!("{l}s")
    };
    format!("{head}{p}")
}

fn cap(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
        None => String::new(),
    }
}

fn np(f: &Fact) -> String {
    if f.proper {
        f.subject.clone()
    } else {
        article(&f.subject)
    }
}

fn np_of(w: &str) -> String {
    if w.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) {
        w.to_string()
    } else {
        article(w)
    }
}

fn list(items: &[String], with_article: bool) -> String {
    let items: Vec<String> = items.iter().map(|w| if with_article { np_of(w) } else { w.clone() }).collect();
    match items.len() {
        0 => String::new(),
        1 => items[0].clone(),
        n => format!("{} and {}", items[..n - 1].join(", "), items[n - 1]),
    }
}

fn end(s: String) -> String {
    let t = s.trim_end();
    if t.ends_with('.') || t.ends_with('?') || t.ends_with('!') || t.ends_with('"') {
        t.to_string()
    } else {
        format!("{t}.")
    }
}

/// One sentence for a fact. `v` picks between two wordings; the wave picks `v`.
pub fn phrase(wn: &WordNet, f: &Fact, v: u8) -> String {
    let o = |i: usize| f.object.get(i).cloned().unwrap_or_default();
    let pos = f.pos.unwrap_or(Pos::N);
    let s = match (f.rel, pos) {
        (Rel::Def, Pos::N) => {
            let d = o(0);
            let determined = ["a ", "an ", "the ", "any ", "one ", "a(n) "].iter().any(|a| d.starts_with(a));
            match (determined, v) {
                (true, 0) => format!("{} is {}", cap(&np(f)), d),
                (_, 1) => format!("About {}: {}", np(f), d),
                _ => format!("{}: {}", cap(&f.subject), d),
            }
        }
        (Rel::Def, Pos::V) => {
            if v == 0 {
                format!("To {} means to {}", f.subject, o(0))
            } else {
                format!("\"{}\": to {}", cap(&f.subject), o(0))
            }
        }
        (Rel::Def, _) => {
            if v == 0 {
                format!("{} means {}", cap(&f.subject), o(0))
            } else {
                format!("\"{}\": {}", cap(&f.subject), o(0))
            }
        }
        (Rel::IsA, Pos::V) => {
            if v == 0 {
                format!("To {} is a way to {}", f.subject, o(0))
            } else {
                format!("One way to {} is to {}", o(0), f.subject)
            }
        }
        (Rel::IsA, _) => {
            if v == 0 || f.proper {
                format!("{} is a kind of {}", cap(&np(f)), o(0))
            } else {
                format!("{} are a kind of {}", cap(&plural(wn, &f.subject)), o(0))
            }
        }
        (Rel::InstanceOf, _) => {
            if v == 0 {
                format!("{} is {}", f.subject, np_of(&o(0)))
            } else {
                format!("{} is one of the {}", f.subject, plural(wn, &o(0)))
            }
        }
        (Rel::PartOf, _) => {
            let located = f.lexfile == LEX_LOCATION;
            match (located, v) {
                (true, 0) => format!("{} is in {}", cap(&np(f)), o(0)),
                (true, _) => format!("{} is part of {}", cap(&np(f)), o(0)),
                (false, 0) => format!("{} is part of {}", cap(&np(f)), np_of(&o(0))),
                (false, _) => format!("{} has {}", cap(&np_of(&o(0))), np(f)),
            }
        }
        (Rel::HasPart, _) => {
            if f.lexfile == LEX_LOCATION {
                if v == 0 {
                    format!("{} contains {}", cap(&np(f)), list(&f.object, false))
                } else {
                    format!("Places in {} include {}", np(f), list(&f.object, false))
                }
            } else if v == 0 {
                format!("{} has {}", cap(&np(f)), list(&f.object, true))
            } else {
                format!("Parts of {} include {}", np(f), list(&f.object, true))
            }
        }
        (Rel::MemberOf, _) => {
            if v == 0 || f.proper {
                format!("{} is a member of {}", cap(&np(f)), np_of(&o(0)))
            } else {
                format!("{} belong to {}", cap(&plural(wn, &f.subject)), np_of(&o(0)))
            }
        }
        (Rel::Example, _) => {
            if v == 0 {
                format!("For example: \"{}\"", o(0))
            } else {
                format!("As in \"{}\"", o(0))
            }
        }
        (Rel::Synonym, _) => {
            if v == 0 {
                format!("Another word for {} is {}", f.subject, list(&f.object, false).replace(" and ", " or "))
            } else {
                format!("{} is also called {}", cap(&f.subject), list(&f.object, false).replace(" and ", " or "))
            }
        }
        (Rel::Antonym, _) => {
            if v == 0 {
                format!("The opposite of {} is {}", f.subject, o(0))
            } else {
                format!("{} is the opposite of {}", cap(&o(0)), f.subject)
            }
        }
        (Rel::Chain, _) => {
            let y = f.object.last().cloned().unwrap_or_default();
            let mids = &f.object[..f.object.len() - 1];
            if v == 0 || f.proper || mids.is_empty() {
                let mut links = vec![np(f)];
                links.extend(f.object.iter().map(|w| np_of(w)));
                let steps: Vec<String> = links.windows(2).map(|w| format!("{} is {}", w[0], w[1])).collect();
                format!("Yes: {}", steps.join(", "))
            } else {
                format!("Yes, {} are {} (by way of {})", plural(wn, &f.subject), plural(wn, &y), list(mids, false))
            }
        }
        (Rel::RelOf, _) => {
            if v == 0 {
                format!("The {} of {} is {}", f.subject, o(1), o(0))
            } else {
                format!("{} is the {} of {}", o(0), f.subject, o(1))
            }
        }
        (Rel::Memory, _) => memory_phrase(f, v),
        (Rel::Said, _) => {
            if v == 0 {
                format!("You said: \"{}\"", o(0))
            } else {
                format!("Earlier you said \"{}\"", o(0))
            }
        }
        (Rel::SelfNote, _) => o(0),
        (Rel::World, _) => {
            if v == 0 {
                o(0)
            } else {
                format!("From Wikipedia's article on {}: {}", o(1), o(0))
            }
        }
        (Rel::Primo, _) => o(0),
        (Rel::Book, _) => {
            if v == 0 {
                format!("In {}'s {}: \"{}\"", f.subject, o(1), o(0))
            } else {
                format!("{} ({}): \"{}\"", f.subject, o(1), o(0))
            }
        }
        (Rel::Ethic, _) => {
            if v == 0 {
                o(0)
            } else {
                format!("My moral core holds: {}", o(0))
            }
        }
    };
    end(s)
}

fn memory_phrase(f: &Fact, v: u8) -> String {
    let val = f.object.first().cloned().unwrap_or_default();
    let key = f.subject.as_str();
    let (plain, told) = match key {
        "am" => (format!("You are {val}"), format!("You told me you are {val}")),
        "like" | "love" | "have" => (format!("You {key} {val}"), format!("You told me you {key} {val}")),
        "live in" => (format!("You live in {val}"), format!("You told me you live in {val}")),
        k => (format!("Your {k} is {val}"), format!("You told me your {k} is {val}")),
    };
    if v == 0 {
        plain
    } else {
        told
    }
}
