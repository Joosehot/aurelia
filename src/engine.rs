//! A message in, a reply out.
//!
//! 1. The message is read whole into an ask (what it asks, about which words).
//! 2. Its words' knowledge is branded from the word network onto free particles
//!    near each word's home ring, each fact with its strike (source, turn, word).
//!    What the user says about themselves is branded as memory.
//! 3. The particles that answer the ask get charge; a charged ring's limit rises
//!    with its charge. The wave starts at the home ring of the ask's first word.
//! 4. The wave rolls (a train of fronts) until the last front passes the far end.
//!    Low limits peak early; the rings that peak last hold the most charged
//!    knowledge. The reply is read from the last charged rings to peak, in the
//!    order they peaked, every fact checked against its strike first.

use crate::block::{Block, Cfg, Rng};
use crate::knowledge::{self, phrase, verify, Fact, Rel, Source, Strike};
use crate::wordnet::{Pos, WordNet, LEX_PERSON};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

#[derive(Clone, Debug)]
pub struct Topic {
    pub lemma: String,
    pub pos: Pos,
    pub offset: u32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum QWord {
    What,
    Who,
    Where,
    When,
    Why,
    How,
}

#[derive(Clone, Debug)]
pub enum Ask {
    Greet,
    Thanks,
    Bye,
    Myself(&'static str),
    Tell { key: String, value: String },
    Recall { key: Option<String> },
    RecallSaid,
    IsA { x: Topic, y: Topic },
    RelOf { class: String, y: Topic },
    Opposite { x: Topic },
    About { q: QWord, topics: Vec<Topic>, unknown: Vec<String> },
    Unknown { words: Vec<String> },
    Nothing,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PeakRead {
    pub ring: (usize, usize, usize),
    pub peak_step: i32,
    pub limit: f32,
    pub facts: Vec<(u32, f32, String, String, bool)>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Reply {
    pub n: u32,
    pub seed: u64,
    pub message: String,
    pub text: String,
    pub steps: u32,
    pub collisions: u64,
    pub charged_rings: usize,
    pub peaked_rings: usize,
    pub origin: (usize, usize, usize),
    pub read: Vec<PeakRead>,
    /// every charged ring: (ring, limit, how far its top particle rose, peak step)
    pub charged: Vec<((usize, usize, usize), f32, f32, i32)>,
    pub snapshot: String,
}

#[derive(Serialize, Deserialize)]
pub struct Snapshot {
    pub turn: u32,
    pub facts: Vec<Fact>,
    pub branded: Vec<(u32, u32)>,
}

pub struct Engine {
    pub wn: WordNet,
    pub block: Block,
    pub facts: Vec<Fact>,
    keys: HashMap<String, u32>,
    pub turn: u32,
    pub dir: PathBuf,
    pub last: Option<Reply>,
}

const STOP: &[&str] = &[
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "am", "do", "does", "did", "what", "who", "whom", "whose", "where",
    "when", "why", "how", "which", "tell", "me", "about", "please", "can", "could", "would", "will", "you", "your", "i", "my", "of",
    "in", "on", "at", "to", "for", "with", "and", "or", "mean", "means", "meaning", "define", "explain", "describe", "it", "this",
    "that", "there", "here", "some", "any", "know", "many", "much", "have", "has", "had", "something", "word", "so", "really",
    "thing", "things", "kind", "us", "we", "they", "he", "she", "them", "him", "her", "its", "not", "no", "yes", "ok", "okay",
];

/// How many particles one fact is branded onto.
const COPIES: usize = 3;

const CONTRACTIONS: &[(&str, &[&str])] = &[
    ("what's", &["what", "is"]),
    ("who's", &["who", "is"]),
    ("where's", &["where", "is"]),
    ("when's", &["when", "is"]),
    ("how's", &["how", "is"]),
    ("it's", &["it", "is"]),
    ("i'm", &["i", "am"]),
    ("name's", &["name", "is"]),
    ("don't", &["do", "not"]),
    ("doesn't", &["does", "not"]),
    ("isn't", &["is", "not"]),
];

/// The message's words, lowercase and as written.
pub fn tokens(msg: &str) -> (Vec<String>, Vec<String>) {
    let mut low = Vec::new();
    let mut orig = Vec::new();
    for raw in msg.split(|c: char| !(c.is_alphanumeric() || c == '\'' || c == '-')) {
        let raw = raw.trim_matches(|c| c == '\'' || c == '-');
        if raw.is_empty() {
            continue;
        }
        let l = raw.to_lowercase();
        if let Some((_, exp)) = CONTRACTIONS.iter().find(|(c, _)| *c == l) {
            for e in exp.iter() {
                low.push(e.to_string());
                orig.push(e.to_string());
            }
        } else if let Some(stem) = l.strip_suffix("'s") {
            low.push(stem.to_string());
            orig.push(raw[..raw.len() - 2].to_string());
        } else {
            low.push(l);
            orig.push(raw.to_string());
        }
    }
    (low, orig)
}

fn qword(w: &str) -> Option<QWord> {
    Some(match w {
        "what" | "which" => QWord::What,
        "who" => QWord::Who,
        "where" => QWord::Where,
        "when" => QWord::When,
        "why" => QWord::Why,
        "how" => QWord::How,
        _ => return None,
    })
}

fn strip_articles(w: &[String]) -> Vec<String> {
    w.iter().filter(|x| !matches!(x.as_str(), "a" | "an" | "the")).cloned().collect()
}

fn hash(s: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

impl Engine {
    pub fn new(wn: WordNet, cfg: Cfg, dir: PathBuf) -> Engine {
        let mut e = Engine { wn, block: Block::new(cfg), facts: vec![], keys: HashMap::new(), turn: 0, dir, last: None };
        for f in knowledge::self_facts() {
            e.brand(f);
        }
        e
    }

    // ------------------------------------------------------------ reading

    /// The noun (else verb, adjective, adverb) the words name, most common sense first.
    fn look(&self, words: &[String], prefer_noun: bool) -> Option<Topic> {
        if words.is_empty() {
            return None;
        }
        let order: &[Pos] = if prefer_noun { &[Pos::N, Pos::V, Pos::A, Pos::R] } else { &[Pos::N, Pos::A, Pos::V, Pos::R] };
        self.look_as(words, order)
    }

    fn look_as(&self, words: &[String], order: &[Pos]) -> Option<Topic> {
        if words.is_empty() {
            return None;
        }
        let phrase = words.join("_");
        for &pos in order {
            if let Some(lemma) = self.wn.base(&phrase, pos) {
                if let Some(&offset) = self.wn.senses(&lemma, pos).first() {
                    return Some(Topic { lemma, pos, offset });
                }
            }
        }
        None
    }

    /// Content words, longest phrases first (three, two, one word).
    fn topics(&self, words: &[String]) -> (Vec<Topic>, Vec<String>) {
        let content: Vec<String> = words.iter().filter(|w| !STOP.contains(&w.as_str())).cloned().collect();
        // "when did Rome fall": after do/does/did the question ends on its verb
        let verb_last = words.len() > 2 && matches!(words[1].as_str(), "do" | "does" | "did");
        let mut topics = Vec::new();
        let mut unknown = Vec::new();
        let mut i = 0;
        while i < content.len() {
            let mut took = 0;
            for len in (1..=3.min(content.len() - i)).rev() {
                let last = verb_last && i + len == content.len() && len == 1;
                let found = if last { self.look_as(&content[i..i + len], &[Pos::V, Pos::N, Pos::A, Pos::R]) } else { self.look(&content[i..i + len], true) };
                if let Some(t) = found {
                    if !topics.iter().any(|x: &Topic| x.lemma == t.lemma) {
                        topics.push(t);
                    }
                    took = len;
                    break;
                }
            }
            if took == 0 {
                if !content[i].chars().all(|c| c.is_ascii_digit()) {
                    unknown.push(content[i].clone());
                }
                took = 1;
            }
            i += took;
        }
        // a noun is what a question is about; its verbs are the asking
        if topics.iter().any(|t| t.pos == Pos::N) {
            topics.retain(|t| t.pos == Pos::N);
        }
        (topics, unknown)
    }

    pub fn read(&self, msg: &str) -> Ask {
        let (w, orig) = tokens(msg);
        if w.is_empty() {
            return Ask::Nothing;
        }
        let s = w.join(" ");
        let question = msg.trim_end().ends_with('?') || qword(&w[0]).is_some() || matches!(w[0].as_str(), "is" | "are" | "do" | "does" | "can");
        if w.len() <= 3 && matches!(w[0].as_str(), "hello" | "hi" | "hey" | "greetings" | "moi" | "terve") {
            return Ask::Greet;
        }
        if w.contains(&"thanks".to_string()) || s.starts_with("thank you") {
            return Ask::Thanks;
        }
        if matches!(w[0].as_str(), "bye" | "goodbye") || s.starts_with("see you") {
            return Ask::Bye;
        }
        match s.as_str() {
            "who are you" | "what are you" => return Ask::Myself("self:who"),
            "what is your name" => return Ask::Myself("self:name"),
            "how are you" | "how are you doing" => return Ask::Myself("self:how"),
            "what can you do" => return Ask::Myself("self:can"),
            "how do you know" | "where do you get your facts" => return Ask::Myself("self:know"),
            _ => {}
        }
        // what the user tells about themselves
        if !question {
            if w[0] == "my" {
                if let Some(k) = w.iter().position(|x| x == "is") {
                    if k > 1 && k + 1 < w.len() {
                        return Ask::Tell { key: w[1..k].join(" "), value: orig[k + 1..].join(" ") };
                    }
                }
            }
            let tell = |n: usize, key: &str| Ask::Tell { key: key.to_string(), value: orig[n..].join(" ") };
            if w.len() > 2 && w[0] == "i" {
                match w[1].as_str() {
                    "am" => return tell(2, "am"),
                    "like" | "love" | "have" => return tell(2, &w[1].clone()),
                    "live" if w.len() > 3 && w[2] == "in" => return tell(3, "live in"),
                    _ => {}
                }
            }
            if w.len() > 2 && w[0] == "call" && w[1] == "me" {
                return tell(2, "name");
            }
        }
        // what the user asks about themselves
        if s.starts_with("what did i") || s.starts_with("what have i said") {
            return Ask::RecallSaid;
        }
        if s == "who am i" {
            return Ask::Recall { key: Some("name".into()) };
        }
        if s.starts_with("do you remember") || s.contains("about me") {
            return Ask::Recall { key: None };
        }
        if let Some(k) = w.iter().position(|x| x == "my") {
            if question && k + 1 < w.len() {
                return Ask::Recall { key: Some(w[k + 1..].join(" ")) };
            }
        }
        if w.len() >= 3 && w[1] == "do" && w[2] == "i" {
            let key = match w.get(3).map(|x| x.as_str()) {
                Some("like") => "like",
                Some("love") => "love",
                Some("have") => "have",
                Some("live") => "live in",
                _ => "",
            };
            if !key.is_empty() {
                return Ask::Recall { key: Some(key.into()) };
            }
        }
        // the opposite of X
        if let Some(k) = w.iter().position(|x| x == "opposite" || x == "antonym") {
            if w.get(k + 1).map(|x| x == "of").unwrap_or(false) {
                let rest = strip_articles(&w[k + 2..]);
                return match self.look_as(&rest, &[Pos::A, Pos::N, Pos::V, Pos::R]) {
                    Some(x) => Ask::Opposite { x },
                    None => Ask::Unknown { words: rest },
                };
            }
        }
        // the R of Y
        if w.len() > 4 && qword(&w[0]).is_some() && matches!(w[1].as_str(), "is" | "are" | "was") && w[2] == "the" {
            if let Some(j) = w.iter().skip(3).position(|x| x == "of").map(|j| j + 3) {
                if j > 3 {
                    let class = self.look(&w[3..j], true);
                    let y = self.look(&strip_articles(&w[j + 1..]), true);
                    match (class, y) {
                        (Some(c), Some(y)) if c.pos == Pos::N && y.pos == Pos::N => return Ask::RelOf { class: c.lemma, y },
                        (_, None) => return Ask::Unknown { words: strip_articles(&w[j + 1..]) },
                        _ => {}
                    }
                }
            }
        }
        // is X a Y
        if matches!(w[0].as_str(), "is" | "are") && w.len() >= 3 {
            let rest = &w[1..];
            let mut splits: Vec<usize> = (1..rest.len()).filter(|&k| matches!(rest[k].as_str(), "a" | "an" | "the")).collect();
            splits.extend(1..rest.len());
            for k in splits {
                let (l, r) = (strip_articles(&rest[..k]), strip_articles(&rest[k..]));
                if l.is_empty() || r.is_empty() {
                    continue;
                }
                if let (Some(x), Some(y)) = (self.look(&l, true), self.look(&r, true)) {
                    if x.pos == Pos::N && y.pos == Pos::N {
                        return Ask::IsA { x, y };
                    }
                }
            }
        }
        let q = qword(&w[0]).unwrap_or(QWord::What);
        let (topics, unknown) = self.topics(&w);
        if topics.is_empty() {
            if unknown.is_empty() {
                return Ask::Nothing;
            }
            return Ask::Unknown { words: unknown };
        }
        Ask::About { q, topics, unknown }
    }

    // ------------------------------------------------------------ branding

    fn home(&self, topic: &str) -> usize {
        (hash(topic) % self.block.rings() as u64) as usize
    }

    /// Brand a fact onto the free particle nearest its topic's home ring.
    pub fn brand(&mut self, f: Fact) -> u32 {
        let key = f.key();
        if let Some(&id) = self.keys.get(&key) {
            return id;
        }
        let id = self.facts.len() as u32;
        let home = self.home(&f.topic);
        self.keys.insert(key, id);
        self.facts.push(f);
        // branded onto several particles, each on a ring of its own
        for _ in 0..COPIES {
            if let Some(p) = self.free_near(home) {
                self.block.parts[p].fact = Some(id);
            }
        }
        id
    }

    fn free_near(&self, home: usize) -> Option<usize> {
        let b = &self.block;
        let pp = b.cfg.particles_per_ring;
        let mut seen = vec![false; b.rings()];
        let mut queue = std::collections::VecDeque::from([home]);
        seen[home] = true;
        let mut order = Vec::new();
        while let Some(r) = queue.pop_front() {
            order.push(r);
            // a ring of its own for each fact, so the wave chooses between facts
            if (0..pp).all(|q| b.parts[r * pp + q].fact.is_none()) {
                return Some(r * pp);
            }
            let (i, j, z) = b.coords(r);
            let mut push = |i: i64, j: i64, z: i64| {
                if i >= 0 && j >= 0 && z >= 0 && (i as usize) < b.rx && (j as usize) < b.ry && (z as usize) < b.rz {
                    let n = b.ring(i as usize, j as usize, z as usize);
                    if !seen[n] {
                        seen[n] = true;
                        queue.push_back(n);
                    }
                }
            };
            let (i, j, z) = (i as i64, j as i64, z as i64);
            push(i + 1, j, z);
            push(i - 1, j, z);
            push(i, j + 1, z);
            push(i, j - 1, z);
            push(i, j, z + 1);
            push(i, j, z - 1);
        }
        // the block is full of one-fact rings: share a ring, nearest first
        order.iter().find_map(|&r| (0..pp).find(|&q| b.parts[r * pp + q].fact.is_none()).map(|q| r * pp + q))
    }

    fn brand_topic(&mut self, t: &Topic, cue: &str) -> Vec<u32> {
        let facts = knowledge::synset_facts(&self.wn, t.pos, t.offset, &t.lemma, self.turn, cue);
        facts.into_iter().map(|f| self.brand(f)).collect()
    }

    fn turn_fact(&self, rel: Rel, topic: &str, subject: &str, object: Vec<String>, said: &str) -> Fact {
        Fact {
            rel,
            topic: topic.to_string(),
            subject: subject.to_string(),
            object,
            proper: false,
            lexfile: 0,
            pos: None,
            strike: Strike { source: Source::Turn { said: said.to_string() }, turn: self.turn, cue: topic.to_string() },
        }
    }

    // ------------------------------------------------------------ answering

    pub fn answer(&mut self, msg: &str, seed: u64, viz: Option<&mut dyn FnMut(&Block, u32, u32)>) -> Result<Reply> {
        self.turn += 1;
        let ask = self.read(msg);
        let said = self.turn_fact(Rel::Said, "you:said", "you", vec![msg.trim().to_string()], msg.trim());
        let said_id = self.brand(said);

        // what to charge, how many rings to read, and what to say around them
        let mut charge: HashMap<u32, f32> = HashMap::new();
        let mut take = 2usize;
        let mut before: Vec<String> = vec![];
        let mut after: Vec<String> = vec![];
        let mut origin_topic = String::from("you:said");
        let mut reading = true;
        let w = |rel: Rel| -> f32 {
            match rel {
                Rel::Def => 1.0,
                Rel::IsA | Rel::InstanceOf => 0.9,
                Rel::PartOf => 0.8,
                Rel::Synonym => 0.75,
                Rel::HasPart | Rel::Antonym => 0.7,
                Rel::MemberOf => 0.65,
                Rel::Example => 0.6,
                _ => 0.0,
            }
        };
        match &ask {
            Ask::Greet => {
                before.push("Hello. Ask me about a word, or tell me something about yourself.".into());
                reading = false;
            }
            Ask::Thanks => {
                before.push("You're welcome.".into());
                reading = false;
            }
            Ask::Bye => {
                before.push("Goodbye. The wave keeps rolling.".into());
                reading = false;
            }
            Ask::Nothing => {
                before.push("I don't know what you are asking.".into());
                reading = false;
            }
            Ask::Myself(topic) => {
                for (i, f) in self.facts.iter().enumerate() {
                    if f.topic == *topic {
                        charge.insert(i as u32, 1.0);
                    }
                }
                origin_topic = topic.to_string();
                take = 1;
            }
            Ask::Tell { key, value } => {
                let f = self.turn_fact(Rel::Memory, &format!("you:{key}"), key, vec![value.clone()], msg.trim());
                origin_topic = f.topic.clone();
                let id = self.brand(f);
                charge.insert(id, 1.0);
                before.push("I'll remember that.".into());
                take = 1;
            }
            Ask::Recall { key } => {
                let ids: Vec<u32> = (0..self.facts.len() as u32)
                    .filter(|&i| {
                        let f = &self.facts[i as usize];
                        f.rel == Rel::Memory && key.as_ref().map(|k| f.subject == *k).unwrap_or(true)
                    })
                    .collect();
                for &i in &ids {
                    let age = self.turn.saturating_sub(self.facts[i as usize].strike.turn) as f32;
                    charge.insert(i, (1.0 - 0.02 * age).max(0.3));
                }
                if ids.is_empty() {
                    match key {
                        Some(k) => before.push(format!("I don't know your {k}. You haven't told me.")),
                        None => before.push("You haven't told me anything about yourself yet.".into()),
                    }
                    reading = false;
                }
                origin_topic = key.as_ref().map(|k| format!("you:{k}")).unwrap_or("you:said".into());
                take = if key.is_some() { 1 } else { 3 };
            }
            Ask::RecallSaid => {
                let ids: Vec<u32> =
                    (0..self.facts.len() as u32).filter(|&i| i != said_id && self.facts[i as usize].rel == Rel::Said).collect();
                for &i in &ids {
                    let age = self.turn.saturating_sub(self.facts[i as usize].strike.turn) as f32;
                    charge.insert(i, (1.0 - 0.1 * age).max(0.2));
                }
                if ids.is_empty() {
                    before.push("You haven't said anything before this.".into());
                    reading = false;
                }
                take = 2;
            }
            Ask::IsA { x, y } => {
                origin_topic = x.lemma.clone();
                let facts = self.brand_topic(x, &x.lemma);
                match knowledge::chain(&self.wn, (x.pos, x.offset), &y.lemma) {
                    Some(path) => {
                        let f = knowledge::chain_fact(&self.wn, &path, &x.lemma, &y.lemma, self.turn, &y.lemma);
                        let id = self.brand(f);
                        charge.insert(id, 1.0);
                        take = 1;
                    }
                    None => {
                        let yn = knowledge::synset_facts(&self.wn, y.pos, y.offset, &y.lemma, 0, "")
                            .first()
                            .map(|f| if f.proper { f.subject.clone() } else { format!("a {}", f.subject) })
                            .unwrap_or(y.lemma.clone());
                        before.push(format!("I have nothing that says {} is {}. What I do know:", self.np(x), yn));
                        for id in facts {
                            if self.facts[id as usize].rel == Rel::IsA || self.facts[id as usize].rel == Rel::InstanceOf {
                                charge.insert(id, 1.0);
                            }
                        }
                        take = 1;
                    }
                }
            }
            Ask::RelOf { class, y } => {
                origin_topic = y.lemma.clone();
                let facts = self.brand_topic(y, &y.lemma);
                match knowledge::rel_of(&self.wn, (y.pos, y.offset), class) {
                    Some(found) => {
                        let f = knowledge::rel_of_fact(&self.wn, (y.pos, y.offset), found, class, &y.lemma, self.turn, class);
                        let id = self.brand(f);
                        charge.insert(id, 1.0);
                        take = 1;
                    }
                    None => {
                        before.push(format!("I don't know the {} of {}. What I do know:", crate::wordnet::show(class), self.np(y)));
                        for id in facts {
                            charge.insert(id, w(self.facts[id as usize].rel));
                        }
                        take = 1;
                    }
                }
            }
            Ask::Opposite { x } => {
                origin_topic = x.lemma.clone();
                let facts = self.brand_topic(x, &x.lemma);
                let ants: Vec<u32> = facts.iter().cloned().filter(|&i| self.facts[i as usize].rel == Rel::Antonym).collect();
                if ants.is_empty() {
                    before.push(format!("I don't know an opposite of {}. What I do know:", crate::wordnet::show(&x.lemma)));
                    for id in facts {
                        if self.facts[id as usize].rel == Rel::Def {
                            charge.insert(id, 1.0);
                        }
                    }
                } else {
                    for id in ants {
                        charge.insert(id, 1.0);
                    }
                }
                take = 1;
            }
            Ask::Unknown { words } => {
                before.push(format!("I don't know what \"{}\" is. Nothing in the word network or our talk holds it.", words.join(" ")));
                reading = false;
            }
            Ask::About { q, topics, unknown } => {
                origin_topic = topics[0].lemma.clone();
                if !unknown.is_empty() {
                    before.push(format!("I don't know \"{}\".", unknown.join("\", \"")));
                }
                let mut gap: Option<String> = None;
                for t in topics {
                    let facts = self.brand_topic(t, &t.lemma);
                    let rels: Vec<Rel> = facts.iter().map(|&i| self.facts[i as usize].rel).collect();
                    let lexfile = self.wn.get((t.pos, t.offset)).map(|s| s.lexfile).unwrap_or(0);
                    for &id in &facts {
                        let rel = self.facts[id as usize].rel;
                        let c = match q {
                            QWord::Where => match rel {
                                Rel::PartOf => 1.0,
                                Rel::Def => 0.85,
                                _ => 0.0,
                            },
                            QWord::Who => match rel {
                                Rel::Def => 1.0,
                                Rel::InstanceOf | Rel::IsA => 0.9,
                                _ => 0.4,
                            },
                            _ => w(rel),
                        };
                        if c > 0.0 {
                            charge.insert(id, c);
                        }
                    }
                    match q {
                        QWord::Where if !rels.contains(&Rel::PartOf) => {
                            gap = Some(format!("I don't know where {} is.", self.np(t)));
                        }
                        QWord::Who if lexfile != LEX_PERSON && !rels.contains(&Rel::InstanceOf) => gap = Some("I don't know who.".into()),
                        QWord::When => gap = Some("I don't know when.".into()),
                        QWord::Why => gap = Some("I don't know why.".into()),
                        QWord::How => gap = Some("I don't know how.".into()),
                        _ => {}
                    }
                }
                take = (2 * topics.len()).min(4);
                if let Some(g) = gap {
                    before.push(format!("{g} What I do know:"));
                    take = topics.len().min(2);
                }
            }
        }

        // brand-checked: the strikes, the block's state before the wave
        for p in self.block.parts.iter_mut() {
            p.charge = p.fact.and_then(|f| charge.get(&f).cloned()).unwrap_or(0.0);
        }
        let snapshot = self.save_snapshot()?;
        let origin = self.home(&origin_topic);
        let mut b = std::mem::replace(&mut self.block, Block::new(Cfg { nx: 2, ny: 2, nz: 1, ..Default::default() }));
        b.stir(seed);
        b.charge(origin);
        let steps = b.wave_steps();
        let mut viz = viz;
        for t in 0..steps {
            b.step(true);
            if let Some(v) = viz.as_mut() {
                v(&b, t + 1, steps);
            }
        }

        // the last charged rings to peak, in the order they peaked
        let pp = b.cfg.particles_per_ring;
        let charged: Vec<usize> = (0..b.rings()).filter(|&r| b.limit[r] > 1.0).collect();
        let mut peaked: Vec<usize> = charged.iter().cloned().filter(|&r| b.peak[r] >= 0).collect();
        peaked.sort_by_key(|&r| (b.peak[r], r));
        let mut read = Vec::new();
        let mut said_keys = std::collections::HashSet::new();
        let mut sentences = Vec::new();
        let mut rings_read = 0;
        if reading {
            for &r in peaked.iter().rev() {
                if rings_read == take {
                    break;
                }
                let mut held: Vec<(usize, f32)> =
                    (0..pp).map(|q| r * pp + q).filter(|&j| b.parts[j].charge > 0.0).map(|j| (j, b.parts[j].charge)).collect();
                held.sort_by(|a, c| c.1.partial_cmp(&a.1).unwrap());
                let mut pr = PeakRead { ring: b.coords(r), peak_step: b.peak[r], limit: b.limit[r], facts: vec![] };
                let mut any = false;
                for (j, c) in held {
                    let id = b.parts[j].fact.unwrap();
                    let f = &self.facts[id as usize];
                    let ok = verify(&self.wn, f);
                    // the wave picks the wording: the particle's direction against its ring's spin
                    let v = ((b.parts[j].dir * b.omega[r].signum()) < 0.0) as u8;
                    let text = phrase(&self.wn, f, v);
                    pr.facts.push((id, c, format!("{:?}", f.strike), text.clone(), ok));
                    if ok && said_keys.insert(f.key()) {
                        sentences.push(text);
                        any = true;
                    }
                }
                read.push(pr);
                if any {
                    rings_read += 1;
                }
            }
        }
        read.reverse();
        let iv = sentences.len();
        sentences.reverse();
        let mut parts = before.clone();
        if reading && iv == 0 {
            if before.is_empty() {
                parts.push("I don't know. No ring that holds knowledge about this rose to its limit.".into());
            } else {
                parts.push("(nothing more rose to its limit)".into());
            }
        }
        parts.extend(sentences);
        parts.extend(after.drain(..));
        let text = parts.join(" ");
        let reply = Reply {
            n: self.turn,
            seed,
            message: msg.trim().to_string(),
            text,
            steps,
            collisions: b.collisions,
            charged_rings: charged.len(),
            peaked_rings: (0..b.rings()).filter(|&r| b.peak[r] >= 0).count(),
            origin: b.coords(origin),
            read,
            charged: charged.iter().map(|&r| (b.coords(r), b.limit[r], b.fill(r) * b.limit[r], b.peak[r])).collect(),
            snapshot,
        };
        self.block = b;
        self.log(&reply)?;
        self.last = Some(reply.clone());
        Ok(reply)
    }

    fn save_snapshot(&self) -> Result<String> {
        std::fs::create_dir_all(&self.dir)?;
        let branded: Vec<(u32, u32)> =
            self.block.parts.iter().enumerate().filter_map(|(i, p)| p.fact.map(|f| (i as u32, f))).collect();
        let snap = Snapshot { turn: self.turn - 1, facts: self.facts.clone(), branded };
        let name = format!("snap-{:04}.json", self.turn);
        std::fs::write(self.dir.join(&name), serde_json::to_vec(&snap)?)?;
        Ok(name)
    }

    /// Put the block back as it was before a reply's wave (for replay).
    pub fn load_snapshot(&mut self, snap: Snapshot) {
        self.facts = snap.facts;
        self.keys = self.facts.iter().enumerate().map(|(i, f)| (f.key(), i as u32)).collect();
        for p in self.block.parts.iter_mut() {
            p.fact = None;
        }
        for (p, f) in snap.branded {
            self.block.parts[p as usize].fact = Some(f);
        }
        self.turn = snap.turn;
    }

    fn log(&self, r: &Reply) -> Result<()> {
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new().create(true).append(true).open(self.dir.join("replies.jsonl"))?;
        writeln!(f, "{}", serde_json::to_string(r)?)?;
        std::fs::write(self.dir.join("last.json"), serde_json::to_vec_pretty(r)?)?;
        Ok(())
    }
}

impl Engine {
    fn np(&self, t: &Topic) -> String {
        let w = self
            .wn
            .get((t.pos, t.offset))
            .and_then(|s| s.words.iter().find(|x| x.to_lowercase() == t.lemma).map(|x| crate::wordnet::show(x)))
            .unwrap_or(crate::wordnet::show(&t.lemma));
        np_word(&w, t.pos)
    }
}

fn np_word(w: &str, pos: Pos) -> String {
    let w = w.to_string();
    if w.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) {
        w
    } else if pos == Pos::N {
        if "aeiou".contains(w.chars().next().unwrap_or('x')) {
            format!("an {w}")
        } else {
            format!("a {w}")
        }
    } else {
        format!("\"{w}\"")
    }
}

pub fn fresh_seed() -> u64 {
    let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(1);
    Rng::new(t ^ (std::process::id() as u64) << 32).next()
}
