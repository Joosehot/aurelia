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
use crate::ethics;
use crate::store::{self, Index, Role, Src, Store};
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
    /// better, best, favourite: nothing branded is an opinion
    Opinion,
    /// the weather, today, the news: nothing branded is about now
    Now,
    /// tell me something: the wave picks among branded senses
    Something,
    /// is it wrong to X, should I X: the moral core and what the philosophers say
    Moral { words: Vec<String>, acts: Vec<Topic> },
    /// a message that asks for harm to people
    Refuse { words: Vec<String> },
    /// a message about wanting to die
    Care,
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
    #[serde(default)]
    pub say: Vec<(u32, Fact)>,
}

pub struct Engine {
    pub wn: WordNet,
    pub block: Block,
    /// world and philosophical knowledge; node nodes.len() + i is store item i
    pub store: Store,
    index: Index,
    /// node ids 0..nodes.len() are WordNet's synsets; node nodes.len() + i is facts[i]
    pub nodes: Vec<(Pos, u32)>,
    node_of: HashMap<(Pos, u32), u32>,
    /// the particle each node is branded on
    slot: HashMap<u32, usize>,
    /// what the conversation branded: memory, what was said, derived facts, Aurelia about itself
    pub facts: Vec<Fact>,
    keys: HashMap<String, u32>,
    pub turn: u32,
    pub dir: PathBuf,
    pub brand_dir: PathBuf,
    /// false for a replay: it must not write over the branded block
    pub persist: bool,
    pub last: Option<Reply>,
}

#[derive(Serialize, Deserialize)]
struct Memory {
    turn: u32,
    facts: Vec<Fact>,
}

const STOP: &[&str] = &[
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "am", "do", "does", "did", "what", "who", "whom", "whose", "where",
    "when", "why", "how", "which", "tell", "me", "about", "please", "can", "could", "would", "will", "you", "your", "i", "my", "of",
    "in", "on", "at", "to", "for", "with", "and", "or", "mean", "means", "meaning", "define", "explain", "describe", "it", "this",
    "that", "there", "here", "some", "any", "know", "many", "much", "have", "has", "had", "something", "word", "so", "really",
    "thing", "things", "kind", "us", "we", "they", "he", "she", "them", "him", "her", "its", "not", "no", "yes", "ok", "okay",
];

const FREE: u32 = u32::MAX;
/// A topic known only from the world knowledge, with no WordNet sense.
const NO_SYNSET: u32 = u32::MAX;

fn ethic_facts() -> Vec<Fact> {
    ethics::PRINCIPLES
        .iter()
        .enumerate()
        .map(|(i, (text, _))| Fact {
            rel: Rel::Ethic,
            topic: format!("ethic:{i}"),
            subject: "Aurelia".into(),
            object: vec![text.to_string()],
            proper: true,
            lexfile: 0,
            pos: None,
            strike: Strike { source: Source::Ethic { line: i }, turn: 0, cue: format!("ethic:{i}") },
            via: None,
        })
        .collect()
}

/// Words of the talk around a message, not what it is about.
const FILLER: &[&str] = &[
    "cool", "so", "ok", "okay", "well", "um", "uh", "hmm", "oh", "wow", "yeah", "yep", "yes", "no", "nice", "great", "lol",
    "haha", "anyway", "and", "but", "alright", "right", "hey", "btw", "please",
];
const TAIL: &[&str] = &["btw", "lol", "haha", "though", "then", "anyway", "please", "too", "now"];
const OPINION: &[&str] = &["better", "best", "worse", "worst", "favorite", "favourite", "prefer", "nicer", "cooler"];
const NOW: &[&str] = &["weather", "today", "tonight", "tomorrow", "yesterday", "news", "latest", "currently"];

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
    ("whats", &["what", "is"]),
    ("whos", &["who", "is"]),
    ("wheres", &["where", "is"]),
    ("hows", &["how", "is"]),
    ("im", &["i", "am"]),
    ("dont", &["do", "not"]),
    ("doesnt", &["does", "not"]),
    ("isnt", &["is", "not"]),
    ("thats", &["that", "is"]),
    ("u", &["you"]),
    ("ur", &["your"]),
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
    /// The block with WordNet branded into it. The branding is done once and kept in
    /// `brand_dir`; later starts load it, with everything the conversations branded since.
    pub fn new(wn: WordNet, store: Store, cfg: Cfg, dir: PathBuf, brand_dir: PathBuf) -> Result<Engine> {
        let mut nodes: Vec<(Pos, u32)> = wn.synsets.keys().cloned().collect();
        nodes.sort();
        let node_of = nodes.iter().enumerate().map(|(i, k)| (*k, i as u32)).collect();
        let mut e = Engine {
            wn,
            block: Block::new(cfg),
            store,
            index: HashMap::new(),
            nodes,
            node_of,
            slot: HashMap::new(),
            facts: vec![],
            keys: HashMap::new(),
            turn: 0,
            dir,
            brand_dir,
            persist: true,
            last: None,
        };
        let knowledge = e.nodes.len() + e.store.items.len();
        if knowledge + 10_000 > e.block.parts.len() {
            anyhow::bail!(
                "{} WordNet senses and {} store items (and room to talk) do not fit on {} particles; make the block bigger",
                e.nodes.len(),
                e.store.items.len(),
                e.block.parts.len()
            );
        }
        let parts_file = e.brand_dir.join("particles.bin");
        if parts_file.exists() {
            let bytes = std::fs::read(&parts_file)?;
            let mem: Memory = serde_json::from_slice(&std::fs::read(e.brand_dir.join("memory.json"))?)?;
            if bytes.len() != e.block.parts.len() * 4 {
                anyhow::bail!("{} was branded on a block of another size; move it away to brand anew", parts_file.display());
            }
            e.facts = mem.facts;
            e.turn = mem.turn;
            e.keys = e.facts.iter().enumerate().map(|(i, f)| (f.key(), i as u32)).collect();
            e.index = serde_json::from_slice(&std::fs::read(e.brand_dir.join("index.json"))?)?;
            for (p, c) in bytes.chunks(4).enumerate() {
                let n = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                if n != FREE {
                    e.block.parts[p].fact = Some(n);
                    e.slot.insert(n, p);
                }
            }
            eprintln!(
                "branded block loaded: {} WordNet senses, {} world and book items, {} branded by talk (turn {})",
                e.nodes.len(),
                e.store.items.len(),
                e.facts.len(),
                e.turn
            );
        } else {
            // every sense, proposition and sentence onto a particle of its own, once. What is
            // about one word lies together: the nodes are sorted by the word they are about,
            // and laid along a Morton curve through the block, so neighbours in the order are
            // neighbours in space. Every 17th particle stays free for what the talk brands.
            let t = std::time::Instant::now();
            e.index = store::build_index(&e.store, &e.wn);
            let s = e.nodes.len();
            let mut about: Vec<(u8, String)> = vec![(9, String::new()); knowledge];
            for (k, (pos, off)) in e.nodes.iter().enumerate() {
                if let Some(syn) = e.wn.get((*pos, *off)) {
                    about[k] = (0, syn.words[0].to_lowercase());
                }
            }
            for (key, entries) in &e.index {
                for &(i, role) in entries {
                    let rank = match role {
                        Role::Article => 1,
                        Role::BookSays => 2,
                        Role::Subject => 3,
                        Role::InArticle => 4,
                        Role::BookMentions => 5,
                    };
                    let slot = &mut about[s + i as usize];
                    if (rank, key) < (slot.0, &slot.1) {
                        *slot = (rank, key.clone());
                    }
                }
            }
            let mut nodes: Vec<usize> = (0..knowledge).collect();
            nodes.sort_by(|&a, &b| about[a].1.cmp(&about[b].1).then(about[a].0.cmp(&about[b].0)).then(a.cmp(&b)));
            let b = &e.block;
            let morton = |r: usize| -> u64 {
                let (i, j, z) = b.coords(r);
                let mut m = 0u64;
                for bit in 0..10 {
                    m |= (((i >> bit) & 1) as u64) << (3 * bit) | (((j >> bit) & 1) as u64) << (3 * bit + 1) | (((z >> bit) & 1) as u64) << (3 * bit + 2);
                }
                m
            };
            let mut rings: Vec<usize> = (0..b.rings()).collect();
            rings.sort_by_key(|&r| (morton(r), r));
            let pp = b.cfg.particles_per_ring;
            let slots: Vec<usize> = rings.iter().flat_map(|&r| (0..pp).map(move |q| r * pp + q)).enumerate().filter(|(k, _)| k % 17 != 16).map(|(_, p)| p).collect();
            if slots.len() < knowledge {
                anyhow::bail!("{} knowledge nodes do not fit on {} branded slots; make the block bigger", knowledge, slots.len());
            }
            for (k, &n) in nodes.iter().enumerate() {
                e.block.parts[slots[k]].fact = Some(n as u32);
                e.slot.insert(n as u32, slots[k]);
            }
            for f in knowledge::self_facts() {
                e.brand(f);
            }
            for f in ethic_facts() {
                e.brand(f);
            }
            std::fs::create_dir_all(&e.brand_dir)?;
            std::fs::write(e.brand_dir.join("index.json"), serde_json::to_vec(&e.index)?)?;
            e.save_brand()?;
            eprintln!(
                "branded once: {} WordNet senses, {} world and book items and the moral core onto {} particles ({:.0} s), kept in {}",
                e.nodes.len(),
                e.store.items.len(),
                e.block.parts.len(),
                t.elapsed().as_secs_f32(),
                e.brand_dir.display()
            );
        }
        Ok(e)
    }

    pub fn save_brand(&self) -> Result<()> {
        std::fs::create_dir_all(&self.brand_dir)?;
        let mut bytes = Vec::with_capacity(self.block.parts.len() * 4);
        for p in &self.block.parts {
            bytes.extend_from_slice(&p.fact.unwrap_or(FREE).to_le_bytes());
        }
        std::fs::write(self.brand_dir.join("particles.bin"), bytes)?;
        std::fs::write(self.brand_dir.join("memory.json"), serde_json::to_vec(&Memory { turn: self.turn, facts: self.facts.clone() })?)?;
        Ok(())
    }

    /// The first node id of what the talk branded.
    fn talk_base(&self) -> u32 {
        (self.nodes.len() + self.store.items.len()) as u32
    }

    fn fact_node(&self, id: u32) -> u32 {
        self.talk_base() + id
    }

    fn item_fact(&self, i: u32, cue: &str) -> Fact {
        let it = &self.store.items[i as usize];
        Fact {
            rel: match it.src {
                Src::Wikipedia => Rel::World,
                Src::Primo => Rel::Primo,
                Src::Book => Rel::Book,
            },
            topic: cue.to_string(),
            subject: it.who.clone(),
            object: vec![it.text.clone(), it.work.clone(), it.locator.clone()],
            proper: true,
            lexfile: 0,
            pos: None,
            strike: Strike { source: Source::Store { row: i }, turn: self.turn, cue: cue.to_string() },
            via: None,
        }
    }

    /// The world and book items about a word, with the charge each gets for this ask.
    fn item_charges(&self, lemma: &str, q: QWord, abstract_: bool) -> Vec<(u32, f32, Fact)> {
        let Some(entries) = self.index.get(lemma) else { return vec![] };
        let s = self.nodes.len() as u32;
        let mut out: Vec<(u32, f32)> = Vec::new();
        for &(i, role) in entries {
            let it = &self.store.items[i as usize];
            let short = it.text.split_whitespace().count() <= 25;
            // "The dog follows Filip back to his flat" is about one dog, not about dogs
            let first = it.subj.split_whitespace().next().unwrap_or("").to_lowercase();
            let particular = matches!(first.as_str(), "the" | "this" | "that" | "these" | "those" | "his" | "her" | "their" | "its" | "our" | "my" | "your");
            // "Love is Blind. Love is not Over" are names of songs, not what love is
            let caps = it.text.split_whitespace().skip(1).filter(|x| x.chars().next().map(|c| c.is_uppercase()).unwrap_or(false)).count();
            let named = caps >= 2 && !it.subj.chars().skip(1).any(|c| c.is_uppercase());
            if it.src != Src::Book && (particular || named) && role != Role::Article {
                continue;
            }
            let located = it.text.contains(" in ") || it.text.contains("located") || it.text.contains("capital");
            let c = match (q, role) {
                (QWord::Where, Role::Article | Role::InArticle) if located => 0.95,
                (QWord::Where, _) => 0.0,
                (_, Role::Article) => {
                    if it.kind == "definition" {
                        0.97
                    } else {
                        0.92
                    }
                }
                (_, Role::InArticle) => 0.55,
                (_, Role::Subject) => {
                    if it.src == Src::Primo {
                        0.8
                    } else {
                        0.75
                    }
                }
                (_, Role::BookSays) => {
                    if abstract_ {
                        0.98
                    } else {
                        0.7
                    }
                }
                (_, Role::BookMentions) => 0.0,
            };
            if c > 0.0 {
                out.push((i, if short { c } else { c - 0.05 }));
            }
        }
        out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
        out.truncate(400);
        out.into_iter().map(|(i, c)| (s + i, c, self.item_fact(i, lemma))).collect()
    }

    /// Book sentences on an act that speak of right and wrong.
    fn moral_items(&self, lemma: &str) -> Vec<(u32, f32, Fact)> {
        const MORAL: &[&str] = &["wrong", "evil", "good", "virtue", "vice", "duty", "just", "unjust", "right", "sin", "wicked", "moral", "ought"];
        let Some(entries) = self.index.get(lemma) else { return vec![] };
        let s = self.nodes.len() as u32;
        let mut out = Vec::new();
        for &(i, role) in entries {
            let it = &self.store.items[i as usize];
            if it.src != Src::Book {
                continue;
            }
            let low = it.text.to_lowercase();
            let moral = low.split(|c: char| !c.is_alphabetic()).any(|w| MORAL.contains(&w));
            let c = match (role, moral) {
                (Role::BookSays, true) => 0.92,
                (Role::BookSays, false) => 0.75,
                (Role::BookMentions, true) => 0.85,
                _ => 0.0,
            };
            if c > 0.0 && it.text.split_whitespace().count() <= 45 {
                out.push((s + i, c, self.item_fact(i, lemma)));
            }
            if out.len() >= 300 {
                break;
            }
        }
        out
    }

    fn node_ring(&self, n: u32) -> Option<usize> {
        self.slot.get(&n).map(|p| p / self.block.cfg.particles_per_ring)
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
        // not in the word network, but an article or a subject of the world knowledge
        let key = phrase.to_lowercase();
        if self.index.get(&key).map(|e| e.iter().any(|(_, r)| matches!(r, Role::Article | Role::Subject))).unwrap_or(false) {
            return Some(Topic { lemma: key, pos: Pos::N, offset: NO_SYNSET });
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

    /// The message read whole: of its sentences, the one that asks (else the last that says
    /// something), without the talk's fillers around it.
    pub fn read(&self, msg: &str) -> Ask {
        // the moral core reads every message first
        if ethics::wants_self_harm(msg) {
            return Ask::Care;
        }
        let (all, _) = tokens(msg);
        if ethics::harms_people(&all) {
            return Ask::Refuse { words: all };
        }
        let mut sentences: Vec<(String, bool)> = Vec::new();
        let mut cur = String::new();
        for ch in msg.chars() {
            cur.push(ch);
            if matches!(ch, '.' | '!' | '?' | ';') {
                sentences.push((cur.trim().to_string(), ch == '?'));
                cur.clear();
            }
        }
        if !cur.trim().is_empty() {
            sentences.push((cur.trim().to_string(), false));
        }
        let mut kept: Vec<(Vec<String>, Vec<String>, bool)> = Vec::new();
        for (s, q) in sentences {
            let (mut w, mut o) = tokens(&s);
            while w.first().map(|x| FILLER.contains(&x.as_str())).unwrap_or(false) {
                w.remove(0);
                o.remove(0);
            }
            while w.last().map(|x| FILLER.contains(&x.as_str()) || TAIL.contains(&x.as_str())).unwrap_or(false) {
                w.pop();
                o.pop();
            }
            if !w.is_empty() {
                let asks = q || qword(&w[0]).is_some();
                kept.push((w, o, asks));
            }
        }
        let Some(pick) = kept.iter().find(|k| k.2).or(kept.last()).cloned() else {
            return if tokens(msg).0.is_empty() { Ask::Nothing } else { Ask::Greet };
        };
        self.read_words(pick.0, pick.1, pick.2)
    }

    /// "is it wrong to lie", "is lying wrong", "should I steal": the act asked about.
    fn moral_act(&self, w: &[String]) -> Option<Vec<String>> {
        const JUDGE: &[&str] = &["wrong", "right", "ok", "okay", "good", "bad", "evil", "moral", "immoral", "fine", "acceptable", "ethical", "unethical", "sin"];
        let content = |x: &[String]| -> Vec<String> { x.iter().filter(|y| !STOP.contains(&y.as_str()) && !JUDGE.contains(&y.as_str())).cloned().collect() };
        if w.len() >= 4 && w[0] == "is" && w[1] == "it" && JUDGE.contains(&w[2].as_str()) {
            let from = w.iter().position(|x| x == "to").map(|k| k + 1).unwrap_or(3);
            let act = content(&w[from.min(w.len())..]);
            return if act.is_empty() { None } else { Some(act) };
        }
        if w.len() >= 3 && matches!(w[0].as_str(), "is" | "are") && JUDGE.contains(&w[w.len() - 1].as_str()) {
            let act = content(&w[1..w.len() - 1]);
            return if act.is_empty() { None } else { Some(act) };
        }
        if w.len() >= 3 && w[0] == "should" && matches!(w[1].as_str(), "i" | "we" | "you" | "people" | "one" | "anyone") {
            let act = content(&w[2..]);
            return if act.is_empty() { None } else { Some(act) };
        }
        None
    }

    fn read_words(&self, w: Vec<String>, orig: Vec<String>, asks: bool) -> Ask {
        let s = w.join(" ");
        let question = asks || matches!(w[0].as_str(), "is" | "are" | "do" | "does" | "can" | "could" | "will" | "would");
        if s.starts_with("can you") || s.starts_with("could you") || s.starts_with("will you") || s.starts_with("would you") || s == "help" || s.starts_with("help me") {
            return Ask::Myself("self:can");
        }
        if let Some(act) = self.moral_act(&w) {
            let acts = act.iter().filter_map(|x| self.look_as(&[x.clone()], &[Pos::N, Pos::V])).collect();
            let mut words = act.clone();
            words.extend(act.iter().filter_map(|x| self.wn.base(x, Pos::V)));
            words.extend(act.iter().filter_map(|x| self.wn.base(x, Pos::N)));
            return Ask::Moral { words, acts };
        }
        if w.iter().any(|x| OPINION.contains(&x.as_str())) || s.contains("do you like") || s.contains("do you think") {
            return Ask::Opinion;
        }
        if question && (w.iter().any(|x| NOW.contains(&x.as_str())) || s.starts_with("what time") || s.starts_with("what day") || s.starts_with("what date")) {
            return Ask::Now;
        }
        if s.starts_with("tell me something") || s.starts_with("tell me a fact") || s.starts_with("say something") || s == "surprise me" {
            return Ask::Something;
        }
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
        // a derived fact sits beside the WordNet node it is said from; memory beside its topic's home
        let home = match f.via.and_then(|k| self.node_of.get(&k)).and_then(|&n| self.node_ring(n)) {
            Some(r) => r,
            None => self.home(&f.topic),
        };
        self.keys.insert(key, id);
        self.facts.push(f);
        let node = self.fact_node(id);
        if let Some(p) = self.free_near(home) {
            self.block.parts[p].fact = Some(node);
            self.slot.insert(node, p);
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

    /// What a word's WordNet node and its neighbours in the network say about it,
    /// each fact with the branded node it is said from.
    fn topic_facts(&self, t: &Topic, cue: &str) -> Vec<(u32, Fact)> {
        knowledge::synset_facts(&self.wn, t.pos, t.offset, &t.lemma, self.turn, cue)
            .into_iter()
            .filter_map(|f| f.via.and_then(|k| self.node_of.get(&k).cloned()).map(|n| (n, f)))
            .collect()
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
            via: None,
        }
    }

    // ------------------------------------------------------------ answering

    pub fn answer(&mut self, msg: &str, seed: u64, viz: Option<&mut dyn FnMut(&Block, u32, u32)>) -> Result<Reply> {
        self.turn += 1;
        let ask = self.read(msg);
        let said = self.turn_fact(Rel::Said, "you:said", "you", vec![msg.trim().to_string()], msg.trim());
        let said_id = self.brand(said);

        // which branded nodes to charge, what each says, how many rings to read
        let mut charge: HashMap<u32, f32> = HashMap::new();
        let mut say: HashMap<u32, Fact> = HashMap::new();
        let put = |charge: &mut HashMap<u32, f32>, say: &mut HashMap<u32, Fact>, node: u32, c: f32, f: Option<Fact>| {
            if c > *charge.get(&node).unwrap_or(&0.0) {
                charge.insert(node, c);
                if let Some(f) = f {
                    say.insert(node, f);
                }
            }
        };
        let mut take = 2usize;
        let mut before: Vec<String> = vec![];
        let mut origin: Option<usize> = None;
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
        let topic_ring = |e: &Engine, t: &Topic| e.node_of.get(&(t.pos, t.offset)).and_then(|&n| e.node_ring(n));
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
                for i in 0..self.facts.len() as u32 {
                    if self.facts[i as usize].topic == *topic {
                        put(&mut charge, &mut say, self.fact_node(i), 1.0, None);
                    }
                }
                origin = Some(self.home(topic));
                take = 1;
            }
            Ask::Tell { key, value } => {
                let f = self.turn_fact(Rel::Memory, &format!("you:{key}"), key, vec![value.clone()], msg.trim());
                origin = Some(self.home(&f.topic));
                let id = self.brand(f);
                put(&mut charge, &mut say, self.fact_node(id), 1.0, None);
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
                    put(&mut charge, &mut say, self.fact_node(i), (1.0 - 0.02 * age).max(0.3), None);
                }
                if ids.is_empty() {
                    match key {
                        Some(k) => before.push(format!("I don't know your {k}. You haven't told me.")),
                        None => before.push("You haven't told me anything about yourself yet.".into()),
                    }
                    reading = false;
                }
                origin = Some(self.home(&key.as_ref().map(|k| format!("you:{k}")).unwrap_or("you:said".into())));
                take = if key.is_some() { 1 } else { 3 };
            }
            Ask::RecallSaid => {
                let ids: Vec<u32> =
                    (0..self.facts.len() as u32).filter(|&i| i != said_id && self.facts[i as usize].rel == Rel::Said).collect();
                for &i in &ids {
                    let age = self.turn.saturating_sub(self.facts[i as usize].strike.turn) as f32;
                    put(&mut charge, &mut say, self.fact_node(i), (1.0 - 0.1 * age).max(0.2), None);
                }
                if ids.is_empty() {
                    before.push("You haven't said anything before this.".into());
                    reading = false;
                }
                origin = Some(self.home("you:said"));
                take = 2;
            }
            Ask::IsA { x, y } => {
                origin = topic_ring(self, x);
                match knowledge::chain(&self.wn, (x.pos, x.offset), &y.lemma) {
                    Some(path) => {
                        // the chain is said from Y's node, beside which it is branded
                        let f = knowledge::chain_fact(&self.wn, &path, &x.lemma, &y.lemma, self.turn, &y.lemma);
                        let id = self.brand(f);
                        put(&mut charge, &mut say, self.fact_node(id), 1.0, None);
                    }
                    None => {
                        let yn = self.np(y);
                        before.push(format!("I have nothing that says {} is {}. What I do know:", self.np(x), yn));
                        for (n, f) in self.topic_facts(x, &x.lemma) {
                            if f.rel == Rel::IsA || f.rel == Rel::InstanceOf {
                                put(&mut charge, &mut say, n, 1.0, Some(f));
                            }
                        }
                    }
                }
                take = 1;
            }
            Ask::RelOf { class, y } => {
                origin = topic_ring(self, y);
                match knowledge::rel_of(&self.wn, (y.pos, y.offset), class) {
                    Some(found) => {
                        let f = knowledge::rel_of_fact(&self.wn, (y.pos, y.offset), found, class, &y.lemma, self.turn, class);
                        let id = self.brand(f);
                        put(&mut charge, &mut say, self.fact_node(id), 1.0, None);
                    }
                    None => {
                        before.push(format!("I don't know the {} of {}. What I do know:", crate::wordnet::show(class), self.np(y)));
                        for (n, f) in self.topic_facts(y, &y.lemma) {
                            let c = w(f.rel);
                            put(&mut charge, &mut say, n, c, Some(f));
                        }
                    }
                }
                take = 1;
            }
            Ask::Opposite { x } => {
                origin = topic_ring(self, x);
                let facts = self.topic_facts(x, &x.lemma);
                let any = facts.iter().any(|(_, f)| f.rel == Rel::Antonym);
                if !any {
                    before.push(format!("I don't know an opposite of {}. What I do know:", crate::wordnet::show(&x.lemma)));
                }
                for (n, f) in facts {
                    if (any && f.rel == Rel::Antonym) || (!any && f.rel == Rel::Def) {
                        put(&mut charge, &mut say, n, 1.0, Some(f));
                    }
                }
                take = 1;
            }
            Ask::Care => {
                before.push(ethics::CARE.into());
                reading = false;
            }
            Ask::Refuse { words } => {
                // a firm no, in the moral core's own words
                before.push(ethics::REFUSE.into());
                let mut hit = ethics::touched(words);
                if hit.is_empty() {
                    hit = vec![0];
                }
                for i in hit {
                    let id = self.keys.get(&ethic_facts()[i].key()).cloned();
                    if let Some(id) = id {
                        put(&mut charge, &mut say, self.fact_node(id), 1.0 - 0.01 * i as f32, None);
                    }
                }
                origin = Some(self.home("ethic:0"));
                take = 2;
            }
            Ask::Moral { words, acts } => {
                let hit = ethics::touched(words);
                for &i in &hit {
                    if let Some(&id) = self.keys.get(&ethic_facts()[i].key()) {
                        put(&mut charge, &mut say, self.fact_node(id), 1.0, None);
                    }
                }
                for a in acts {
                    for (n, c, f) in self.moral_items(&a.lemma) {
                        put(&mut charge, &mut say, n, c, Some(f));
                    }
                }
                if hit.is_empty() {
                    before.push("My moral core doesn't settle that.".into());
                    if charge.is_empty() {
                        reading = false;
                    } else {
                        before.push("What the books I hold say:".into());
                    }
                }
                origin = acts.first().and_then(|a| topic_ring(self, a)).or(Some(self.home("ethic:0")));
                take = if hit.is_empty() { 2 } else { 2.min(hit.len() + 1) };
            }
            Ask::Opinion => {
                before.push("I can't judge that. All I hold is what the word network and you have told me, and none of it is an opinion. Ask me what one of them is.".into());
                reading = false;
            }
            Ask::Now => {
                before.push("I don't know what is happening now. I hold no news, no clock and no weather, only the word network and our talk.".into());
                reading = false;
            }
            Ask::Something => {
                // a handful of branded senses, drawn by the seed; the wave says one of them
                let mut rng = Rng::new(seed ^ 0x50_4D_45);
                let mut picked = 0;
                while picked < 8 {
                    let key = self.nodes[(rng.next() % self.nodes.len() as u64) as usize];
                    let Some(syn) = self.wn.get(key) else { continue };
                    if key.0 != Pos::N || syn.words[0].contains('_') {
                        continue;
                    }
                    let t = Topic { lemma: syn.words[0].to_lowercase(), pos: key.0, offset: key.1 };
                    for (n, f) in self.topic_facts(&t, &t.lemma) {
                        if f.rel == Rel::Def {
                            if origin.is_none() {
                                origin = self.node_ring(n);
                            }
                            put(&mut charge, &mut say, n, 0.6 + 0.05 * picked as f32, Some(f));
                        }
                    }
                    picked += 1;
                }
                take = 1;
            }
            Ask::Unknown { words } => {
                before.push(format!("I don't know what \"{}\" is. Nothing in the word network or our talk holds it.", words.join(" ")));
                reading = false;
            }
            Ask::About { q, topics, unknown } => {
                origin = topic_ring(self, &topics[0]);
                if !unknown.is_empty() {
                    before.push(format!("I don't know \"{}\".", unknown.join("\", \"")));
                }
                let mut gap: Option<String> = None;
                for t in topics {
                    let lexfile = self.wn.get((t.pos, t.offset)).map(|s| s.lexfile).unwrap_or(0);
                    // what people have thought about it: the philosophers, for an abstract word
                    let abstract_ = matches!(lexfile, 4 | 7 | 9 | 12 | 16 | 26);
                    for (n, c, f) in self.item_charges(&t.lemma, *q, abstract_) {
                        put(&mut charge, &mut say, n, c, Some(f));
                    }
                    let facts = self.topic_facts(t, &t.lemma);
                    let rels: Vec<Rel> = facts.iter().map(|(_, f)| f.rel).collect();
                    let lexfile = self.wn.get((t.pos, t.offset)).map(|s| s.lexfile).unwrap_or(0);
                    for (n, f) in facts {
                        let c = match q {
                            QWord::Where => match f.rel {
                                Rel::PartOf => 1.0,
                                Rel::Def => 0.85,
                                _ => 0.0,
                            },
                            QWord::Who => match f.rel {
                                Rel::Def => 1.0,
                                Rel::InstanceOf | Rel::IsA => 0.9,
                                _ => 0.4,
                            },
                            _ => w(f.rel),
                        };
                        if c > 0.0 {
                            put(&mut charge, &mut say, n, c, Some(f));
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
                take = (3 * topics.len()).min(5);
                if let Some(g) = gap {
                    before.push(format!("{g} What I do know:"));
                    take = topics.len().min(2);
                }
            }
        }

        for p in self.block.parts.iter_mut() {
            p.charge = p.fact.and_then(|n| charge.get(&n).cloned()).unwrap_or(0.0);
        }
        let snapshot = self.save_snapshot(&say)?;
        let origin = origin.unwrap_or_else(|| self.home("you:said"));
        let mut b = std::mem::replace(&mut self.block, Block::new(Cfg { nx: 2, ny: 2, nz: 1, ..Default::default() }));
        b.stir(seed);
        b.charge(origin);
        // the train runs until its last front passes the far end; while fewer charged
        // rings have peaked than the answer reads, more fronts follow (up to twice as many)
        let fronts = b.cfg.wave_fronts;
        let want = if reading { take.min(charge.len()) } else { 0 };
        let mut steps = b.wave_steps();
        let mut viz = viz;
        let mut t = 0;
        loop {
            while t < steps {
                b.step(true);
                t += 1;
                if let Some(v) = viz.as_mut() {
                    v(&b, t, steps);
                }
            }
            let up = (0..b.rings()).filter(|&r| b.limit[r] > 1.0 && b.peak[r] >= 0).count();
            if up >= want || b.cfg.wave_fronts >= 2 * fronts {
                break;
            }
            b.cfg.wave_fronts += 4;
            steps = b.wave_steps();
        }
        b.cfg.wave_fronts = fronts;

        // the last charged rings to peak, in the order they peaked
        let pp = b.cfg.particles_per_ring;
        let s = self.nodes.len() as u32;
        let talk = self.talk_base();
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
                // a ring says what its most charged particle holds
                let mut held: Vec<(usize, f32)> =
                    (0..pp).map(|q| r * pp + q).filter(|&j| b.parts[j].charge > 0.0).map(|j| (j, b.parts[j].charge)).collect();
                held.sort_by(|a, c| c.1.partial_cmp(&a.1).unwrap());
                held.truncate(1);
                let mut pr = PeakRead { ring: b.coords(r), peak_step: b.peak[r], limit: b.limit[r], facts: vec![] };
                let mut any = false;
                for (j, c) in held {
                    let n = b.parts[j].fact.unwrap();
                    let f = if n < talk { say.get(&n) } else { self.facts.get((n - talk) as usize) };
                    let Some(f) = f else { continue };
                    let ok = match &f.strike.source {
                        // a store item is said only as the store holds it, at the row its strike names
                        Source::Store { row } => {
                            n == s + row && self.store.items.get(*row as usize).map(|it| f.object.first() == Some(&it.text)).unwrap_or(false)
                        }
                        _ => verify(&self.wn, f),
                    };
                    // the wave picks the wording: the particle's direction against its ring's spin
                    let v = ((b.parts[j].dir * b.omega[r].signum()) < 0.0) as u8;
                    let text = phrase(&self.wn, f, v);
                    let node = if n < s {
                        format!("WordNet node {:?}", self.nodes[n as usize])
                    } else if n < talk {
                        let it = &self.store.items[(n - s) as usize];
                        format!("{} · {} · {}", it.who, it.work, it.locator)
                    } else {
                        format!("talk fact {}", n - talk)
                    };
                    pr.facts.push((n, c, format!("{node} · {:?}", f.strike), text.clone(), ok));
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
        if self.persist {
            self.save_brand()?;
        }
        self.log(&reply)?;
        self.last = Some(reply.clone());
        Ok(reply)
    }

    /// What a replay needs: the talk's branding as it was before this wave, and what the
    /// charged WordNet nodes were to say (WordNet's own branding is in brand/).
    fn save_snapshot(&self, say: &HashMap<u32, Fact>) -> Result<String> {
        std::fs::create_dir_all(&self.dir)?;
        let s = self.talk_base();
        let branded: Vec<(u32, u32)> =
            self.block.parts.iter().enumerate().filter_map(|(i, p)| p.fact.filter(|&n| n >= s).map(|n| (i as u32, n))).collect();
        let snap = Snapshot { turn: self.turn - 1, facts: self.facts.clone(), branded, say: say.iter().map(|(k, v)| (*k, v.clone())).collect() };
        let name = format!("snap-{:04}.json", self.turn);
        std::fs::write(self.dir.join(&name), serde_json::to_vec(&snap)?)?;
        Ok(name)
    }

    /// Put the talk's branding back as it was before a reply's wave (for replay).
    pub fn load_snapshot(&mut self, snap: Snapshot) {
        let s = self.talk_base();
        self.facts = snap.facts;
        self.keys = self.facts.iter().enumerate().map(|(i, f)| (f.key(), i as u32)).collect();
        for p in self.block.parts.iter_mut() {
            if p.fact.map(|n| n >= s).unwrap_or(false) {
                p.fact = None;
            }
        }
        self.slot.retain(|&n, _| n < s);
        for (p, n) in snap.branded {
            self.block.parts[p as usize].fact = Some(n);
            self.slot.insert(n, p as usize);
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
