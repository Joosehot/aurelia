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
use crate::infer;
use crate::think;
use crate::store::{self, Index, Role, Src, Store};
use crate::talk;
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
    /// is X a Y; `why` asks for the steps ("why is a penguin a bird")
    IsA { x: Topic, y: Topic, why: bool },
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
    /// tell me more: the last question again, wider
    More,
    /// "I am Joose, your creator"
    Introduce { name: Option<String>, role: Option<String> },
    /// "do you know who I am?"
    WhoAmI,
    /// "I feel a bit lonely"
    Feeling { word: String, lemma: String },
    /// "I just kissed you"
    ActOnMe { verb: String, said: String },
    /// "do you need a kiss?"
    AboutMe { verb: String, x: Option<Topic>, word: String },
    /// "are you a robot?"
    AreYou { word: String, x: Option<Topic> },
    /// "what do you want to talk about?"
    Topic,
    /// "want to know what happened today?"
    Offer,
    /// "today I went to the market"
    Told { clause: String, when: String },
    /// a social move no act or knowledge answers: a human reply from the dialog corpus
    Social { keys: Vec<String> },
    /// which is better, X or Y: no ranking, a comparison
    Rank { things: Vec<Topic> },
    Nothing,
}

/// What Aurelia reasons from a node if its ring peaks.
#[derive(Clone, Debug)]
pub enum Plan {
    /// a thing is its familiar kind, through the chain of kinds
    Identity { thing: Topic, path: Vec<infer::Key> },
    /// what is said of a kind holds for the thing
    Inherit { thing: Topic, path: Vec<infer::Key>, prem: infer::Premise },
    /// a place is in what its places are in
    Place { thing: Topic, chain: Vec<infer::Key> },
    /// two things meet at their nearest common kind; `mem` is what the user said of the first
    Compare { a: Topic, b: Topic, pa: Vec<infer::Key>, pb: Vec<infer::Key>, mem: Option<u32> },
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
    /// the thinking's memory of this talk: what was said, when the asker was last linked to,
    /// and the last question about something (for "tell me more")
    #[serde(default)]
    pub said_before: Vec<String>,
    #[serde(default)]
    pub last_link_turn: u32,
    #[serde(default)]
    pub last_about: Option<String>,
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
    /// the last question about something, for "tell me more"
    last_about: Option<Ask>,
    last_about_msg: Option<String>,
    /// what was said in this talk, not to be said again
    said_before: std::collections::HashSet<String>,
    /// the turn a reply last reached for what the user told about themselves
    last_link_turn: u32,
    pub last: Option<Reply>,
}

#[derive(Serialize, Deserialize)]
struct Memory {
    turn: u32,
    facts: Vec<Fact>,
    /// how many knowledge nodes the block was branded with; another count brands it anew, keeping the talk
    #[serde(default)]
    knowledge: usize,
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
    ("you're", &["you", "are"]),
    ("youre", &["you", "are"]),
    ("you've", &["you", "have"]),
    ("you'll", &["you", "will"]),
    ("we're", &["we", "are"]),
    ("they're", &["they", "are"]),
    ("i've", &["i", "have"]),
    ("ive", &["i", "have"]),
    ("i'll", &["i", "will"]),
    ("i'd", &["i", "would"]),
    ("can't", &["can", "not"]),
    ("cant", &["can", "not"]),
    ("won't", &["will", "not"]),
    ("wont", &["will", "not"]),
    ("aren't", &["are", "not"]),
    ("wasn't", &["was", "not"]),
    ("didn't", &["did", "not"]),
    ("didnt", &["did", "not"]),
    ("let's", &["let", "us"]),
    ("there's", &["there", "is"]),
    ("that's", &["that", "is"]),
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
            last_about: None,
            last_about_msg: None,
            said_before: Default::default(),
            last_link_turn: 0,
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
        let old: Option<Memory> = std::fs::read(e.brand_dir.join("memory.json")).ok().and_then(|b| serde_json::from_slice(&b).ok());
        let same = parts_file.exists()
            && std::fs::metadata(&parts_file).map(|m| m.len() as usize == e.block.parts.len() * 4).unwrap_or(false)
            && old.as_ref().map(|m| m.knowledge == knowledge).unwrap_or(false);
        let mut keep: Vec<Fact> = Vec::new();
        let mut keep_turn = 0;
        if !same {
            if let Some(m) = old {
                keep = m.facts;
                keep_turn = m.turn;
            }
        }
        if same {
            let bytes = std::fs::read(&parts_file)?;
            let mem: Memory = serde_json::from_slice(&std::fs::read(e.brand_dir.join("memory.json"))?)?;
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
            // what Aurelia holds of herself and the moral core: what is missing is branded, once
            for f in knowledge::self_facts().into_iter().chain(ethic_facts()) {
                e.brand(f);
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
                        Role::DialogPrompt => 6,
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
            // what the talk branded on the block before stays branded on the new one
            for f in keep.drain(..) {
                e.brand(f);
            }
            e.turn = keep_turn;
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
        let knowledge = self.nodes.len() + self.store.items.len();
        std::fs::write(self.brand_dir.join("memory.json"), serde_json::to_vec(&Memory { turn: self.turn, facts: self.facts.clone(), knowledge })?)?;
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
                Src::Dialog => Rel::Dialog,
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
                (_, Role::BookMentions | Role::DialogPrompt) => 0.0,
            };
            if c > 0.0 {
                out.push((i, if short { c } else { c - 0.05 }));
            }
        }
        out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
        out.truncate(400);
        out.into_iter().map(|(i, c)| (s + i, c, self.item_fact(i, lemma))).collect()
    }

    /// What can be reasoned about a thing: each premise that could peak, and what follows from it.
    fn plan(&self, t: &Topic, q: QWord, plans: &mut HashMap<u32, Plan>, charge: &mut HashMap<u32, f32>) {
        let key = (t.pos, t.offset);
        let mut add = |plans: &mut HashMap<u32, Plan>, node: u32, c: f32, plan: Plan| {
            if c > *charge.get(&node).unwrap_or(&0.0) {
                charge.insert(node, c);
                plans.insert(node, plan);
            }
        };
        if q == QWord::Where {
            let chain = infer::part_chain(&self.wn, key, 4);
            if chain.len() >= 3 {
                if let Some(&node) = self.node_of.get(chain.last().unwrap()) {
                    add(plans, node, 1.0, Plan::Place { thing: t.clone(), chain });
                }
            }
            return;
        }
        let up = infer::ancestors(&self.wn, key, 10);
        if let Some(path) = infer::best_kind(&self.wn, &self.index, &up) {
            if let Some(&node) = self.node_of.get(path.last().unwrap()) {
                add(plans, node, 0.98, Plan::Identity { thing: t.clone(), path });
            }
        }
        let s = self.nodes.len() as u32;
        let mut n = 0;
        'kinds: for path in &up {
            let kind = self.wn.get(*path.last().unwrap()).map(|x| x.words[0].to_lowercase()).unwrap_or_default();
            if infer::TOO_WIDE.contains(&kind.as_str()) {
                continue;
            }
            for prem in infer::premises_of(&self.wn, &self.store, &self.index, &kind, 25) {
                let (denied, known) = infer::checked(&self.wn, &self.store, &self.index, &t.lemma, &prem);
                if denied {
                    continue;
                }
                let surprise = (infer::surprise(&self.wn, path, &prem.pred) - if known { 0.3 } else { 0.0 }).max(0.0);
                let node = s + prem.item;
                let clean = if self.store.items[prem.item as usize].src == Src::Primo { 0.05 } else { 0.0 };
                add(plans, node, 0.5 + 0.35 * surprise + clean, Plan::Inherit { thing: t.clone(), path: path.clone(), prem });
                n += 1;
                if n >= 600 {
                    break 'kinds;
                }
            }
        }
        // what the user likes or has, met with this thing; not in every reply
        let linked_lately = self.last_link_turn > 0 && self.turn < self.last_link_turn + 3;
        for (i, f) in self.facts.iter().enumerate() {
            if linked_lately {
                break;
            }
            if f.rel != Rel::Memory || !matches!(f.subject.as_str(), "like" | "love" | "have") {
                continue;
            }
            let (w, _) = tokens(&f.object[0]);
            let found = self.topics(&w);
            if std::env::var("AURELIA_TRACE").is_ok() {
                eprintln!("memory {:?} -> {:?}", f.object, found);
            }
            let Some(m) = found.0.into_iter().find(|x| x.pos == Pos::N && x.offset != NO_SYNSET) else { continue };
            if m.lemma == t.lemma {
                continue;
            }
            let met = infer::meet(&self.wn, (m.pos, m.offset), key);
            if std::env::var("AURELIA_TRACE").is_ok() {
                eprintln!("meet {:?}", met.as_ref().map(|(a, b)| (a.len(), b.len())));
            }
            if let Some((pm, pt)) = met {
                if pm.len() <= 6 && pt.len() <= 6 {
                    add(plans, self.fact_node(i as u32), 0.97, Plan::Compare { a: m, b: t.clone(), pa: pm, pb: pt, mem: Some(i as u32) });
                }
            }
        }
    }

    fn chain_ok(&self, path: &[infer::Key], sym: &[&str]) -> bool {
        path.windows(2).all(|w| self.wn.get(w[0]).map(|s| s.ptrs.iter().any(|p| sym.contains(&p.sym.as_str()) && p.target == w[1])).unwrap_or(false))
    }

    /// The sentence a plan comes to, the steps it stands on, and whether every step checks.
    #[allow(clippy::type_complexity)]
    fn reason(&self, plan: &Plan, v: u8) -> Option<(String, String, bool, think::Kind, f32, usize, String)> {
        let kinds = ["@", "@i"];
        match plan {
            Plan::Identity { thing, path } => {
                let kind = infer::word(&self.wn, *path.last()?);
                let text = format!("{} is {}{}.", infer::cap(&self.np(thing)), infer::np(&kind), infer::by_way(&self.wn, path));
                Some((text, format!("transitivity of kinds: {path:?}"), self.chain_ok(path, &kinds), think::Kind::Identity, 0.0, 99, "identity".into()))
            }
            Plan::Inherit { thing, path, prem } => {
                let it = self.store.items.get(prem.item as usize)?;
                let again = infer::premise(&self.wn, it, prem.item).map(|p| p.pred == prem.pred && p.head == prem.head).unwrap_or(false);
                let ok = again && self.chain_ok(path, &kinds);
                let me = self.np(thing);
                let kind = infer::np(&infer::word(&self.wn, *path.last()?));
                let verb = infer::singular(&prem.verb.to_lowercase(), prem.plural);
                let said = it.text.trim_end_matches('.').to_string();
                let said_lc = {
                    let mut c = said.chars();
                    match c.next() {
                        Some(f) => f.to_lowercase().collect::<String>() + c.as_str(),
                        None => String::new(),
                    }
                };
                let text = if v == 0 {
                    format!("{} {} {}, because {} and {} is {}.", infer::cap(&me), verb, prem.pred, said_lc, me, kind)
                } else {
                    format!("{}, and {} is {}, so {} {} {} too.", said, me, kind, me, verb, prem.pred)
                };
                let steps = format!("premise {} · {} · {} (\"{}\"); kinds {path:?}", it.who, it.work, it.locator, it.text);
                let kind_lemma = self.wn.get(*path.last()?).map(|s| s.words[0].to_lowercase()).unwrap_or_default();
                let surprise = infer::surprise(&self.wn, path, &prem.pred);
                let familiar = infer::familiar(&self.index, &kind_lemma);
                let key = format!("{verb} {}", prem.pred).to_lowercase();
                Some((text, steps, ok, think::Kind::Inference, surprise, familiar, key))
            }
            Plan::Place { thing, chain } => {
                let names: Vec<String> = chain.iter().map(|k| infer::word(&self.wn, *k)).collect();
                let mut text = format!("{} is in {}, because {} is in {}", infer::cap(&self.np(thing)), names.last()?, names[0], names[1]);
                for w in names[1..].windows(2) {
                    text.push_str(&format!(" and {} is in {}", w[0], w[1]));
                }
                text.push('.');
                Some((text, format!("transitivity of places: {chain:?}"), self.chain_ok(chain, &["#p"]), think::Kind::Link, 0.0, 99, "place".into()))
            }
            Plan::Compare { a, b, pa, pb, mem } => {
                let kind = knowledge::plural(&self.wn, &infer::word(&self.wn, *pa.last()?));
                let how = |t: &Topic, p: &[infer::Key]| -> String {
                    if p.len() > 2 {
                        format!("{} is {}", self.np(t), infer::np(&infer::word(&self.wn, p[1])))
                    } else {
                        format!("{} is one directly", self.np(t))
                    }
                };
                let mut ok = self.chain_ok(pa, &kinds) && self.chain_ok(pb, &kinds);
                let core = if pa.len() <= 2 && pb.len() <= 2 {
                    format!("{} and {} are both {}.", self.np(a), self.np(b), kind)
                } else {
                    format!("{} and {} are both {}: {}, {}.", self.np(a), self.np(b), kind, how(a, pa), how(b, pb))
                };
                let text = match mem {
                    Some(i) => {
                        let f = self.facts.get(*i as usize)?;
                        ok &= verify(&self.wn, f);
                        let name = self.facts.iter().rev().find(|x| x.rel == Rel::Memory && x.subject == "name").map(|x| x.object[0].clone());
                        match name {
                            Some(n) => format!("You {} {}, {}, and {}", f.subject, f.object[0], n, core),
                            None => format!("You {} {}, and {}", f.subject, f.object[0], core),
                        }
                    }
                    None => infer::cap(&core),
                };
                Some((text, format!("nearest common kind: {pa:?} / {pb:?}"), ok, think::Kind::Link, 0.0, 99, format!("compare {}", self.np(a))))
            }
        }
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

    /// Which sense a word is in: the one whose kinds the world knowledge uses when it
    /// speaks of the word ("whale" -> the mammal, not "a very large person"); the
    /// most common sense when it says nothing.
    fn sense(&self, lemma: &str, pos: Pos) -> Option<u32> {
        let senses = self.wn.senses(lemma, pos);
        if senses.len() < 2 || pos != Pos::N {
            return senses.first().cloned();
        }
        let texts: Vec<String> = self
            .index
            .get(lemma)
            .map(|e| {
                e.iter()
                    .filter(|(_, r)| matches!(r, Role::Article | Role::InArticle))
                    .take(200)
                    .map(|(i, _)| self.store.items[*i as usize].text.to_lowercase())
                    .collect()
            })
            .unwrap_or_default();
        if texts.is_empty() {
            return senses.first().cloned();
        }
        let mut best = (0usize, senses[0]);
        for (rank, &off) in senses.iter().enumerate() {
            // the sense's kinds, its own kinds below it, and the words of its gloss
            let mut cues: std::collections::HashSet<String> = std::collections::HashSet::new();
            for path in infer::ancestors(&self.wn, (pos, off), 4) {
                if let Some(s) = self.wn.get(*path.last().unwrap()) {
                    cues.extend(s.words.iter().map(|w| w.to_lowercase().replace('_', " ")));
                }
            }
            let mut below = vec![(pos, off)];
            for _ in 0..3 {
                let mut next = Vec::new();
                for k in &below {
                    if let Some(s) = self.wn.get(*k) {
                        for p in s.ptrs.iter().filter(|p| p.sym == "~" || p.sym == "~i") {
                            if let Some(t) = self.wn.get(p.target) {
                                cues.extend(t.words.iter().map(|w| w.to_lowercase().replace('_', " ")));
                            }
                            next.push(p.target);
                        }
                    }
                }
                below = next;
            }
            cues.remove(lemma);
            let mut hits = 2 * cues.iter().filter(|w| texts.iter().any(|t| t.contains(w.as_str()))).count();
            // the gloss's words count, but less than the kinds: they share words with every sense's article
            if let Some(s) = self.wn.get((pos, off)) {
                let gloss: std::collections::HashSet<String> =
                    s.gloss.split(|c: char| !c.is_alphabetic()).filter(|w| w.len() >= 5).map(|w| w.to_lowercase()).filter(|w| w != lemma).collect();
                hits += gloss.iter().filter(|w| texts.iter().any(|t| t.contains(w.as_str()))).count();
            }
            // a later sense must show clearly more to win over a more common one
            if hits > best.0 + 2 * rank.min(1) {
                best = (hits, off);
            }
        }
        Some(best.1)
    }

    fn look_as(&self, words: &[String], order: &[Pos]) -> Option<Topic> {
        if words.is_empty() {
            return None;
        }
        let phrase = words.join("_");
        for &pos in order {
            if let Some(lemma) = self.wn.base(&phrase, pos) {
                if let Some(offset) = self.sense(&lemma, pos) {
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
        // a phrase never runs over a left-out word: "cats or dogs" is two things, not "Cats & Dogs"
        let mut content: Vec<String> = Vec::new();
        for w in words {
            if STOP.contains(&w.as_str()) {
                if content.last().map(|x| x != "|").unwrap_or(false) {
                    content.push("|".into());
                }
            } else {
                content.push(w.clone());
            }
        }
        while content.last().map(|x| x == "|").unwrap_or(false) {
            content.pop();
        }
        // "when did Rome fall": after do/does/did the question ends on its verb
        let verb_last = words.len() > 2 && matches!(words[1].as_str(), "do" | "does" | "did");
        let mut topics = Vec::new();
        let mut unknown = Vec::new();
        let mut i = 0;
        while i < content.len() {
            if content[i] == "|" {
                i += 1;
                continue;
            }
            let mut took = 0;
            for len in (1..=3.min(content.len() - i)).rev() {
                if content[i..i + len].iter().any(|x| x == "|") {
                    continue;
                }
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
        // a greeting is read before the talk's fillers are taken away ("hey there")
        if all.len() <= 3 && all.first().map(|w| matches!(w.as_str(), "hello" | "hi" | "hey" | "greetings" | "moi" | "terve" | "yo" | "howdy")).unwrap_or(false) {
            return Ask::Greet;
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
        let whole = tokens(msg).0.join(" ");
        // asked about herself by name
        if ["is aurelia", "are you aurelia", "you are aurelia", "about aurelia", "aurelia who are you", "aurelia what are you"].iter().any(|p| whole.contains(p)) {
            return Ask::Myself("self:who");
        }
        // what the message does, before what it is about
        if let Some(act) = self.talk_act(msg) {
            return act;
        }
        let mut named = false;
        for (s, q) in sentences {
            let (mut w, mut o) = tokens(&s);
            // her name said to her is not what the message is about ("Ok Aurelia, can we chat?")
            let before = w.len();
            let keep: Vec<bool> = w.iter().map(|x| x != "aurelia").collect();
            let mut k = keep.iter();
            w.retain(|_| *k.next().unwrap());
            let mut k = keep.iter();
            o.retain(|_| *k.next().unwrap());
            named |= w.len() < before;
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
            // only the talk's own words ("haha", "ok then"): answered as talk; her name alone is a greeting
            return if tokens(msg).0.is_empty() && !named {
                Ask::Nothing
            } else if all.iter().all(|x| x == "aurelia") {
                Ask::Greet
            } else {
                Ask::Social { keys: all.clone() }
            };
        };
        let words = pick.0.clone();
        let ask = self.read_words(pick.0, pick.1, pick.2);
        // what neither an act of talk nor the knowledge answers, said socially, gets a human reply
        match ask {
            Ask::About { .. } | Ask::Unknown { .. } | Ask::Nothing if self.social(msg, &words, &all) => {
                Ask::Social { keys: all.iter().map(|x| [Pos::N, Pos::V, Pos::A].iter().find_map(|&p| self.wn.base(x, p)).unwrap_or(x.clone())).collect() }
            }
            a => a,
        }
    }

    /// Is the message a social move rather than a question about the world?
    fn social(&self, msg: &str, words: &[String], all: &[String]) -> bool {
        const START: &[&str] = &[
            "wow", "haha", "lol", "cool", "nice", "great", "awesome", "thanks", "thank", "yes", "no", "nope", "yeah", "sure", "sorry",
            "oh", "aww", "hmm", "really", "but", "so", "ok", "okay", "lmao", "omg", "same", "true", "agreed", "interesting", "amazing",
            "good", "fine", "well", "hehe", "yay", "ugh", "that", "this", "it", "thats", "lovely", "sweet", "perfect", "exactly",
        ];
        let first = words.first().map(|x| x.as_str()).unwrap_or("");
        if qword(first).is_some() || matches!(first, "is" | "are" | "does" | "do" | "did" | "can" | "was" | "were" | "define" | "tell") {
            return false;
        }
        let personal = all.iter().any(|x| matches!(x.as_str(), "you" | "your" | "we" | "us" | "me" | "yourself"));
        let first_all = all.first().map(|x| x.as_str()).unwrap_or("");
        personal || msg.trim_end().ends_with('!') || START.contains(&first_all)
    }

    /// The act of talk a message is, if it is one (src/talk.rs).
    fn talk_act(&self, msg: &str) -> Option<Ask> {
        let (lw, lo) = tokens(msg);
        let mut w: Vec<String> = Vec::new();
        let mut o: Vec<String> = Vec::new();
        for (x, y) in lw.into_iter().zip(lo) {
            if x != "aurelia" {
                w.push(x);
                o.push(y);
            }
        }
        let said: Vec<String> = w.clone();
        while w.first().map(|x| matches!(x.as_str(), "oh" | "ok" | "okay" | "well" | "so" | "um" | "alright" | "yeah" | "wow")).unwrap_or(false) {
            w.remove(0);
            o.remove(0);
        }
        if w.is_empty() {
            // "wow", "ok": only the talk's own words, answered as talk
            return if said.is_empty() { None } else { Some(Ask::Social { keys: said }) };
        }
        let s = w.join(" ");
        const GREET: &[&str] = &["hello", "hi", "hey", "greetings", "moi", "terve", "yo", "howdy", "hiya"];
        // "Hey! I just kissed you!": past the greeting to what the message does
        if GREET.contains(&w[0].as_str()) && w.len() > 2 {
            let rest = msg.splitn(2, |c: char| matches!(c, '!' | '.' | ',' | '?')).nth(1).unwrap_or("").trim().to_string();
            if !rest.is_empty() && rest.split_whitespace().count() >= 2 {
                if let Some(act) = self.talk_act(&rest) {
                    return Some(act);
                }
            }
        }
        if w.len() <= 5 && w.iter().take(2).any(|x| GREET.contains(&x.as_str())) && !s.contains("you") {
            return Some(Ask::Greet);
        }
        const TALK: &[&str] = &["chat", "talk", "speak", "converse"];
        let invite = ["let us", "can we", "could we", "shall we", "wanna", "want to", "do you want to", "can i talk", "can you chat", "can you talk", "lets"];
        if invite.iter().any(|p| s.starts_with(p)) && w.iter().any(|x| TALK.contains(&x.as_str())) {
            return Some(Ask::Myself("self:chat"));
        }
        if ["who am i", "who i am", "do you know me", "do you remember me", "know my name", "remember my name"].iter().any(|p| s.contains(p)) {
            return Some(Ask::WhoAmI);
        }
        // "what do you want to talk about?": she has no wants, but she can offer what the asker gave her
        let pick = ["want to talk about", "wanna talk about", "like to talk about", "should we talk about", "shall we talk about", "do we talk about", "to talk about", "pick a topic", "you choose", "you pick"];
        if pick.iter().any(|p| s.contains(p)) && (s.starts_with("what") || s.starts_with("you") || s.starts_with("pick")) {
            return Some(Ask::Topic);
        }
        // "want to know what happened today?", "guess what", "let me tell you": an offer to tell her
        let offer = ["want to know", "want to hear", "wanna know", "wanna hear", "can i tell you", "let me tell you", "guess what", "may i tell you", "shall i tell you", "you know what"];
        if offer.iter().any(|p| s.starts_with(p) || s.starts_with(&format!("do you {p}")) || s.starts_with(&format!("would you {p}")) || s.contains(&format!("you {p}"))) {
            return Some(Ask::Offer);
        }
        // "today I went to the market": something told, to be remembered
        if !msg.trim_end().ends_with('?') {
            let start = w.iter().position(|x| x == "i").filter(|&k| k == 0 || w[..k].iter().all(|x| matches!(x.as_str(), "today" | "yesterday" | "so" | "well" | "and" | "this" | "morning" | "tonight" | "earlier" | "then" | "last" | "night" | "week")));
            if let Some(k) = start {
                if let Some(v) = w[k + 1..].iter().find(|x| !talk::SOFT.contains(&x.as_str())) {
                    // "saw" is "see" before it is "to saw": the irregular forms first
                    let irregular = self.wn.exc.get(&(Pos::V, v.clone())).map(|b| b != v).unwrap_or(false);
                    let past = irregular || self.wn.base(v, Pos::V).map(|b| &b != v && v.ends_with("ed")).unwrap_or(false);
                    if past && !w.contains(&"you".to_string()) {
                        let clause = o[k..].join(" ");
                        return Some(Ask::Told { clause, when: w[..k].join(" ") });
                    }
                }
            }
        }
        // "need another kiss?", "want a cookie?": the question about her, said short
        if matches!(w[0].as_str(), "need" | "want" | "wanna" | "like") && w.len() >= 2 {
            let verb = if w[0] == "wanna" { "want".to_string() } else { w[0].clone() };
            let rest: Vec<String> = w[1..].iter().filter(|x| !matches!(x.as_str(), "a" | "an" | "the" | "another" | "some" | "more" | "one")).cloned().collect();
            let x = self.look(&rest, true);
            return Some(Ask::AboutMe { verb, x, word: rest.join(" ") });
        }
        // "I am Joose your creator", "I'm sad", "I feel a bit lonely today"
        let after = if s.starts_with("i am feeling") || s.starts_with("i have been feeling") {
            Some(if s.starts_with("i am") { 3 } else { 4 })
        } else if s.starts_with("i am") || s.starts_with("i feel") {
            Some(2)
        } else {
            None
        };
        if let Some(k) = after {
            let rest: Vec<usize> = (k..w.len()).filter(|&i| !talk::DEGREE.contains(&w[i].as_str())).collect();
            let role = w.iter().position(|x| x == "your").and_then(|i| w.get(i + 1)).filter(|r| talk::ROLES.contains(&r.as_str())).cloned();
            let name = rest.iter().find(|&&i| o[i].chars().next().map(|c| c.is_uppercase()).unwrap_or(false) && w[i] != "i").map(|&i| o[i].clone());
            if name.is_some() || role.is_some() {
                return Some(Ask::Introduce { name, role });
            }
            if let Some(&i) = rest.first() {
                let word = &w[i];
                if let Some(a) = self.wn.base(word, Pos::A).or_else(|| self.wn.base(word, Pos::N).filter(|b| self.wn.senses(b, Pos::N).first().and_then(|o| self.wn.get((Pos::N, *o))).map(|s| s.lexfile == 12).unwrap_or(false))) {
                    return Some(Ask::Feeling { word: word.clone(), lemma: a });
                }
            }
        }
        // "I just kissed you", "I love you"
        if w[0] == "i" {
            if let Some(k) = w.iter().position(|x| x == "you") {
                let mid: Vec<&String> = w[1..k].iter().filter(|x| !talk::SOFT.contains(&x.as_str())).collect();
                if mid.len() == 1 {
                    if let Some(v) = self.wn.base(mid[0], Pos::V) {
                        return Some(Ask::ActOnMe { verb: v, said: mid[0].clone() });
                    }
                }
            }
        }
        // "do you need a kiss?", "do you have a body?", "would you like a cookie?"
        if w.len() >= 3 && matches!(w[0].as_str(), "do" | "would" | "will") && w[1] == "you" {
            let mut verb = w[2].clone();
            let mut rest = strip_articles(&w[3..]);
            // "would you like a cookie" is wanting one
            if w[0] == "would" && verb == "like" {
                verb = "want".into();
            }
            // "do you want to sleep": the act wanted
            if matches!(verb.as_str(), "want" | "need" | "like" | "love") && rest.first().map(|x| x == "to").unwrap_or(false) && rest.len() >= 2 {
                verb = rest[1].clone();
                rest = rest[2..].to_vec();
            }
            let x = self.look(&rest, true);
            return Some(Ask::AboutMe { verb, x, word: rest.join(" ") });
        }
        // "are you a robot?", "you're alive?"
        if w.len() >= 3 && ((w[0] == "are" && w[1] == "you") || (w[0] == "you" && w[1] == "are")) {
            let rest = strip_articles(&w[2..]);
            if rest.is_empty() {
                return None;
            }
            // "you are sweet": a word said of her, not a question of what kind she is
            const LIFE_MOOD: &[&str] = &["alive", "living", "human", "conscious", "sentient", "ok", "okay", "fine", "well", "happy", "sad", "tired", "alright", "good", "lonely", "angry", "bored"];
            if rest.len() == 1 && !LIFE_MOOD.contains(&rest[0].as_str()) && self.wn.base(&rest[0], Pos::A).is_some() && !rest[0].ends_with("er") {
                return Some(Ask::Social { keys: w.clone() });
            }
            let x = self.look_as(&rest, &[Pos::N, Pos::A]);
            return Some(Ask::AreYou { word: rest.join(" "), x });
        }
        None
    }

    /// The text of a self note, if it is branded and checks.
    fn note(&self, topic: &str) -> Option<String> {
        self.facts.iter().find(|f| f.rel == Rel::SelfNote && f.topic == topic && verify(&self.wn, f)).map(|f| f.object[0].clone())
    }

    fn remembered(&self, key: &str) -> Option<String> {
        self.facts.iter().rev().find(|f| f.rel == Rel::Memory && f.subject == key).map(|f| f.object[0].clone())
    }

    /// What she answers to an act of talk, reasoned from herself, the speaker and the word network.
    fn answer_act(&mut self, ask: &Ask, msg: &str) -> Option<Vec<String>> {
        let name = self.remembered("name");
        let to = |n: &Option<String>| n.as_ref().map(|x| format!(", {x}")).unwrap_or_default();
        let lacks = |e: &Engine, what: &str| e.note(&format!("self:lacks:{what}")).unwrap_or_default();
        Some(match ask {
            Ask::Greet => match &name {
                Some(n) => vec![format!("Hello, {n}.")],
                None => vec!["Hello. I'm Aurelia.".into(), self.note("self:chat")?.replacen("Yes. ", "", 1)],
            },
            Ask::Introduce { name: who, role } => {
                let mut out = Vec::new();
                if let Some(n) = who {
                    let f = self.turn_fact(Rel::Memory, "you:name", "name", vec![n.clone()], msg.trim());
                    self.brand(f);
                }
                if let Some(r) = role {
                    let f = self.turn_fact(Rel::Memory, "you:role", "role", vec![r.clone()], msg.trim());
                    self.brand(f);
                }
                match (who, role) {
                    (Some(n), Some(r)) => out.push(format!("Hello, {n}. I'll remember that you are my {r}.")),
                    (Some(n), None) => out.push(format!("Hello, {n}. I'll remember your name.")),
                    (None, Some(r)) => out.push(format!("I'll remember that you are my {r}.")),
                    _ => {}
                }
                // what the role means, from the word network: a creator makes things, so you made me
                if let Some(r) = role {
                    if let Some(t) = self.look(&[r.clone()], true) {
                        if let Some(def) = self.topic_facts(&t, &t.lemma).into_iter().find(|(_, f)| f.rel == Rel::Def).map(|(_, f)| f.object[0].clone()) {
                            let makes = ["make", "invent", "creat", "build", "design", "develop", "write", "program"].iter().any(|m| def.contains(m));
                            if makes {
                                out.push(format!("{} is {}, so you made me.", infer::cap(&infer::np(r)), def));
                            }
                        }
                    }
                }
                out
            }
            Ask::WhoAmI => {
                let role = self.remembered("role");
                match (&name, &role) {
                    (None, None) => vec!["I don't know who you are yet. Tell me your name.".into()],
                    _ => {
                        let mut first = match (&name, &role) {
                            (Some(n), Some(r)) => format!("You are {n}, my {r}."),
                            (Some(n), None) => format!("You are {n}."),
                            (None, Some(r)) => format!("You are my {r}."),
                            _ => unreachable!(),
                        };
                        let more: Vec<String> = self
                            .facts
                            .iter()
                            .rev()
                            .filter(|f| f.rel == Rel::Memory && !matches!(f.subject.as_str(), "name" | "role" | "am"))
                            .take(2)
                            .map(|f| phrase(&self.wn, f, 0))
                            .collect();
                        if !more.is_empty() {
                            first.push(' ');
                            first.push_str(&more.join(" "));
                        }
                        vec![first]
                    }
                }
            }
            Ask::Feeling { word, lemma } => {
                let f = self.turn_fact(Rel::Memory, "you:feel", "feel", vec![word.clone()], msg.trim());
                self.brand(f);
                let def = [Pos::A, Pos::N].iter().find_map(|&p| {
                    let off = *self.wn.senses(lemma, p).first()?;
                    let t = Topic { lemma: lemma.clone(), pos: p, offset: off };
                    self.topic_facts(&t, lemma).into_iter().find(|(_, f)| f.rel == Rel::Def).map(|(_, f)| f.object[0].clone())
                });
                let mut out = vec![match (&name, &def) {
                    (Some(n), Some(d)) => format!("{n}, you feel {word}: {d}."),
                    (None, Some(d)) => format!("You feel {word}: {d}."),
                    (Some(n), None) => format!("{n}, you feel {word}."),
                    (None, None) => format!("You feel {word}."),
                }];
                if def.as_deref().map(talk::hard_feeling).unwrap_or(false) {
                    // the moral core: helping someone in need is good; and what she can do
                    if let Some(p) = self.facts.iter().find(|f| f.rel == Rel::Ethic && f.object[0].starts_with("Helping")) {
                        out.push(p.object[0].clone());
                    }
                    out.push(self.note("self:listen")?);
                }
                out
            }
            Ask::ActOnMe { verb, said } => {
                let f = self.turn_fact(Rel::Memory, "you:did", "did", vec![said.clone()], msg.trim());
                self.brand(f);
                let key = self.wn.senses(verb, Pos::V).first().map(|o| (Pos::V, *o));
                if talk::feeling_verb(&self.wn, verb) {
                    vec![format!("You {said} me{}? {} So I can't {verb} you back, but I'll remember that you said so.", to(&name), lacks(self, "feeling"))]
                } else if let Some(k) = key.and_then(|k| talk::bodily(&self.wn, k).map(|_| k)) {
                    let _ = k;
                    let kind = talk::verb_kind(&self.wn, verb).unwrap_or("touch".into());
                    vec![format!("You {said} a block of rings{}. To {verb} is to {kind}, and {} So I couldn't feel it, but I'll remember that you did.", to(&name), lacks(self, "body").to_lowercase().replacen("i ", "I ", 1))]
                } else {
                    vec![format!("You {said} me{}. I'll remember that.", to(&name))]
                }
            }
            Ask::AboutMe { verb, x, word } => {
                let v = self.wn.base(verb, Pos::V).unwrap_or(verb.clone());
                match v.as_str() {
                    "have" => {
                        let lemma = x.as_ref().map(|t| t.lemma.clone()).unwrap_or(word.clone());
                        match talk::has(&lemma) {
                            Some(true) if lemma == "name" => vec!["Yes: my name is Aurelia.".into()],
                            Some(true) if lemma == "ring" || lemma == "particle" => vec![format!(
                                "Yes: {} rings and {} particles.",
                                self.block.rings(),
                                self.block.parts.len()
                            )],
                            Some(true) => vec![self.note("self:know")?],
                            Some(false) => match lemma.as_str() {
                                "body" | "heart" | "face" | "eye" | "hand" | "skin" | "brain" | "mouth" | "lip" => vec![format!("No. {}", lacks(self, "body"))],
                                "feeling" | "emotion" | "soul" => vec![format!("No. {}", lacks(self, "feeling"))],
                                "need" => vec![format!("No. {}", lacks(self, "need"))],
                                _ => vec![format!("No. I have no {word}.")],
                            },
                            None => vec![format!("No. I hold knowledge and what you tell me; I don't have {}.", infer::np(word))],
                        }
                    }
                    "need" | "want" => match x {
                        Some(t) if t.offset != NO_SYNSET => match talk::bodily(&self.wn, (t.pos, t.offset)) {
                            Some(kind) => vec![format!("No. {} is {}, and that takes a body. {}", infer::cap(&self.np(t)), infer::np(&kind), lacks(self, "body"))],
                            None => vec![format!("No. {}", lacks(self, "need"))],
                        },
                        _ => vec![format!("No. {}", lacks(self, "need"))],
                    },
                    "like" | "love" | "hate" | "miss" | "enjoy" | "fear" => {
                        let what = x.as_ref().map(|t| self.np(t)).unwrap_or(word.clone());
                        vec![format!("{} So I can't {v} {what}.", lacks(self, "feeling"))]
                    }
                    "feel" => vec![lacks(self, "feeling")],
                    "eat" | "drink" | "sleep" | "breathe" | "dream" => vec![format!("No. To {v} takes a body. {}", lacks(self, "body"))],
                    // "do you want to kiss me": an act of the body
                    _ if self.wn.senses(&v, Pos::V).first().and_then(|o| talk::bodily(&self.wn, (Pos::V, *o))).is_some() => {
                        let kind = talk::verb_kind(&self.wn, &v).unwrap_or("touch".into());
                        vec![format!("No. To {v} is to {kind}, and that takes a body. {}", lacks(self, "body"))]
                    }
                    // "do you remember what I told you?": what the speaker gave her
                    "know" | "remember" if ["told", "said", "say", "tell", "me", "i"].iter().any(|x| word.split(' ').any(|w| w == *x)) => {
                        let mem: Vec<String> = self
                            .facts
                            .iter()
                            .rev()
                            .filter(|f| f.rel == Rel::Memory && !matches!(f.subject.as_str(), "am"))
                            .take(4)
                            .map(|f| phrase(&self.wn, f, 0))
                            .collect();
                        if mem.is_empty() {
                            vec!["You haven't told me anything about yourself yet.".into()]
                        } else {
                            vec![format!("Yes{}. {}", to(&name), mem.join(" "))]
                        }
                    }
                    "know" | "remember" => vec![self.note("self:know")?],
                    _ => vec![self.note("self:can")?],
                }
            }
            Ask::Offer => vec![format!("Yes{}. {}", to(&name), self.note("self:listen")?)],
            Ask::Told { clause, when } => {
                let f = self.turn_fact(Rel::Memory, "you:told", "told", vec![clause.clone()], msg.trim());
                self.brand(f);
                let back = knowledge::swap_person(clause);
                let when = if when.is_empty() { String::new() } else { format!(" {when}") };
                vec![format!("{}{when}{}. I'll remember that.", infer::cap(&back), to(&name))]
            }
            Ask::Topic => {
                // offer what the asker gave her: what they like, have or love; else their feeling
                let offer = self
                    .facts
                    .iter()
                    .rev()
                    .find(|f| f.rel == Rel::Memory && matches!(f.subject.as_str(), "like" | "love" | "have"))
                    .map(|f| (f.subject.clone(), f.object[0].clone()));
                let mut out = vec![lacks(self, "need")];
                match offer {
                    Some((k, v)) => {
                        let (w, _) = tokens(&v);
                        let thing = self.topics(&w).0.into_iter().find(|t| t.pos == Pos::N).map(|t| knowledge::plural(&self.wn, &crate::wordnet::show(&t.lemma)));
                        let about = thing.unwrap_or(v.clone());
                        out.push(format!("But you told me you {k} {v}{}: we could talk about {about}.", to(&name)));
                    }
                    None => out.push(self.note("self:chat")?.replacen("Yes. ", "", 1)),
                }
                out
            }
            Ask::AreYou { word, x } => {
                const LIFE: &[&str] = &["alive", "living", "human", "person", "conscious", "sentient", "real person", "man", "woman", "girl", "boy"];
                if LIFE.contains(&word.as_str()) {
                    return Some(vec![lacks(self, "life")]);
                }
                const MOOD: &[&str] = &["ok", "okay", "fine", "well", "happy", "sad", "tired", "alright", "good", "lonely", "angry", "bored"];
                if MOOD.contains(&word.as_str()) {
                    return Some(vec![self.note("self:how")?]);
                }
                let me = self.look(&[talk::KIND.to_string()], true)?;
                let mk = (me.pos, me.offset);
                match x {
                    Some(t) if t.pos == Pos::N && t.offset != NO_SYNSET => {
                        let tk = (t.pos, t.offset);
                        let me_up = infer::ancestors(&self.wn, mk, 12);
                        let it_up = infer::ancestors(&self.wn, tk, 12);
                        if me_up.iter().any(|p| *p.last().unwrap() == tk) || mk == tk {
                            vec![format!("Yes. I am a device, and a device is {}.", self.np(t))]
                        } else if it_up.iter().any(|p| *p.last().unwrap() == mk) {
                            vec![format!("Not quite. {} is a kind of device, and so am I, but I am a block of rings, not {}.", infer::cap(&self.np(t)), self.np(t))]
                        } else if it_up.iter().any(|p| infer::word(&self.wn, *p.last().unwrap()) == "organism") {
                            vec![format!("No. {} is an organism. {}", infer::cap(&self.np(t)), lacks(self, "life"))]
                        } else {
                            let parent = it_up.first().map(|p| infer::np(&infer::word(&self.wn, *p.last().unwrap()))).unwrap_or_default();
                            vec![format!("No. I am a device, a block of rings; {} is {}.", self.np(t), parent)]
                        }
                    }
                    _ => vec![self.note("self:who")?],
                }
            }
            _ => return None,
        })
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
        // an invitation to talk is answered, not looked up
        let talk = ["chat", "talk", "speak", "converse"];
        if (s.starts_with("can we") || s.starts_with("could we") || s.starts_with("shall we") || s.starts_with("let's") || s.starts_with("lets")
            || s.starts_with("do you want to") || s.starts_with("want to") || s.starts_with("wanna") || s.starts_with("can i talk") || s.starts_with("can you chat") || s.starts_with("can you talk"))
            && w.iter().any(|x| talk.contains(&x.as_str()))
        {
            return Ask::Myself("self:chat");
        }
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
            // no ranking, but two things can still be compared
            let rest: Vec<String> = w.iter().filter(|x| !OPINION.contains(&x.as_str())).cloned().collect();
            let (topics, _) = self.topics(&rest);
            let things: Vec<Topic> = topics.into_iter().filter(|t| t.pos == Pos::N && t.offset != NO_SYNSET).collect();
            if things.len() >= 2 {
                return Ask::Rank { things };
            }
            return Ask::Opinion;
        }
        if question && (w.iter().any(|x| NOW.contains(&x.as_str())) || s.starts_with("what time") || s.starts_with("what day") || s.starts_with("what date")) {
            return Ask::Now;
        }
        if matches!(s.as_str(), "tell me more" | "more" | "go on" | "and" | "what else" | "tell me more about it" | "anything else" | "keep going") {
            return Ask::More;
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
        // is X a Y, why is X a Y
        let why = w[0] == "why" && w.len() > 1 && matches!(w[1].as_str(), "is" | "are");
        let off = why as usize;
        if matches!(w[off].as_str(), "is" | "are") && w.len() >= 3 + off {
            let rest = &w[1 + off..];
            let mut splits: Vec<usize> = (1..rest.len()).filter(|&k| matches!(rest[k].as_str(), "a" | "an" | "the")).collect();
            splits.extend(1..rest.len());
            for k in splits {
                let (l, r) = (strip_articles(&rest[..k]), strip_articles(&rest[k..]));
                if l.is_empty() || r.is_empty() {
                    continue;
                }
                if let (Some(x), Some(y)) = (self.look(&l, true), self.look(&r, true)) {
                    if x.pos == Pos::N && y.pos == Pos::N {
                        return Ask::IsA { x, y, why };
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
        let mut ask = self.read(msg);
        let mut more = 0;
        if matches!(ask, Ask::More) {
            // "tell me more": the same question again, a wider budget, nothing said twice
            ask = self.last_about.clone().unwrap_or(Ask::Nothing);
            more = 2;
        }
        let about_before = self.last_about_msg.clone();
        if matches!(ask, Ask::About { .. }) {
            self.last_about = Some(ask.clone());
            self.last_about_msg = Some(msg.to_string());
        }
        let said = self.turn_fact(Rel::Said, "you:said", "you", vec![msg.trim().to_string()], msg.trim());
        let said_id = self.brand(said);

        // which branded nodes to charge, what each says, how many rings to read
        let mut charge: HashMap<u32, f32> = HashMap::new();
        let mut say: HashMap<u32, Fact> = HashMap::new();
        // what Aurelia reasons from a charged node, if it peaks
        let mut plans: HashMap<u32, Plan> = HashMap::new();
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
        // an act of talk is answered from herself, the speaker and the word network
        let act_words = match &ask {
            Ask::Greet | Ask::Introduce { .. } | Ask::WhoAmI | Ask::Feeling { .. } | Ask::ActOnMe { .. } | Ask::AboutMe { .. } | Ask::AreYou { .. } | Ask::Topic | Ask::Offer | Ask::Told { .. } => {
                let a = ask.clone();
                self.answer_act(&a, msg)
            }
            _ => None,
        };
        if let Some(words) = act_words {
            before.extend(words);
            reading = false;
        }
        match &ask {
            Ask::Greet | Ask::Introduce { .. } | Ask::WhoAmI | Ask::Feeling { .. } | Ask::ActOnMe { .. } | Ask::AboutMe { .. } | Ask::AreYou { .. } | Ask::Topic | Ask::Offer | Ask::Told { .. } => {
                if before.is_empty() {
                    before.push("I don't know what you are asking.".into());
                }
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
            Ask::Nothing | Ask::More => {
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
            Ask::IsA { x, y, .. } => {
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
            Ask::Rank { things } => {
                before.push("I can't rank them, but I can compare them.".into());
                let (a, b) = (&things[0], &things[1]);
                origin = topic_ring(self, a);
                if let Some((pa, pb)) = infer::meet(&self.wn, (a.pos, a.offset), (b.pos, b.offset)) {
                    if let Some(&node) = self.node_of.get(pa.last().unwrap()) {
                        plans.insert(node, Plan::Compare { a: a.clone(), b: b.clone(), pa, pb, mem: None });
                        put(&mut charge, &mut say, node, 1.0, None);
                    }
                }
                for t in [a, b] {
                    for (n, f) in self.topic_facts(t, &t.lemma) {
                        if f.rel == Rel::Def {
                            put(&mut charge, &mut say, n, 0.6, Some(f));
                        }
                    }
                }
                take = 3;
            }
            Ask::Social { keys } => {
                // the human replies whose turn shares the most (and rarest) words with this one
                let mut score: HashMap<u32, f32> = HashMap::new();
                for k in keys {
                    let Some(entries) = self.index.get(&format!("dlg:{k}")) else { continue };
                    if entries.len() > 20_000 {
                        continue;
                    }
                    let idf = 1.0 / (1.0 + (entries.len() as f32).ln());
                    for &(i, _) in entries.iter().take(4000) {
                        *score.entry(i).or_insert(0.0) += idf;
                    }
                }
                let mut best: Vec<(u32, f32)> = score.into_iter().collect();
                best.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
                best.truncate(300);
                let top = best.first().map(|x| x.1).unwrap_or(1.0).max(1e-6);
                let s = self.nodes.len() as u32;
                // a reply that brings in a thing the message didn't name ("do you know much about fish?")
                // belongs to another talk
                let mine: std::collections::HashSet<&String> = keys.iter().collect();
                best.retain(|(i, _)| {
                    let ws = store::words(&self.store.items[*i as usize].text);
                    // "are they expensive though?": a pronoun for something said in another talk
                    let elsewhere = ws.iter().any(|w| matches!(w.as_str(), "they" | "them" | "their" | "he" | "she" | "his" | "her" | "him" | "those" | "these" | "theirs"));
                    !elsewhere && ws.iter().all(|w| {
                        let n = self.wn.base(w, Pos::N);
                        n.is_none() || store::SOCIAL_NOUNS.contains(&w.as_str()) || self.wn.base(w, Pos::A).is_some() || mine.contains(n.as_ref().unwrap()) || mine.contains(w)
                    })
                });
                for (i, sc) in best {
                    let f = self.item_fact(i, "dialog");
                    put(&mut charge, &mut say, s + i, 0.55 + 0.4 * sc / top, Some(f));
                }
                if charge.is_empty() {
                    before.push("I don't know what you are asking.".into());
                    reading = false;
                }
                take = 1;
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
                    if t.offset == NO_SYNSET {
                        // known only from the world knowledge: nothing to reason from, so what it says
                        for (n, c, f) in self.item_charges(&t.lemma, *q, false) {
                            put(&mut charge, &mut say, n, c, Some(f));
                        }
                    } else {
                        self.plan(t, *q, &mut plans, &mut charge);
                    }
                    let facts = self.topic_facts(t, &t.lemma);
                    let rels: Vec<Rel> = facts.iter().map(|(_, f)| f.rel).collect();
                    let lexfile = self.wn.get((t.pos, t.offset)).map(|s| s.lexfile).unwrap_or(0);
                    for (n, f) in facts {
                        // the dictionary only when nothing can be reasoned: below every premise
                        let c = match q {
                            QWord::Where => match f.rel {
                                Rel::PartOf => 0.6,
                                Rel::Def => 0.5,
                                _ => 0.0,
                            },
                            _ => match f.rel {
                                Rel::Def => 0.5,
                                Rel::Antonym | Rel::HasPart => 0.45,
                                _ => 0.0,
                            },
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
                // two things asked together meet at their nearest common kind
                if topics.len() >= 2 && topics[0].offset != NO_SYNSET && topics[1].offset != NO_SYNSET {
                    let (a, b) = (&topics[0], &topics[1]);
                    if let Some((pa, pb)) = infer::meet(&self.wn, (a.pos, a.offset), (b.pos, b.offset)) {
                        if let Some(&node) = self.node_of.get(pa.last().unwrap()) {
                            plans.insert(node, Plan::Compare { a: a.clone(), b: b.clone(), pa, pb, mem: None });
                            put(&mut charge, &mut say, node, 1.0, None);
                        }
                    }
                }
                take = if topics.len() >= 2 { 3 } else { 3 };
                if let Some(g) = gap {
                    before.push(format!("{g} What I do know:"));
                    take = topics.len().min(2);
                }
            }
        }

        for p in self.block.parts.iter_mut() {
            p.charge = p.fact.and_then(|n| charge.get(&n).cloned()).unwrap_or(0.0);
        }
        let base = match &ask {
            Ask::About { .. } => 2,
            _ => take,
        };
        let budget = think::budget(msg, base) + more;
        let snapshot = self.save_snapshot(&say, about_before)?;
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
            // more fronts while too few have peaked: twice as many at most, four times when none has
            let cap = if up == 0 { 4 * fronts } else { 2 * fronts };
            if up >= want || b.cfg.wave_fronts >= cap {
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

        let mut rings_read = 0;
        // the wave brings up more than is said: a pool of thoughts, weighed before answering
        let mut thoughts: Vec<think::Thought> = Vec::new();
        let pool = (take.max(budget) * 3).max(6);
        if reading {
            for &r in peaked.iter().rev() {
                if rings_read == pool {
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
                    let v = ((b.parts[j].dir * b.omega[r].signum()) < 0.0) as u8;
                    if let Some(plan) = plans.get(&n) {
                        if let Some((text, steps, ok, kind, surprise, familiar, key)) = self.reason(plan, v) {
                            pr.facts.push((n, c, steps, text.clone(), ok));
                            if said_keys.insert(text.clone()) {
                                thoughts.push(think::Thought { text, kind, peak: b.peak[r], charge: c, surprise, familiar, key, ok });
                                any = true;
                            }
                        }
                        continue;
                    }
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
                    if said_keys.insert(f.key()) {
                        let kind = if matches!(f.rel, Rel::Memory | Rel::Ethic | Rel::SelfNote | Rel::Chain | Rel::RelOf) { think::Kind::Link } else { think::Kind::Fact };
                        thoughts.push(think::Thought { text, kind, peak: b.peak[r], charge: c, surprise: 0.0, familiar: 99, key: f.key(), ok });
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
        // think before answering: keep what the question needs, drop the rest, say why in /why
        let verdict = think::think(thoughts, budget, &self.said_before);
        let mut sentences = verdict.kept;
        if let Ask::IsA { why: true, .. } = &ask {
            // why: the same chain, given as the reason
            for s in sentences.iter_mut() {
                if let Some(rest) = s.strip_prefix("Yes: ") {
                    *s = format!("Because {rest}");
                } else if let Some(rest) = s.strip_prefix("Yes, ") {
                    *s = format!("Because {rest}");
                }
            }
        }
        if sentences.iter().any(|s| s.starts_with("You like") || s.starts_with("You love") || s.starts_with("You have")) {
            self.last_link_turn = self.turn;
        }
        for s in sentences.iter().chain(verdict.kept_keys.iter()) {
            self.said_before.insert(s.clone());
        }
        if let Some(pr) = read.last_mut() {
            for (text, why) in &verdict.dropped {
                pr.facts.push((u32::MAX, 0.0, format!("thought dropped: {why}"), text.clone(), false));
            }
        }
        let iv = sentences.len();
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
    fn save_snapshot(&self, say: &HashMap<u32, Fact>, about_before: Option<String>) -> Result<String> {
        std::fs::create_dir_all(&self.dir)?;
        let s = self.talk_base();
        let branded: Vec<(u32, u32)> =
            self.block.parts.iter().enumerate().filter_map(|(i, p)| p.fact.filter(|&n| n >= s).map(|n| (i as u32, n))).collect();
        let mut said_before: Vec<String> = self.said_before.iter().cloned().collect();
        said_before.sort();
        let snap = Snapshot {
            turn: self.turn - 1,
            facts: self.facts.clone(),
            branded,
            say: say.iter().map(|(k, v)| (*k, v.clone())).collect(),
            said_before,
            last_link_turn: self.last_link_turn,
            last_about: about_before,
        };
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
        self.said_before = snap.said_before.into_iter().collect();
        self.last_link_turn = snap.last_link_turn;
        self.last_about_msg = snap.last_about.clone();
        self.last_about = snap.last_about.map(|m| self.read(&m));
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
