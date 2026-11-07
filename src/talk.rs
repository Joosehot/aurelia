//! The talk: what a message does, before what it is about.
//!
//! A message is first read as an act of talk: a greeting, an invitation, someone
//! saying who they are, a feeling, something done to Aurelia ("I just kissed you"),
//! a question about her ("do you need a kiss?", "are you a robot?") or about the
//! speaker ("do you know who I am?"). Only a question about the world goes on to
//! the knowledge.
//!
//! The answers are reasoned from what she holds of herself, branded like any
//! knowledge (she is a device, a block of rings; she has no body, no feelings, no
//! needs), from what the speaker told, and from the word network: a kiss is a
//! touch, a touch takes a body, she has none.

use crate::infer::{self, Key};
use crate::wordnet::{Pos, WordNet};

/// Her kind in the word network.
pub const KIND: &str = "device";

/// Kinds of things and acts that only a body can have or take.
const BODILY: &[&str] = &[
    "touch", "touching", "contact", "physical_contact", "caress", "embrace", "food", "nutriment", "beverage", "drink", "meal",
    "sleep", "rest", "exercise", "massage", "bath", "medicine", "sex", "breathing", "hug", "kiss",
];

/// Things she has, or not, when asked "do you have ...".
pub fn has(x: &str) -> Option<bool> {
    match x {
        "name" | "ring" | "particle" | "wave" | "memory" | "knowledge" => Some(true),
        "body" | "heart" | "face" | "eye" | "hand" | "skin" | "feeling" | "emotion" | "soul" | "need" | "brain" | "mouth" | "lip"
        | "family" | "mother" | "father" => Some(false),
        _ => None,
    }
}

/// Does a thing or an act take a body? The kind on the way that says so.
pub fn bodily(wn: &WordNet, key: Key) -> Option<String> {
    // what it is a kind of says whether it takes a body ("a kiss is a touch"), not the word itself
    let paths = infer::ancestors(wn, key, 8);
    for p in paths {
        let k = *p.last().unwrap();
        if let Some(s) = wn.get(k) {
            if let Some(w) = s.words.iter().find(|w| BODILY.contains(&w.to_lowercase().as_str())) {
                return Some(w.replace('_', " "));
            }
            // verbs of the body, of eating and drinking, of contact, of moving
            if k.0 == Pos::V && matches!(s.lexfile, 29 | 34 | 35 | 38) {
                return Some(crate::wordnet::show(&s.words[0]));
            }
        }
    }
    None
}

/// A verb of feeling ("love", "miss", "like").
pub fn feeling_verb(wn: &WordNet, lemma: &str) -> bool {
    wn.senses(lemma, Pos::V).first().and_then(|o| wn.get((Pos::V, *o))).map(|s| s.lexfile == 37).unwrap_or(false)
}

/// The verb a verb is a way of: kiss -> touch.
pub fn verb_kind(wn: &WordNet, lemma: &str) -> Option<String> {
    let off = *wn.senses(lemma, Pos::V).first()?;
    let s = wn.get((Pos::V, off))?;
    let up = s.ptrs.iter().find(|p| p.sym == "@")?;
    wn.get(up.target).map(|h| crate::wordnet::show(&h.words[0]))
}

/// A feeling that tells of a need ("lonely: lacking companions").
pub fn hard_feeling(gloss: &str) -> bool {
    let g = gloss.to_lowercase();
    ["lack", "without", "sad", "unhappy", "pain", "suffer", "sorrow", "fear", "anxi", "weary", "tired", "alone", "depress", "grief", "worr", "ill", "hurt", "miser", "afraid", "angry", "lonel"]
        .iter()
        .any(|w| g.contains(w))
}

/// Adverbs that sit between "I" and the act ("I just kissed you").
pub const SOFT: &[&str] = &["just", "really", "so", "totally", "also", "actually", "finally", "already", "kinda", "truly", "still", "do"];

/// Roles the speaker can have toward her.
pub const ROLES: &[&str] = &["creator", "maker", "builder", "father", "mother", "parent", "friend", "owner", "developer", "programmer", "user", "author", "designer"];

/// Words of a feeling report after "I am" / "I feel".
pub const DEGREE: &[&str] = &["a", "bit", "little", "so", "very", "really", "kind", "of", "kinda", "quite", "pretty", "feeling", "rather", "too", "today", "now", "tonight", "right"];

/// What she does: talk, answer, reason, remember. Anything else she cannot.
pub const ABLE: &[&str] = &[
    "talk", "chat", "speak", "answer", "listen", "remember", "reason", "think", "compare", "explain", "reply", "read", "write",
    "tell", "help", "converse", "recall", "infer", "deduce", "conclude", "learn", "count", "check",
];

/// Can she do it? Yes or no, and what it takes when no.
pub fn able(wn: &WordNet, verb: &str) -> (bool, Option<&'static str>) {
    if ABLE.contains(&verb) {
        return (true, None);
    }
    let Some(&off) = wn.senses(verb, Pos::V).first() else { return (false, None) };
    let lex = wn.get((Pos::V, off)).map(|s| s.lexfile).unwrap_or(0);
    match lex {
        29 | 34 | 35 | 38 => (false, Some("body")),
        39 => (false, Some("sense")),
        37 => (false, Some("feeling")),
        _ => {
            if bodily(wn, (Pos::V, off)).is_some() {
                (false, Some("body"))
            } else {
                (false, None)
            }
        }
    }
}