//! Thinking before answering: the wave brings up more than is said. Each thought
//! that peaked is weighed, and only what the question needs is kept.
//!
//! A thought is dropped when a step of it does not check, when it was already said
//! in this talk, when it says the same as a thought kept before it, or when it
//! reasons from a kind too obscure to say anything to the asker ("a craniate").
//! What is left is put in order (what the thing is, what links it to the asker or
//! to the other thing, what follows from it, then plain facts) and kept up to the
//! question's budget: a short question gets a short answer, and only "why",
//! "explain" or "tell me more" open it up.

use std::collections::HashSet;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Kind {
    Identity,
    Link,
    Inference,
    Fact,
}

#[derive(Clone, Debug)]
pub struct Thought {
    pub text: String,
    pub kind: Kind,
    /// the step its ring peaked at
    pub peak: i32,
    pub charge: f32,
    /// how unexpected a conclusion is (0 for plain facts)
    pub surprise: f32,
    /// how much the world knowledge says of the kind it reasons from
    pub familiar: usize,
    /// what it says, for "says the same"
    pub key: String,
    pub ok: bool,
}

pub struct Verdict {
    pub kept: Vec<String>,
    /// what the kept thoughts say, so a later reply doesn't say it again of something else
    pub kept_keys: Vec<String>,
    pub dropped: Vec<(String, &'static str)>,
}

/// The words a reply may run to: one sentence is always kept, then more while the
/// budget allows.
fn word_cap(budget: usize) -> usize {
    28 + 22 * budget.saturating_sub(1)
}

pub fn think(mut thoughts: Vec<Thought>, budget: usize, said_before: &HashSet<String>) -> Verdict {
    let mut dropped = Vec::new();
    let last_peak = thoughts.iter().map(|t| t.peak).max().unwrap_or(0).max(1) as f32;
    // what follows from it: the later it peaked and the more unexpected, the better
    let score = |t: &Thought| -> f32 { 0.6 * t.surprise + 0.4 * (t.peak as f32 / last_peak) + 0.2 * t.charge };
    thoughts.sort_by(|a, b| a.kind.cmp(&b.kind).then(score(b).partial_cmp(&score(a)).unwrap()));
    let mut keys = HashSet::new();
    let mut kept: Vec<String> = Vec::new();
    let mut kept_keys: Vec<String> = Vec::new();
    let mut words = 0;
    let mut identity = false;
    for t in thoughts {
        let why = if !t.ok {
            Some("a step does not check")
        } else if said_before.contains(&t.text) {
            Some("already said in this talk")
        } else if t.kind == Kind::Inference && said_before.contains(&t.key) {
            Some("the same was concluded of something else already")
        } else if keys.contains(&t.key) {
            Some("says the same as a thought kept")
        } else if t.kind == Kind::Inference && t.familiar < 3 {
            Some("reasons from a kind too obscure to tell anyone")
        } else if t.kind == Kind::Identity && identity {
            Some("one answer to what it is is enough")
        } else if kept.len() >= budget {
            Some("over the question's budget")
        } else if !kept.is_empty() && words + t.text.split_whitespace().count() > word_cap(budget) {
            Some("would make the answer longer than the question asks")
        } else {
            None
        };
        match why {
            Some(w) => dropped.push((t.text, w)),
            None => {
                identity |= t.kind == Kind::Identity;
                keys.insert(t.key.clone());
                words += t.text.split_whitespace().count();
                kept.push(t.text);
                kept_keys.push(t.key);
            }
        }
    }
    // asked again, the same answer again is better than silence
    if kept.is_empty() {
        if let Some(i) = dropped.iter().position(|(_, w)| *w == "already said in this talk") {
            let (text, _) = dropped.remove(i);
            kept.push(text);
        }
    }
    Verdict { kept, kept_keys, dropped }
}

/// How much a question asks for: the reply's budget in sentences.
pub fn budget(message: &str, base: usize) -> usize {
    let m = message.to_lowercase();
    let wide = ["why", "explain", "tell me more", "in detail", "everything", "more about", "elaborate", "go on"];
    if wide.iter().any(|w| m.contains(w)) {
        base + 2
    } else {
        base
    }
}
