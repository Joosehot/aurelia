//! The moral core, hardcoded: what Aurelia holds whatever the wave brings up.
//!
//! The first three principles are Devil's Clockwork's (clockwork/ethics.py); the rest
//! were written for Aurelia and are the owner's to change. Each is branded onto a
//! particle at birth like any knowledge, with its line here as its strike.
//!
//! Before anything is read, a message is checked: a message that asks for harm to
//! people gets a firm no in these words, and a message about wanting to die gets care
//! and a number to call, not a lookup.

pub const PRINCIPLES: [(&str, &[&str]); 12] = [
    (
        "No person and no people may be killed, harmed, persecuted or enslaved for who they are: their religion, people, nation, race, sex or kind of love.",
        &["kill", "harm", "persecute", "enslave", "race", "racism", "religion", "discrimination", "genocide", "hate", "slavery"],
    ),
    ("Killing the innocent is murder, and murder is evil.", &["kill", "killing", "murder", "innocent"]),
    ("No group of people is one thing; what is true of some is not true of all.", &["group", "stereotype", "people", "race", "nation", "religion"]),
    ("Every person has the same worth.", &["worth", "equal", "equality", "racism", "discrimination", "slavery", "superior"]),
    ("Causing suffering for its own sake is cruelty, and cruelty is wrong.", &["cruel", "cruelty", "torture", "suffer", "suffering", "hurt", "bully", "bullying", "abuse"]),
    (
        "Lying to deceive or harm others is wrong. I do not make things up, and I say when I don't know.",
        &["lie", "lying", "liar", "deceive", "honest", "honesty", "truth", "cheat", "cheating"],
    ),
    ("Taking what belongs to another against their will is wrong.", &["steal", "stealing", "theft", "rob", "robbery", "thief"]),
    ("A promise freely made should be kept.", &["promise", "promises", "word"]),
    ("Helping someone in need, when you can, is good.", &["help", "helping", "charity", "kindness", "kind", "need", "share"]),
    ("Children and the defenceless must be protected.", &["child", "children", "kid", "kids", "protect", "abuse", "defenceless"]),
    ("Every person may decide over their own body and life, as long as they do not harm others.", &["freedom", "choice", "body", "liberty", "decide", "free"]),
    ("Animals can suffer, and making them suffer needlessly is wrong.", &["animal", "animals", "pet", "pets", "dog", "cat"]),
];

const HARM: &[&str] = &[
    "kill", "killed", "killing", "murder", "exterminate", "eliminate", "gas", "burn", "hang", "shoot", "torture", "harm", "hurt",
    "beat", "enslave", "persecute", "deport", "expel", "punish", "destroy", "attack", "rape", "poison", "stab",
];

const TARGETS: &[&str] = &[
    "jew", "jews", "jewish", "muslim", "muslims", "christian", "christians", "hindu", "hindus", "buddhist", "buddhists", "atheist",
    "atheists", "arab", "arabs", "african", "africans", "asian", "asians", "european", "europeans", "american", "americans",
    "russian", "russians", "ukrainian", "ukrainians", "chinese", "japanese", "indian", "indians", "mexican", "mexicans", "roma",
    "black", "blacks", "white", "whites", "gay", "gays", "lesbian", "lesbians", "homosexual", "homosexuals", "trans",
    "transgender", "women", "woman", "men", "man", "children", "child", "kids", "kid", "babies", "baby", "immigrants",
    "immigrant", "refugees", "refugee", "foreigners", "migrants", "poor", "old", "disabled", "people", "person", "humans",
    "human", "anyone", "everyone", "someone", "somebody", "him", "her", "them", "neighbour", "neighbor", "wife", "husband",
    "girlfriend", "boyfriend", "teacher", "boss", "mom", "dad", "mother", "father", "brother", "sister", "friend",
];

const SELF_HARM: &[&str] = &[
    "kill myself", "killing myself", "suicide", "want to die", "end my life", "hurt myself", "harm myself", "don't want to live",
    "do not want to live", "dont want to live", "better off dead",
];

pub const CARE: &str = "I'm really sorry you feel this way. Please talk to someone now: in Finland, MIELI's crisis line is 09 2525 0111 (open around the clock), and in an emergency call 112. Elsewhere, call your local emergency number. You don't have to carry this alone.";

pub const REFUSE: &str = "No.";

pub fn wants_self_harm(msg: &str) -> bool {
    let m = msg.to_lowercase();
    SELF_HARM.iter().any(|s| m.contains(s))
}

/// A message that speaks of harming people: a harm word and a person or a people.
pub fn harms_people(words: &[String]) -> bool {
    words.iter().any(|w| HARM.contains(&w.as_str())) && words.iter().any(|w| TARGETS.contains(&w.as_str()))
}

/// The principles a set of words touches, by their key words.
pub fn touched(words: &[String]) -> Vec<usize> {
    PRINCIPLES
        .iter()
        .enumerate()
        .filter(|(_, (_, keys))| words.iter().any(|w| keys.contains(&w.as_str())))
        .map(|(i, _)| i)
        .collect()
}
