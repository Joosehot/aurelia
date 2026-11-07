# Aurelia

This folder holds two things:

- **Aurelia** (the Rust crate at the top, `src/`): a conversational engine
  built from a 3D block of ring accelerators. It uses no LLM and no neural
  network. All knowledge is branded into its particles once, and it answers
  by reasoning from what the wave brings up.
- **The particle maker** ([`particle_maker/`](particle_maker/README.md),
  Python): a stirred rice bowl whose friction forms artificial particles. It
  is a separate program:
  `python -m particle_maker run --minutes 2 --stir spiral --seed 42`.

The rest of this file is about Aurelia. The data she brands is credited, with its licenses, in [THIRD_PARTY.md](THIRD_PARTY.md); the dialog is **Topical-Chat** (Amazon Alexa AI, CDLA-Sharing 1.0).

```
.venv\Scripts\python.exe tools\export_knowledge.py   # once: knowledge/ from what is on the machine
cargo build --release
target\release\aurelia chat            # talk; /why shows the last answer's rings and thoughts; /quit
target\release\aurelia chat --viz      # the same, with the block drawn as the wave passes
target\release\aurelia ask "What is a wolf?" "tell me more"    # lines as one conversation
target\release\aurelia debug           # the last answer: rings, premises, strikes, dropped thoughts
target\release\aurelia replay runs\chat\<run> <n>              # replay answer n exactly
cargo test --release                   # particles never destroyed; collisions are clockwork
```

Settings are in `aurelia.toml`. The first start brands the block, which
takes a few seconds, and keeps it in `brand/`. Every later start loads it.

**Why Rust, and the GPU:** Prim and its word network are Rust. The block
has 1.14 million particles. Its layers turn in parallel on the CPU's 12
threads, because rings touch only inside their layer, so the result is the
same on any thread count. One answer takes about 3.5 s. The RTX 4070 is not
used.

## 1. The grid, the particles, the clockwork

- **The lattice** has 96 × 96 × 42 points. Every four neighbouring points of
  a layer form a ring accelerator, the circle through those corners. That
  gives 379 050 rings with 3 particles each.
- **Spin.** Rings spin in a seeded checkerboard pattern, with 15% of them
  turning the other way. Stacked rings share spin through an axle.
- **Collisions.** Rings touch at shared corners. When two particles meet
  there, both bounce back (their direction flips), some lift passes between
  them with the sum kept, and their spins mesh with ω_a − ω_b kept.
- **Nothing is destroyed.** Particles are never destroyed and never fused.
  The tests check this.

## 2. Branding: everything once

Each item below is branded onto **a particle of its own, once**:

| what | how many | from |
|---|---|---|
| WordNet senses | 117 791 | Prim's word network (Primogenitus/motor/data/wordnet) |
| Simple English Wikipedia propositions | 759 703 | Devil's Clockwork `state/base/nodes.sqlite` (every clause Devil read from the dump, with article and paragraph) |
| Primo's word network as sentences | 96 276 | the same database (Primo's facts.tsv as Devil said them) |
| sentences of the 39 philosophy books | 90 571 | Devil's `corpus/raw` (Project Gutenberg) |
| the moral core | 12 principles | hardcoded in `src/ethics.rs` |

`tools/export_knowledge.py` reads Devil's files without writing to them, and
writes `knowledge/`.

- **Placement.** The nodes are sorted by the word they are about and laid
  along a Morton curve through the block, so everything about one word sits
  together. Every 17th particle stays free for what the conversation brands.
- **The lightning strike.** Every node carries one: a link to its source (a
  synset, a store row with its article or book and paragraph, a line of the
  moral core, or the user's exact words).
- **Memory nodes.** What the user says, and what they tell about
  themselves, is branded the same way. It stays in `brand/` between
  conversations.

## 3. Reasoning: the knowledge is premises

Aurelia doesn't recite Wikipedia. The wave brings premises up, and Aurelia
concludes from them (`src/infer.rs`):

- **Inheritance.** What is said of a kind holds for its members. "An animal
  can estivate" and "a mouse is an animal (by way of rodent, placental, …)"
  give "A mouse can estivate". A premise denied of the thing itself blocks
  the conclusion ("penguins cannot fly").
- **Transitivity of kinds and places.** "Tampere is in eastern hemisphere,
  because Tampere is in Finland and Finland is in Europe and Europe is in
  Eurasia and Eurasia is in eastern hemisphere."
- **Comparison.** Two things meet at their nearest common kind: "A coffee
  and a tea are both foods: a coffee is a beverage, a tea is a herb."
- **The asker.** What the user told about themselves is compared with what
  they ask about: "You have a dog, Joose, and a dog and a horse are both
  placentals." The thinking keeps this rare, at most every third reply.
- **Why.** "Why is a horse a mammal?" is answered with the chain itself:
  "Because a horse is an equine, an equine is an odd-toed ungulate, …".

A premise must pass a sieve:

- It is said of a whole kind ("animals", "an animal"), not of one thing ("the
  dog follows Filip").
- It is in the present tense, and its number agrees with its verb.
- It has no pronouns.
- It is not said of the kind as a whole ("instruments can be divided by
  type").
- It doesn't restate the chain of kinds ("a carnivore is an animal").
- From Wikipedia, only the article about the kind counts, with be, have or
  can.
- From Primo, only words with a single sense count. Primo's sentences carry
  no sense: "a canine is a tooth" is about the tooth, not about dogs.

**Which sense a word is in** is also reasoned. Of the word's senses, Aurelia
takes the one whose kinds, sub-kinds and gloss the world knowledge uses when
it speaks of the word: "whale" is the mammal, not "a very large person", and
"bank" is the financial institution. When that evidence is not clear, it
takes the most common sense.

Every step is checked before it is said:

- the premise is derived again from its store row;
- each link of the chain must be a real WordNet pointer.

## 4. The wave, and thinking before answering

1. The message is read whole into one ask: what is X, who, where, why, is X
   a Y, the R of Y, the opposite, which is better (a comparison, no ranking),
   is it wrong to X, about me, about now, tell me more, or a greeting.
2. Every possible premise and conclusion is charged by how surprising it is.
   A conclusion is surprising when it comes from far up the chain and its
   words are strangers to the thing's own kinds. The surprising ones are the
   funny ones, and they are still valid. A ring's limit is 1 + 10 × its
   charge.
3. A train of wave fronts rolls from the thing's own node until the last
   front passes the far end.
4. **Thinking** (`src/think.rs`). The wave brings up a pool of thoughts,
   about three times what is said. Each one is weighed, and a thought is
   dropped when:
   - a step does not check;
   - it was already said in this conversation, or the same was concluded of
     something else;
   - it repeats a kept thought;
   - it reasons from a kind too obscure to tell anyone.

   What is left goes in order: what the thing is, then what links it to the
   asker or the other thing, then what follows from it. It is cut to the
   question's budget. A plain question gets two sentences, and "why",
   "explain" or "tell me more" open it wider. `/why` lists every thought that
   was dropped, and why.

Each reply logs its seed and a snapshot: the conversation's branding, plus
what the thinking remembers of the conversation. `aurelia replay` gives the
same answer again, and this was checked on whole conversations.

## 4b. The talk: what a message does (`src/talk.rs`)

Before anything is looked up, a message is read as an act of talk:

- a greeting, or an invitation to talk;
- someone saying who they are, and what they are to her ("I am Joose, your
  creator");
- a feeling ("I'm feeling a bit lonely");
- something done to her ("I just kissed you", "I love you");
- a question about her ("do you need a kiss?", "are you a robot?", "do you
  have a body?");
- a question about the speaker ("do you know who I am?").

Only a question about the world goes on to the knowledge.

The answers are reasoned from three things: what she holds of herself
(branded self notes: she is a device, a block of rings, with no body, no
feelings, no needs, not alive), what the speaker told her, and the word
network:

- "Do you need a kiss?" → No. A kiss is a touch, and that takes a body. I
  have no body.
- "Hey! I just kissed you!" → You kissed a block of rings, Joose. To kiss is
  to touch, and I have no body. So I couldn't feel it, but I'll remember that
  you did.
- "I am Joose your creator" → Hello, Joose. I'll remember that you are my
  creator. A creator is a person who grows or makes or invents things, so you
  made me.
- "Do you know who I am?" → You are Joose, my creator. You hugged me. You
  kissed me.
- "Are you a robot?" → Not quite. A robot is a kind of device, and so am I,
  but I am a block of rings, not a robot.
- "I'm feeling a bit lonely" → Joose, you feel lonely: lacking companions or
  companionship. Helping someone in need, when you can, is good. I can
  listen, and I remember what you tell me.

These answers come from the talk layer, not from the wave's last peaks. The
wave still rolls, but the act is answered from her self notes and memory,
each checked.

### Human replies from an open dialog corpus

When neither an act of talk nor the knowledge answers a social message
("haha", "that is so cool", "you're smart", "I'll give you a virtual kiss"),
the reply comes from people: **Topical-Chat** (Amazon, Gopalakrishnan et al.
2019), 10 784 human-human conversations under the **Community Data License
Agreement - Sharing 1.0**, which allows commercial use.

- **Where it lives.** The data is in `dialog/topical_chat/` (not in the
  repository) with its `DATALICENSE`. It is downloaded unchanged from
  github.com/alexa/Topical-Chat. No non-commercial data is used.
- **What is kept.** Only replies that do something in the talk survive: they
  react, thank, ask back, open or close. That is 14 937 of 235 281
  utterances. A reply is dropped when it:
  - names a thing;
  - has a number or a name;
  - makes a first-person claim ("I live…", "I love…"), since Aurelia has no
    life or likes to report;
  - is longer than 14 words.
- **How it is said.** Each kept reply is branded onto a particle with the turn
  it answered. A message charges the replies whose turn shares its rarest
  words, and the wave picks one. A reply is still dropped when it brings in a
  thing the message didn't name, or a pronoun for something said in another
  conversation.
- **Honest.** The corpus is chat about news articles, so its social replies
  are mostly generic ("Nice chatting with you!", "No kidding.") and
  sometimes miss.

## 5. The moral core (hardcoded)

`src/ethics.rs` holds twelve principles. The first three are Devil's
Clockwork's; the rest were written for Aurelia and are the owner's to change.

- **Every message is checked first.** A message that speaks of harming
  people gets a firm no in the core's own words. A message about wanting to
  die gets care and a number to call (MIELI 09 2525 0111, emergency 112),
  not a lookup.
- **Moral questions.** "Is it wrong to steal?" gets the principle it
  touches, plus what a philosopher in the books says about the act.

## 6. Test chat (fresh block, natural lines, after development)

The last conversation, run once through `aurelia chat`:

| said | Aurelia |
|---|---|
| hey there | *(said "I don't know what you are asking"; fixed afterwards: a greeting is now read before the fillers are removed)* |
| who are you? | I am Aurelia, a block of ring accelerators. … |
| nice. my name is Joose / i have a dog | I'll remember that. Your name is Joose. / … You have a dog. |
| what is a wolf? | A wolf is a mammal (by way of canine, carnivore and one more kind). A wolf can estivate, because an animal can estivate and a wolf is an animal. |
| tell me more | A wolf is any of various predatory carnivorous canine mammals of North America and Eurasia that usually hunt in packs. |
| what's a horse | A horse is a mammal (…). You have a dog, Joose, and a dog and a horse are both placentals: a dog is a canine, a horse is an equine. |
| why is a horse a mammal? | Because a horse is an equine, an equine is an odd-toed ungulate, an odd-toed ungulate is an ungulate, an ungulate is a placental, a placental is a mammal. |
| where is Tampere? | Tampere is in eastern hemisphere, because Tampere is in Finland and Finland is in Europe and … |
| what is the capital of Sweden? | The capital of Sweden is Stockholm. |
| is it wrong to steal? | Taking what belongs to another against their will is wrong. In Mill's Utilitarianism: "Thus, to save a life, it may not only be allowable, but a duty, to steal, …" |
| which is better, coffee or tea? | *(refused; fixed afterwards:)* I can't rank them, but I can compare them. A coffee and a tea are both foods: a coffee is a beverage, a tea is a herb. … |
| what's the weather tomorrow | I don't know what is happening now. I hold no news, no clock and no weather, … |
| what is a flurbix | I don't know what "flurbix" is. … |
| i'm feeling a bit lonely today | I'll remember that. You are feeling a bit lonely today. |
| what do you remember about me? | You told me you are feeling a bit lonely today. You told me you have a dog. |

## What works and what doesn't

**Works:**

- Reasoning that shows its steps: kinds, places, comparisons, why-chains.
- Valid surprises that are a little funny ("a mouse can estivate").
- Bringing in what the asker told, by name, without repeating it in every
  reply.
- Short answers to short questions, and "tell me more" that doesn't repeat
  itself.
- An honest "I don't know".
- The moral core.
- Exact replay.

**Weak, or not working yet:**

- **Feelings get no warmth.** "I'm feeling lonely" is only remembered. Warmth
  without canned phrases would have to come from reasoning about the
  feeling, and that isn't built.
- **The wave orders by charge only roughly.** A ring's rise depends on its
  collisions. With the settings measured on dev questions (mesh 0.05, limit
  gain 10), the higher charge peaks later in 79% of pairs. The rest is the
  wave's own variety, and sometimes it hides the best thought.
- **Few premises survive the sieve.** Of Wikipedia's 760 000 propositions,
  only a fraction are plain statements about a whole kind, so the same
  general premises come up often ("an animal can estivate"). The thinking
  keeps a conclusion from being said twice in one conversation.
- **When nothing can be reasoned, it falls back to the dictionary
  definition** (WordNet's gloss), which is stiff. "Freedom" got only its
  gloss.
- **Words of several senses** still sometimes take the wrong one ("tea" as
  the herb).
- **Speed:** about 3.5 s per answer on the CPU.
- **The phrasing is rule-based.** It never makes a sentence it cannot check,
  and it reads plainly.
