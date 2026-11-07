# Aurelia

This folder holds two things:

- **Aurelia** (the Rust crate at the top, `src/`): a conversational engine
  built from a 3D block of ring accelerators. It uses no LLM and no neural
  network. It answers only from knowledge branded into its particles.
- **The particle maker** ([`particle_maker/`](particle_maker/README.md),
  Python): a stirred rice bowl whose friction forms artificial particles. It
  is a separate program and is used like this:
  `python -m particle_maker run --minutes 2 --stir spiral --seed 42`.

The rest of this file is about Aurelia.

```
cargo build --release
target\release\aurelia chat            # talk; /why shows the last answer's rings; /quit
target\release\aurelia chat --viz      # the same, with the block drawn as the wave passes
target\release\aurelia ask "What is a dog?" "Is a dog an animal?"   # lines as one conversation
target\release\aurelia debug           # which rings peaked and what they held, last answer
target\release\aurelia replay runs\chat\<run> <n>                   # replay answer n exactly
cargo test --release                   # particles never destroyed; collisions are clockwork
```

Settings are in `aurelia.toml`. The word network is Prim's: WordNet 3.1 as
`Primogenitus/motor/data/wordnet/dict` keeps it. Aurelia only reads that
folder.

**Why Rust:** Prim and its word network are Rust. The block has 17 672 rings
and 53 016 particles, and one answer's wave takes 330–350 steps. In Rust on
the CPU that is about 0.85 s per answer, with about 2.3 million collisions.
The machine has an RTX 4070, but Aurelia does not use the GPU, because
the CPU is fast enough. The collisions are sequential (each one changes the
next), and that is also what makes every answer exactly replayable from its
seed.

## 1. The grid

The lattice has 48 × 48 × 8 points. Every four neighbouring points of a
layer are one **ring accelerator**: the circle through those four corners.
That gives 47 × 47 rings per layer, in 8 stacked layers. Particles on a ring
pass its corners at 45°, 135°, 225° and 315°. A corner point is shared by up
to four rings, and that is where rings touch.

- **Spin.** Each ring turns at its own pace. The directions alternate like a
  checkerboard (so neighbours mesh like gears), and a seeded 15% turn the
  other way. Each answer's seed varies every pace by ±30%.
- **Layers.** Stacked rings share spin through an "axle" (2% per step).

## 2. Particles and clockwork

Each ring carries 3 particles that orbit with it. A particle has an angle, a
direction (with its ring or against it), a lift (how far it has risen toward
its ring's limit) and possibly a branded fact.

When a particle passes a corner and another ring's particle is at the same
corner (within 0.15 rad), they collide. A collision is clockwork:

- both particles **bounce back**: their directions flip;
- **lift passes** from the higher particle to the lower one (8% of the
  difference), so the total stays the same;
- the two rings' **spins mesh**: both are pushed toward turning against each
  other, while ω_a − ω_b stays the same.

Nothing is ever destroyed or fused. The particle array never changes length.
A collision passes only direction, lift and spin, and a particle's branded
fact never changes. `cargo test` checks this over five seeds
(`particles_are_never_destroyed`, `a_collision_is_clockwork`,
`rings_touch_at_shared_corners`).

## 3. Knowledge by branding, with a lightning strike

Nothing is trained. A fact is **branded** onto particles when the
conversation first needs it.

- **Knowledge** comes from the word network, from the first (most common)
  sense of a word. That covers the definition, kind of (hypernym), instance
  of, part of, member of, has parts, other words for it, the opposite, and
  example sentences.
- **Derived knowledge** is also branded. "Is X a Y" gets the shortest
  hypernym chain from X up to Y. "The R of Y" gets a part of Y that is a
  kind of R (the capital of France: Paris is a part of France and a kind of
  capital).
- **Memory nodes** are what was said: every line the user says, and what the
  user tells about themselves ("my X is Y", "I am / like / love / have / live
  in …", "call me …").

Every fact is branded together with its **lightning strike**: a link to
where it came from (the synset, the chain of synsets, or the user's exact
line), plus the turn and the word that brought it. Before a fact may be
said, Aurelia derives it again from its strike, and a fact that doesn't come
out the same is not said (`/why` marks it `FAILED CHECK`). So an answer can
only hold facts that come from WordNet or from the conversation. The
phrasing rules add only connecting words.

Each fact is branded onto **3 particles**, each on a ring of its own, at the
free rings nearest the word's home ring (a hash of the word). The facts of
one word therefore sit together. Particles are never re-used for something
else while there is room, and memory is never overwritten.

## 4. The wave, and how a message enters (the design chosen)

1. **The message is read whole** into one ask: what is X, who, where,
   when/why/how, is X a Y, the R of Y, the opposite of X, something about
   me, what did I say, about Aurelia itself, or a greeting. The words it
   asks about are found in the word network (phrases of three words first,
   then two, then one; base forms by WordNet's own rules).
2. **The message brands** the knowledge of those words, if it isn't there
   yet, and charges the particles that answer the ask. A definition gets
   charge 1.0, kind-of 0.9, part-of 0.8, and so on. For "where", part-of
   gets 1.0; for "is X a Y", the chain alone.
3. **The charge sets the ring's limit**: limit = 1 + 4 × the highest charge
   on the ring. An uncharged ring has limit 1.
4. **The message sets where the wave starts**: at the home ring of the first
   word it asks about.
5. **The wave rolls.** Before it, the block turns on its own for 120 steps
   (seeded). Then a train of 12 spherical fronts leaves the start, 24 steps
   apart, at 0.7 rings per step. A front passing a ring spins it up. Spin
   above a ring's own pace lifts its particles toward the ring's limit (half
   as fast for a particle running against its ring). Collisions spread spin
   and lift sideways through the block.

## 5. Reading the answer

When the last front has passed the far end of the block, the reading
begins.

- Uncharged rings (limit 1) peak early. The rings that rise to their limit
  **last** are the charged ones with the highest limits.
- The reply is read from the last charged rings to peak, in the order they
  peaked: 2 per word asked about (4 at most), or 1 for a one-answer ask.
- Each fact is checked against its strike and said once. The wording (one
  of two per kind of fact) is set by the particle at that moment: whether it
  runs with its ring or against it.
- When a ring holding the most charged fact gets drained by its neighbours
  and doesn't reach its limit in time, it isn't read. Then the answer shifts
  to another angle.

The same question gives different answers because the paces, the angles and
the 120 steps of the block's own turning come from a fresh seed each time.
Each reply logs its seed and a snapshot of the branded block
(`runs/chat/<run>/replies.jsonl`, `snap-NNNN.json`), and `aurelia replay`
gives the same answer again. This was checked on two answers.

When nothing holds an answer, Aurelia says so:

- a word in neither the word network nor the talk gets *I don't know what
  "X" is*;
- a question the knowledge cannot answer (who for a thing, when, why, how,
  where for something with no "part of") gets *I don't know when. What I do
  know: …*;
- something asked about the user that the user hasn't told gets *You
  haven't told me*;
- "is X a Y" with no chain gets *I have nothing that says …*. Aurelia never
  says "no" here, because the network not holding it is not evidence.

## 6. The test chat (10 questions, fresh seeds, after development)

The settings were developed on other questions (cat, Rome, Newton, Italy,
cold). After that, this conversation was run once with fresh seeds, and
nothing was changed because of it.

| asked | Aurelia | |
|---|---|---|
| Hello! | Hello. Ask me about a word, or tell me something about yourself. | greeting rule |
| What is a dog? | Dogs are a kind of domestic animal. Parts of a dog include a flag. | true to WordNet; "flag" is WordNet's word for a dog's tail |
| What is a dog? | A dog is a kind of canine. Parts of a dog include a flag. | another angle; no definition either time |
| Is a dog an animal? | Yes, dogs are animals (by way of domestic animal). | the chain, checked |
| Who was Albert Einstein? | Albert Einstein is one of the physicists. Albert Einstein: physicist born in Germany who formulated the special theory of relativity … (1879-1955). | |
| What is the capital of France? | The capital of France is Paris. | derived: part of France and a kind of capital |
| My name is Joose. → What is my name? | I'll remember that. Your name is Joose. → Your name is Joose. | memory |
| Who won the 2022 World Cup? | I don't know who. What I do know: About a won: the basic unit of money in South Korea. World Cup is a kind of tournament. | honest about who, but "won" was read as the Korean money |
| What is a quantum blockchain? | I don't know "blockchain". Quanta are a kind of quantity. A quantum is a discrete amount … | honest about "blockchain", but it answers about "quantum" alone |
| Where is Helsinki? | Helsinki is the capital and largest city of Finland; … Helsinki is in Finland. | |
| What does happy mean? | Unhappy is the opposite of happy. Happy means enjoying or showing or marked by joy or pleasure. | |

No fact was invented: every fact said is in WordNet or was said by the
user, and every one passed its strike check.

## What works and what doesn't

**It does well at these:**

- definitions, kinds, places, parts and opposites, read straight from the
  word network;
- "is X a Y" with the chain shown;
- "the capital of France"-type questions when WordNet links them;
- remembering what you tell it about yourself;
- saying "I don't know" for words it doesn't have;
- exact replay of any answer;
- particles that are never destroyed.

**Weak, or not working yet:**

- **The wave mostly sorts by charge.** What the wave adds is noise in the
  timing (which of the close facts peaks last) and drain (a fact's ring
  losing lift to its neighbours). Distance matters only when the facts of
  two words sit far apart. The variety is real, but it isn't the wave
  "thinking". The order of a reply is the order of the peaks, so a
  definition sometimes comes after a detail.
- **Word senses:** always the most common sense. "won" in "who won" was
  read as the Korean money (the verb rule only covers do/does/did
  questions). "bank" will be the river bank.
- **Partly known phrases** are answered from the known part ("quantum
  blockchain" → quantum). It does say the other word is unknown, but it
  should say it doesn't know the whole.
- **"What is a dog?" gave no definition in either test run**, because the
  definition's rings didn't peak in time. Raising the drive makes everything
  peak and kills the variety. This trade-off is not solved.
- **The phrasing is rule-based and stiff:**
  - mass nouns get plurals ("Rains are a kind of precipitation");
  - true WordNet facts can read oddly ("parts of a dog include a flag");
  - long definitions are given whole;
  - tense is not known ("Albert Einstein is one of the physicists").
- **Questions it cannot read:** anything outside the asks listed in section
  4 falls back to "the words it asks about". Questions about relations
  WordNet lacks (who won, how many, why) only get "I don't know" plus
  related knowledge.
- **Memory** understands only the simple statement forms listed above, and
  there is one user.
- **GPU:** not used (see the start of this file).
- **The viz** is a top view in the terminal (each column is its highest
  ring; a lit `@` is a charged ring that reached its limit), not a 3D view.
