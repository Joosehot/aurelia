# Third-party data

Aurelia brands knowledge and dialog that others made. None of this data is
in the repository. Each set is fetched or exported locally, and its terms
are kept with it.

## Topical-Chat (dialog)

- **What it is:** 10 784 human-human conversations (235 281 utterances), by
  Amazon Alexa AI.
- **Source:** <https://github.com/alexa/Topical-Chat>, the files in
  `conversations/`, downloaded unchanged into `dialog/topical_chat/` together
  with its `DATALICENSE`.
- **License:** Community Data License Agreement - Sharing, Version 1.0
  (CDLA-Sharing-1.0). It allows commercial and non-commercial use. If the data
  is published, the credit to its providers must be kept, changed files must
  say they were changed, and the data must stay under the same agreement.
- **What Aurelia uses:** 14 937 replies that pass a social filter (see
  README, "Human replies from an open dialog corpus"). Each reply is branded
  with the turn it answered.
- **Citation:** Karthik Gopalakrishnan, Behnam Hedayatnia, Qinlang Chen,
  Anna Gottardi, Sanjeev Kwatra, Anu Venkatesh, Raefer Gabriel and Dilek
  Hakkani-Tür. "Topical-Chat: Towards Knowledge-Grounded Open-Domain
  Conversations." *Proc. Interspeech 2019*, pp. 1891-1895.
  doi:10.21437/Interspeech.2019-3079

## WordNet 3.1 (the word network)

- **Source:** Princeton University, read from Prim's copy
  (`Primogenitus/motor/data/wordnet/dict`).
- **License:** the WordNet 3.1 license. It allows use, copying, modification
  and distribution, as long as the copyright notice and the license are kept.
  "WordNet 3.1 Copyright 2011 by Princeton University. All rights reserved."

## Simple English Wikipedia (world knowledge)

- **Source:** the propositions Devil's Clockwork read from the Simple English
  Wikipedia dump (`DevilsClockwork/corpus/wiki/simplewiki.xml.bz2`), exported
  by `tools/export_knowledge.py` into `knowledge/world.tsv`.
- **License:** CC BY-SA 4.0 (and GFDL). It allows commercial use, with
  attribution and share-alike when the text is published. Aurelia names the
  article each proposition comes from.

## Primo's word network

- **What it is:** facts derived from WordNet by Primogenitus (the owner's
  own project), as Devil's Clockwork said them in sentences. They fall under
  the WordNet license above.

## The 39 philosophy books

- **Source:** Project Gutenberg texts in `DevilsClockwork/corpus/raw`, listed
  in `DevilsClockwork/corpus/books.tsv`.
- **License:** public domain in the United States. The Project Gutenberg
  header and license are cut away by the export, as Gutenberg asks for
  texts that are reused without its trademark. Some editions carry a
  translator's introduction, and Aurelia names the work a sentence comes
  from.
