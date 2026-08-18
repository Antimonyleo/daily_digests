"""Curated, dependency-free science notes for the reader's daily tea break."""

from __future__ import annotations

from datetime import date
from hashlib import sha256

DAILY_TEA_DECK_SIZE = 15
# Cards drawn per day. Jokes outnumber facts: Pip is a break from reading, not
# another source of it. Each bank must hold at least twice its draw so two
# consecutive days can never share a card (see ``_daily_pick``).
DAILY_JOKES = 10
DAILY_FACTS = 5

# Deliberately skewed toward the surprising over the textbook: a fact everyone
# met in their first year is not a break, it is a flashcard.
_TEA_FACTS = (
    "Tiny fact — The ribosome’s catalytic center is RNA, so every protein begins with an RNA-powered reaction.",
    "Tiny fact — Most of DNA’s stability comes from bases stacking like coins, not from the hydrogen bonds that get all the credit.",
    "Tiny fact — Cytoplasm carries 300–400 grams of macromolecule per litre; a protein inside a cell lives in a crowd, not a solution.",
    "Tiny fact — A hydrogen bond in liquid water lasts about a picosecond before it swaps partners.",
    "Tiny fact — A swimming bacterium lives near Reynolds number 10⁻⁵: stop pushing and it coasts less than an atom’s width.",
    "Tiny fact — Diffusion is superb at short range and hopeless at long range — a protein crosses a bacterium in about a tenth of a second and would need centuries to cross a metre.",
    "Tiny fact — Titin, the largest human protein, runs to roughly 34,000 amino acids: one gene, one enormous molecular spring.",
    "Tiny fact — ATP synthase is a rotary motor. Its stalk turns inside the membrane, and one full revolution releases three ATP.",
    "Tiny fact — Kinesin walks a microtubule in 8-nanometre steps, one head passing the other.",
    "Tiny fact — Cryo-EM vitrifies water rather than freezing it: cool fast enough and it becomes glass, so no ice crystals shred the sample.",
    "Tiny fact — Single-particle cryo-EM combines images of many frozen particles in different orientations into one three-dimensional reconstruction.",
    "Tiny fact — The genetic code is degenerate along chemical lines: synonymous codons cluster so many single-base slips keep the amino acid’s character.",
    "Tiny fact — An E. coli replisome copies DNA at roughly a thousand base pairs a second and still errs less than once per billion bases.",
    "Tiny fact — Optical tweezers hold a bead in a beam of light, and the trap behaves like a spring soft enough to weigh a single motor protein.",
    "Tiny fact — At the nanoscale, Brownian motion helps self-assembling particles explore possible arrangements.",
    "Tiny fact — DNA origami usually folds one long scaffold with hundreds of short staple strands.",
    "Tiny fact — A piconewton is a trillionth of a newton, yet it is a practical unit for measuring molecular forces.",
    "Tiny fact — The same four DNA bases can encode both genes and programmable nanoscale geometry.",
    "Tiny fact — A colloid can look uniform to the eye while containing millions of particles large enough to scatter light.",
    "Tiny fact — An alpha helix makes one complete turn about every 3.6 amino-acid residues.",
    "Tiny fact — Double-stranded DNA has a persistence length of roughly 50 nanometres under typical physiological conditions.",
    "Tiny fact — Fluorescence lifetimes are measured in nanoseconds, and they report on a molecule’s surroundings rather than on how bright it looks.",
    "Tiny fact — A protein structure is a useful pose, not a frozen biography; proteins keep moving as they work.",
    "Tiny fact — An atomic-force microscope maps a surface by feeling it with a tiny cantilever instead of illuminating it.",
)

_TEA_JOKES = (
    "Lab joke — Reviewer 2 requested one small experiment. It has now developed its own grant proposal.",
    "Lab joke — Reviewer 1 loved it. Reviewer 3 loved it. Reviewer 2 is why we are all here today.",
    "Lab joke — The null hypothesis was not rejected; it has asked to remain anonymous.",
    "Lab joke — The p-value came in at 0.051, so we called it a trend and walked past it briskly.",
    "Lab joke — Good news: the result is fully reproducible. Bad news: so is the artifact.",
    "Lab joke — The pilot experiment worked on the very first try, which is how we knew something was wrong.",
    "Lab joke — The error bars are small because we only kept the runs where they were.",
    "Lab joke — My data shows a beautiful trend. It also has three points, two of which are the same point.",
    "Lab joke — The instrument is perfectly fine. It only makes that noise while being observed.",
    "Lab joke — The cluster job ran for six days and returned a single number. The number was “nan”.",
    "Lab joke — I automated the analysis to save time, and I have been maintaining the script ever since.",
    "Lab joke — The freezer inventory says the sample is in box 7. Box 7 disagrees, and box 7 has seniority.",
    "Lab joke — The buffer is fresh, in the sense that somebody made it once, at some point.",
    "Lab joke — We named the mutant after what it does. It has since stopped doing that.",
    "Lab joke — I finally read the supplementary information. The paper was in there.",
    "Lab joke — Nothing sharpens a literature review like finding the study you planned to run, published in 2011.",
    "Lab joke — The microscope found a perfect field of view. It was on the neighbouring slide.",
    "Lab joke — My control experiment worked beautifully. Unfortunately it controlled the hypothesis out of existence.",
    "Lab joke — The gel had excellent bands. The interpretation immediately went off-lane.",
    "Lab joke — The assay was robust right up until somebody else tried it.",
    "Lab joke — The methods section said “briefly.” The lab remembers it differently.",
    "Lab joke — The protocol said mix gently. The vortex mixer took that personally.",
    "Lab joke — The sample was at room temperature. The room declined to specify which temperature.",
    "Lab joke — My notebook and I agree on everything except what I wrote yesterday.",
    "Lab joke — The deadline improved my writing enormously: it removed every paragraph I could not defend.",
    "Lab joke — I was told to keep the talk high level. I have now explained the whole field on one slide and nothing on the other forty.",
    "Lab joke — Every simulation converges eventually. Mine is exploring the alternatives first.",
    "Lab joke — The benchmark was state of the art until the test set met the training set.",
    "Lab joke — I asked the model for its confidence. It gave six decimal places and changed the subject.",
    "Lab joke — I told the model to be concise. It wrote a paragraph explaining that it would be concise.",
    "Lab joke — In silico experiments never spill, but they can still leak data.",
    "Lab joke — The colloid insisted it was stable. Gravity asked for a longer time course.",
    "Lab joke — DNA origami is molecular flat-pack furniture, except the staples really are included.",
)

TEA_NOTE_BANK = _TEA_FACTS + _TEA_JOKES


def _daily_pick(entries: tuple[str, ...], count: int, day: date, kind: str) -> list[str]:
    ordered = sorted(
        entries,
        key=lambda note: sha256(f"{kind}|v1|{note}".encode()).digest(),
    )
    # Advance by one full deck each day. Each bank is at least twice its daily
    # draw size, so consecutive days cannot repeat a card.
    start = (day.toordinal() * count) % len(ordered)
    return [ordered[(start + offset) % len(ordered)] for offset in range(count)]


def daily_tea_deck(day: date) -> tuple[str, ...]:
    """Return the day's stable deck: ten jokes and five facts, shown one at a time."""
    jokes = _daily_pick(_TEA_JOKES, DAILY_JOKES, day, "joke")
    facts = _daily_pick(_TEA_FACTS, DAILY_FACTS, day, "fact")

    # Two jokes, then a fact, so the facts stay a garnish rather than a lecture.
    interleaved: list[str] = []
    for index, fact in enumerate(facts):
        interleaved.extend(jokes[index * 2 : index * 2 + 2])
        interleaved.append(fact)

    # Rotate the interleaved deck so the first card is not always the same kind.
    offset = sha256(f"order|{day.isoformat()}".encode()).digest()[0] % len(interleaved)
    ordered = interleaved[offset:] + interleaved[:offset]
    return tuple(ordered)
