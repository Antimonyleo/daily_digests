"""Curated, dependency-free science notes for the reader's daily tea break."""

from __future__ import annotations

from datetime import date
from hashlib import sha256

DAILY_TEA_DECK_SIZE = 15

_TEA_FACTS = (
    "Tiny fact — The ribosome’s catalytic center is RNA, so every protein begins with an RNA-powered reaction.",
    "Tiny fact — DNA is about two nanometres wide; roughly 50,000 double helices could fit across a 0.1 mm human hair.",
    "Tiny fact — An ideal PCR doubles its target each cycle; 30 cycles can turn one molecule into more than a billion copies.",
    "Tiny fact — One microlitre fits inside a cube just one millimetre wide—every pipette dispenses tiny geometry.",
    "Tiny fact — Taq polymerase comes from Thermus aquaticus, a bacterium first isolated from a hot spring.",
    "Tiny fact — Some RNA molecules are enzymes. These catalytic RNAs are called ribozymes.",
    "Tiny fact — At the nanoscale, Brownian motion helps self-assembling particles explore possible arrangements.",
    "Tiny fact — DNA origami usually folds one long scaffold with hundreds of short staple strands.",
    "Tiny fact — Green fluorescent protein came from the jellyfish Aequorea victoria and became a molecular lantern for living cells.",
    "Tiny fact — A piconewton is a trillionth of a newton, yet it is a practical unit for measuring molecular forces.",
    "Tiny fact — The same four DNA bases can encode both genes and programmable nanoscale geometry.",
    "Tiny fact — Messenger RNA is read three bases at a time; 64 codons encode 20 standard amino acids and stop signals.",
    "Tiny fact — Restriction enzymes evolved in bacteria as defenses against invading genetic material.",
    "Tiny fact — A colloid can look uniform to the eye while containing millions of particles large enough to scatter light.",
    "Tiny fact — Single-particle cryo-EM combines images of many frozen particles in different orientations into a three-dimensional reconstruction.",
    "Tiny fact — An alpha helix makes one complete turn about every 3.6 amino-acid residues.",
    "Tiny fact — Double-stranded DNA has a persistence length of roughly 50 nanometres under typical physiological conditions.",
    "Tiny fact — A red blood cell is roughly 7,000–8,000 nanometres wide.",
    "Tiny fact — Fluorescence lifetimes are usually measured in nanoseconds rather than by how bright a sample appears.",
    "Tiny fact — RNA uses uracil where DNA uses thymine, but both bases pair with adenine.",
    "Tiny fact — A protein structure is a useful pose, not a frozen biography; proteins keep moving as they work.",
    "Tiny fact — An atomic-force microscope can map a surface by feeling it with a tiny cantilever instead of illuminating it.",
)

_TEA_JOKES = (
    "Lab joke — Why did the protein fold? The unfolded state was making everyone energetically uncomfortable.",
    "Lab joke — My control experiment worked beautifully. Unfortunately, it controlled the hypothesis out of existence.",
    "Lab joke — The centrifuge and I have an agreement: it spins, and I pretend the pellet was always part of the plan.",
    "Lab joke — Reviewer 2 requested one small experiment. It has now developed its own grant proposal.",
    "Lab joke — The colloid said it was stable. Gravity asked for a longer time course.",
    "Lab joke — DNA origami is molecular flat-pack furniture, except the staples really are included.",
    "Lab joke — I asked the model for confidence. It returned six decimal places and changed the subject.",
    "Lab joke — A pipette’s favorite unit is the microlitre: commitment, but only in very small amounts.",
    "Lab joke — The gel had excellent bands. The interpretation immediately went off-lane.",
    "Lab joke — The null hypothesis was not rejected; it has asked to remain anonymous.",
    "Lab joke — RNA walked into a ribosome and left translated.",
    "Lab joke — The assay was robust right up until someone tried to reproduce it.",
    "Lab joke — My notebook and I agree on everything except what I wrote yesterday.",
    "Lab joke — The methods section said “briefly.” The lab remembers it differently.",
    "Lab joke — The sample was at room temperature. The room declined to specify which temperature.",
    "Lab joke — In silico experiments never spill, but they can still leak data.",
    "Lab joke — The benchmark was state of the art until the test set met the training set.",
    "Lab joke — The protocol said mix gently. The vortex mixer took that personally.",
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
    """Return the day's stable deck: ten facts and five jokes, shown one at a time."""
    facts = _daily_pick(_TEA_FACTS, 10, day, "fact")
    jokes = _daily_pick(_TEA_JOKES, 5, day, "joke")

    interleaved: list[str] = []
    for index, joke in enumerate(jokes):
        interleaved.extend(facts[index * 2 : index * 2 + 2])
        interleaved.append(joke)

    # Rotate the interleaved deck so the first card is not always the same kind.
    offset = sha256(f"order|{day.isoformat()}".encode()).digest()[0] % len(interleaved)
    ordered = interleaved[offset:] + interleaved[:offset]
    return tuple(ordered)
