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
    "Tiny fact — A DNA origami staple set arrives as ordinary oligos in a plastic plate, so an object with nanometre features comes by post.",
    "Tiny fact — Magnesium screens the phosphate backbone's charge; without it a folded origami simply pushes itself apart.",
    "Tiny fact — Blunt-end stacking, not base pairing, is what makes separate origami tiles polymerise without forming a single new bond.",
    "Tiny fact — Strand displacement needs no enzyme: a toehold of six or seven bases lets one strand pry another off.",
    "Tiny fact — DNA computing has solved Hamiltonian path problems in a test tube, using hybridisation itself as the search.",
    "Tiny fact — The nuclear pore passes small molecules freely and gates large ones with no moving door — the barrier is a mesh of disordered protein.",
    "Tiny fact — Intrinsically disordered regions make up roughly a third of the human proteome and were long dismissed as unfoldable junk.",
    "Tiny fact — Liquid-liquid phase separation builds a compartment with no membrane at all, held together only by weak multivalent contacts.",
    "Tiny fact — AlphaFold's pLDDT confidence tracks disorder: a low score often means the region genuinely has no fixed structure.",
    "Tiny fact — Protein design became tractable when the field stopped predicting structure from sequence and started choosing a structure to search a sequence for.",
    "Tiny fact — A designed protein can out-stabilise almost anything natural, because evolution never optimised for boiling.",
    "Tiny fact — Directed evolution won a Nobel Prize for deliberately doing what nature does slowly: mutate, select, repeat.",
    "Tiny fact — A ribozyme can be evolved in vitro in weeks, compressing a search that took biology millions of years.",
    "Tiny fact — Lipid nanoparticles deliver mRNA by escaping the endosome, and that escape step is still only a few percent efficient.",
    "Tiny fact — Ionisable lipids are neutral in blood and charged in the endosome — a pH switch that made mRNA vaccines practical.",
    "Tiny fact — Codon choice changes how fast a ribosome moves and therefore how a protein folds, so a synonymous change is not always silent.",
    "Tiny fact — A typical human protein has a half-life of hours to days: the proteome is rebuilt constantly, not maintained.",
    "Tiny fact — Molecular dynamics simulations still spend most of their compute on water.",
    "Tiny fact — A microsecond of all-atom simulation was a landmark result in 2000 and is now an afternoon.",
    "Tiny fact — Force fields are fitted rather than derived, so a simulation is an interpolation wearing the clothes of physics.",
    "Tiny fact — Single-molecule FRET reports distances from two to ten nanometres, a window that happens to match most protein conformational changes.",
    "Tiny fact — Super-resolution microscopy beats the diffraction limit by never imaging two nearby molecules in the same instant.",
    "Tiny fact — Expansion microscopy gains resolution by physically swelling the sample in a hydrogel instead of improving the optics.",
    "Tiny fact — A DNA synthesiser writes about two hundred bases reliably; everything longer is assembled from those pieces.",
    "Tiny fact — Nanopore sequencing reads a strand by measuring the ionic current it blocks, making the signal electrical rather than optical.",
    "Tiny fact — The cost of reading a genome fell faster than Moore's law for a decade, which is why the bottleneck moved to interpretation.",
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
    "Lab joke — The grant was funded, then the budget was cut by forty percent, so we will do all of the aims at half the speed and call it focus.",
    "Lab joke — My no-cost extension has now outlived two lab members and one instrument.",
    "Lab joke — The PI said we should write this up. That was the entire experimental design.",
    "Lab joke — Our collaboration has a shared drive, a shared vision, and no shared file-naming convention.",
    "Lab joke — Someone found a bug in the analysis. The bug turned out to be load-bearing.",
    "Lab joke — We pre-registered the hypothesis. The data pre-registered its objection.",
    "Lab joke — My figure has seven panels because the story has one point and six caveats.",
    "Lab joke — Panel A took two years. Panel B took two hours and is the one everyone cites.",
    "Lab joke — The reviewer asked for a mechanism. We had a correlation and a diagram with arrows.",
    "Lab joke — Every arrow in a schematic conceals at least one entire PhD.",
    "Lab joke — Rejected without review, which is the fastest feedback this project has ever received.",
    "Lab joke — Desk rejection: all the speed of peer review with none of the peer.",
    "Lab joke — The preprint has four thousand downloads and one comment, which is a typo correction.",
    "Lab joke — We posted the preprint at two in the morning so nobody would read the acknowledgements.",
    "Lab joke — The journal's impact factor is twelve. The paper's impact is my mother reading the abstract.",
    "Lab joke — The first citation arrived today. It was us.",
    "Lab joke — The h-index mostly measures how long you have been alive.",
    "Lab joke — I gave a talk to forty people. Two were awake and one was the next speaker.",
    "Lab joke — Question time opened with more of a comment than a question, and the room quietly aged.",
    "Lab joke — The poster session had free wine, which remains the only known treatment for a poster session.",
    "Lab joke — My poster was in the far corner by the fire exit, which at least made leaving convenient.",
    "Lab joke — The conference Wi-Fi held up beautifully right until the first live demo.",
    "Lab joke — The demo worked in the hotel room. The hotel room is now a co-author.",
    "Lab joke — The keynote showed data from 1998 and it was still better than mine.",
    "Lab joke — My talk was scheduled immediately after lunch, the hour when science goes to die.",
    "Lab joke — I networked at the conference, meaning I stood near the coffee and made eye contact with the sugar.",
    "Lab joke — The cells look happy. The cells have never looked happy; I have simply lowered my standards.",
    "Lab joke — Passage forty-seven has developed a personality and I no longer trust it.",
    "Lab joke — Contamination is nature's way of asking whether you really needed that experiment.",
    "Lab joke — The incubator held 37 degrees, give or take one student leaving the door open.",
    "Lab joke — The animal work was approved, budgeted and scheduled. The animals were not consulted.",
    "Lab joke — The knockout has no phenotype, which we will be describing as surprisingly robust compensation.",
    "Lab joke — We cloned it in a week and sequenced it in a week, and it was wrong in both weeks.",
    "Lab joke — The plasmid map and the plasmid have agreed to differ.",
    "Lab joke — The assembly worked on the first attempt, so I have not slept, waiting for the catch.",
    "Lab joke — One clean band, which guarantees it will fail tomorrow for reasons nobody will establish.",
    "Lab joke — The negative control has a band. We are all going home.",
    "Lab joke — The column ran beautifully and eluted absolutely nothing.",
    "Lab joke — The protein expressed magnificently, entirely into inclusion bodies.",
    "Lab joke — My protein is soluble at every concentration too low to be useful.",
    "Lab joke — The crystal diffracted to six angstroms, enough to confirm we have a protein and nothing else.",
    "Lab joke — The spectrum is beautifully clean if you ignore the region containing the answer.",
    "Lab joke — The mass spec found keratin. The mass spec always finds keratin.",
    "Lab joke — The plate reader returns ninety-six wells of hope and one gradient of evaporation.",
    "Lab joke — Row H is always strange. Nobody knows why. Row H knows why.",
    "Lab joke — The autoclave is running, so the whole floor now smells like a decision.",
    "Lab joke — Somebody has taken the good pipette. There is a list, and there will be a reckoning.",
    "Lab joke — The minus-eighty alarm went off at three in the morning, as it always has and always will.",
    "Lab joke — The dewar is full of everything except a record of what is in it.",
    "Lab joke — I ordered the reagent in March. It arrives in November. The project ended in July.",
    "Lab joke — The quote was in euros, the budget in dollars, and the finance office in another dimension.",
    "Lab joke — Purchasing rejected the order because the vendor's name contained an ampersand.",
    "Lab joke — The code runs. The code has always run. What the code does is a separate research question.",
    "Lab joke — The results reproduce on my machine, which is now formally a piece of scientific equipment.",
    "Lab joke — I containerised the pipeline so that it can now fail identically everywhere.",
    "Lab joke — Version control tells me the truth about what I did, which is why I sometimes avoid it.",
    "Lab joke — The commit message says fix. It changed four hundred lines and the conclusion.",
    "Lab joke — The random seed was 42 and, for one glorious afternoon, so was the result.",
    "Lab joke — We tuned the hyperparameters until the model agreed with us. This is called domain knowledge.",
    "Lab joke — The ablation removed the component and performance improved, so there will be no ablation section.",
    "Lab joke — The model is interpretable in the sense that we have a heatmap and a confident tone.",
    "Lab joke — The benchmark saturated, so the field built a harder benchmark and saturated that too.",
    "Lab joke — The model drafted the related work and cited three papers that would have been excellent.",
    "Lab joke — I asked it to check my algebra. It agreed with me warmly, and incorrectly.",
    "Lab joke — Our uncertainty estimates are well calibrated on the data where we measured the calibration.",
    "Lab joke — The dataset is public, the labels are proprietary, and the ground truth is in somebody's notebook.",
    "Lab joke — It self-assembled overnight while nobody watched, which remains our most reliable protocol.",
    "Lab joke — It folds perfectly at two hundred millimolar magnesium, a condition found nowhere in biology.",
    "Lab joke — The micrograph shows exactly what we designed, in the four fields of view we are showing you.",
    "Lab joke — The shift assay says it bound something. The something remains at large.",
    "Lab joke — The yield is eighty percent if you define yield generously and eight percent if you do not.",
    "Lab joke — Twenty-four hours to anneal, three minutes to ruin it at the bench.",
    "Lab joke — Our construct is stable in buffer, stable in serum, and extremely stable in the introduction.",
    "Lab joke — The instrument was serviced last week, so it is now broken in an entirely new way.",
    "Lab joke — I wrote the discussion first, which is how I discovered what the paper was about.",
    "Lab joke — The limitations paragraph is where I keep the experiments I could not afford.",
    "Lab joke — Statistics is the art of being uncertain with confidence intervals.",
    "Lab joke — We powered the study for the effect we wanted rather than the effect that exists.",
    "Lab joke — The outlier was removed for technical reasons, the technical reason being that it was inconvenient.",
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


def _least_recently_shown(
    entries: tuple[str, ...],
    count: int,
    last_shown: dict[str, str],
    kind: str,
) -> list[str]:
    """Draw the *count* cards the reader has not seen for the longest.

    Never-shown cards sort first (their last-shown day is the empty string), and
    the rest sort by the day they last appeared. A card therefore cannot return
    until every other card in its bank has had a turn, which turns the rotation
    from a fixed few-day loop into a full pass over the bank. Ties break on a
    stable hash so the order does not wander between processes.
    """
    return sorted(
        entries,
        key=lambda note: (
            last_shown.get(note, ""),
            sha256(f"{kind}|v1|{note}".encode()).digest(),
        ),
    )[:count]


def _compose_deck(jokes: list[str], facts: list[str], day: date) -> tuple[str, ...]:
    # Two jokes, then a fact, so the facts stay a garnish rather than a lecture.
    interleaved: list[str] = []
    for index, fact in enumerate(facts):
        interleaved.extend(jokes[index * 2 : index * 2 + 2])
        interleaved.append(fact)
    if not interleaved:
        return ()
    # Rotate the interleaved deck so the first card is not always the same kind.
    offset = sha256(f"order|{day.isoformat()}".encode()).digest()[0] % len(interleaved)
    return tuple(interleaved[offset:] + interleaved[:offset])


def daily_tea_deck(day: date) -> tuple[str, ...]:
    """Return the day's stable deck: ten jokes and five facts, shown one at a time.

    The deck is drawn from the cards least recently served and then recorded, so
    Pip works through the whole bank before repeating anything. Previously the
    deck was a pure function of the date, which cycled a 57-card bank every few
    days -- by day 5 every card on offer had already been seen. If the store is
    unavailable (tests, ad-hoc use) this falls back to the old date-derived
    rotation rather than failing.
    """
    try:
        from .store import record_tea_deck, tea_deck_for_day, tea_note_last_shown

        served = tea_deck_for_day(day.isoformat())
        if served:
            return tuple(served)
        last_shown = tea_note_last_shown()
        deck = _compose_deck(
            _least_recently_shown(_TEA_JOKES, DAILY_JOKES, last_shown, "joke"),
            _least_recently_shown(_TEA_FACTS, DAILY_FACTS, last_shown, "fact"),
            day,
        )
        if deck:
            record_tea_deck(day.isoformat(), list(deck))
            return deck
    except Exception:  # noqa: BLE001 - Pip must never break the page
        pass

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
