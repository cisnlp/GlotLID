"""
Generate the synthetic und_{Script} data of GlotLID: lines of random characters of a script, for scripts that have
too little text in any language. Each line has 1 to 100 characters drawn uniformly from the script's code points,
with 0 to 10 spaces inserted at random positions.

The code point ranges come from GlotScript (pip install "GlotScript>=2.1"), so new Unicode scripts are covered when
GlotScript is updated.

    python gen_und.py                                  # every script, 10,000 lines each, into und/
    python gen_und.py --scripts Jurc Pcun Seal --seed 0
    python gen_und.py --letters                        # the small letter sets of und_Latn, und_Arab, und_Cyrl, ...
"""
import argparse
import os
import random

from GlotScript import SCRIPT_RANGES, UNICODE_VERSION

# Pseudo-scripts of Unicode (inherited, common, unknown): no random text of their own.
SKIP = {"Zinh", "Zyyy", "Zzzz"}

# Small sets of everyday letters, for the earlier und_{Latn,Cyrl,Grek,Deva,Arab,Hani} files (GlotLID v2/v3).
LETTERS = {
    "Latn": "abcdefghijklmnopqrstuvwxyz",
    "Cyrl": "абвгдеёжзийклмнопрстуфхцчшщъыьэюя",
    "Grek": "αβγδεζηθικλμνξοπρστυφχψω",
    "Deva": "अआइईउऊऋएऐओऔकखगघङचछजझञट",
    "Arab": "ا ب ت ث ج ح خ د ذ ر ز س ش ص ض ط ظ ع غ ف ق ك ل م ن ه و ي",
    "Hani": "的一是不了人我在有他这为之大来以个中上们说国和时要就出会可也你年得生能还就道他们学都很",
}


def letter_ranges(characters):
    """Code point ranges of the characters of a string (consecutive code points merged)."""
    cps = [ord(c) for c in characters]
    ranges = [[cps[0], cps[0]]]
    for cp in cps[1:]:
        if cp == ranges[-1][1] + 1:
            ranges[-1][1] = cp
        else:
            ranges.append([cp, cp])
    return [tuple(r) for r in ranges]


def code_points(ranges):
    return [cp for start, end in ranges for cp in range(start, end + 1)]


def generate_sentences(ranges, num_sentences, rng, min_size_sent=1, max_size_sent=100, min_num_space=0,
                       max_num_space=10):
    """Random lines: min_size_sent..max_size_sent characters of the ranges, with min_num_space..max_num_space spaces
    inserted at random positions (never more spaces than characters)."""
    numbers = code_points(ranges)
    sentences = []
    for _ in range(num_sentences):
        chars = [chr(rng.choice(numbers)) for _ in range(rng.randint(min_size_sent, max_size_sent))]
        num_spaces = min(rng.randint(min_num_space, max_num_space), len(chars))
        for pos in sorted(rng.sample(range(len(chars)), num_spaces), reverse=True):
            chars.insert(pos, " ")
        sentences.append("".join(chars))
    return sentences


def save(lines, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scripts", nargs="*", help="ISO 15924 codes (default: every script of GlotScript)")
    ap.add_argument("--num-sentences", type=int, default=10_000, help="lines per script (default 10,000)")
    ap.add_argument("--out", default="und", help="output folder (default und/)")
    ap.add_argument("--seed", type=int, help="random seed, for reproducible files")
    ap.add_argument("--letters", action="store_true",
                    help="also write the small-letter-set files: und_{Latn,Cyrl,Grek,Deva,Arab,Hani}_randomhumanreadble "
                         "(100,000 lines), und_Latn_randomhumanreadblesmall (50,000), und_{Arab,Cyrl}_randomshort (100,000)")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    scripts = a.scripts if a.scripts else [s for s in SCRIPT_RANGES if s not in SKIP]
    unknown = [s for s in scripts if s not in SCRIPT_RANGES]
    if unknown:
        raise SystemExit("not a script of GlotScript (Unicode %s): %s" % (UNICODE_VERSION, ", ".join(unknown)))
    for s in scripts:
        path = os.path.join(a.out, "und_%s_randomhumanreadble.txt" % s)
        save(generate_sentences(SCRIPT_RANGES[s], a.num_sentences, rng), path)
        print("wrote", path)
    if a.letters:
        letters = {s: letter_ranges(c) for s, c in LETTERS.items()}
        for s, ranges in letters.items():
            save(generate_sentences(ranges, 100_000, rng), os.path.join(a.out, "und_%s_randomhumanreadble.txt" % s))
        save(generate_sentences(letters["Latn"], 50_000, rng), os.path.join(a.out, "und_Latn_randomhumanreadblesmall.txt"))
        for s in ("Arab", "Cyrl"):
            save(generate_sentences(letters[s], 100_000, rng), os.path.join(a.out, "und_%s_randomshort.txt" % s))
        print("wrote the small-letter-set files to", a.out)


if __name__ == "__main__":
    main()
