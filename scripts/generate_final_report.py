from __future__ import annotations

import html
import mimetypes
import os
import re
import struct
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "final_report.odt"


def esc(text: object) -> str:
    return html.escape(str(text), quote=True)


def words(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))


def png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as f:
        sig = f.read(24)
    if sig[:8] != b"\x89PNG\r\n\x1a\n":
        return (1200, 675)
    return struct.unpack(">II", sig[16:24])


def p(text: str, style: str = "P1") -> str:
    return f'<text:p text:style-name="{style}">{esc(text)}</text:p>'


def h(text: str, level: int = 1) -> str:
    style = "Heading1" if level == 1 else "Heading2"
    return f'<text:h text:style-name="{style}" text:outline-level="{level}">{esc(text)}</text:h>'


def caption(text: str) -> str:
    return p(text, "Caption")


def table(rows: list[list[object]], name: str) -> str:
    out = [f'<table:table table:name="{esc(name)}">']
    for r_i, row in enumerate(rows):
        out.append("<table:table-row>")
        for cell in row:
            style = "TableHeader" if r_i == 0 else "TableCell"
            out.append(f'<table:table-cell office:value-type="string"><text:p text:style-name="{style}">{esc(cell)}</text:p></table:table-cell>')
        out.append("</table:table-row>")
    out.append("</table:table>")
    return "\n".join(out)


def image(path: Path, idx: int, title: str, width_in: float = 6.3) -> str:
    w, h_px = png_size(path)
    height_in = min(width_in * h_px / max(w, 1), 3.8)
    href = f"Pictures/{path.name}"
    return (
        f'<text:p text:style-name="P1"><draw:frame draw:name="fig{idx}" text:anchor-type="paragraph" '
        f'svg:width="{width_in:.2f}in" svg:height="{height_in:.2f}in" draw:z-index="{idx}">'
        f'<draw:image xlink:href="{href}" xlink:type="simple" xlink:show="embed" xlink:actuate="onLoad"/>'
        f'</draw:frame></text:p>\n{caption(title)}'
    )


figures = [
    ("working_files/train_harakat_adder/train_harakat_adder_files/harakat_loss_error.png", "Fig. 1. Harakat-adder training and validation loss/error over epochs."),
    ("working_files/train_harakat_adder/train_harakat_adder_files/harakat_sequence_error.png", "Fig. 2. Harakat-adder sequence accuracy over epochs."),
    ("working_files/train_harakat_adder/train_harakat_adder_files/harakat_learning_rate.png", "Fig. 3. Harakat-adder learning-rate schedule."),
    ("working_files/train_arabizi_to_arabic/train_arabizi_to_arabic_files/train_arabizi_to_arabic_loss_error.png", "Fig. 4. Arabizi-to-Arabic loss and error trends."),
    ("working_files/train_arabizi_to_arabic/train_arabizi_to_arabic_files/train_arabizi_to_arabic_sequence_error.png", "Fig. 5. Arabizi-to-Arabic sequence accuracy trend."),
    ("working_files/train_arabizi_to_arabic/train_arabizi_to_arabic_files/train_arabizi_to_arabic_top_k_accuracy.png", "Fig. 6. Arabizi-to-Arabic softmax-90 top-k accuracy."),
    ("working_files/train_arabizi_to_arabic/train_arabizi_to_arabic_files/train_arabizi_to_arabic_error_by_source_length.png", "Fig. 7. Arabizi-to-Arabic error rate by source length."),
    ("working_files/train_arabizi_to_arabic/train_arabizi_to_arabic_files/train_arabizi_to_arabic_error_by_target_length.png", "Fig. 8. Arabizi-to-Arabic error rate by target length."),
    ("working_files/train_arabic_w_harakat_to_arabizi/train_arabic_w_harakat_to_arabizi_files/train_arabic_w_harakat_to_arabizi_loss_error.png", "Fig. 9. Original-harakat-to-Arabizi loss and error trends."),
    ("working_files/train_arabic_w_harakat_to_arabizi/train_arabic_w_harakat_to_arabizi_files/train_arabic_w_harakat_to_arabizi_sequence_error.png", "Fig. 10. Original-harakat-to-Arabizi sequence accuracy trend."),
    ("working_files/train_arabic_w_harakat_to_arabizi/train_arabic_w_harakat_to_arabizi_files/train_arabic_w_harakat_to_arabizi_top_k_accuracy.png", "Fig. 11. Original-harakat-trained model softmax-90 stress test on stripped input."),
    ("working_files/train_arabic_w_harakat_to_arabizi/train_arabic_w_harakat_to_arabizi_files/train_arabic_w_harakat_to_arabizi_error_by_source_length.png", "Fig. 12. Original-harakat-to-Arabizi error rate by source length."),
    ("working_files/train_arabic_w_harakat_to_arabizi/train_arabic_w_harakat_to_arabizi_files/train_arabic_w_harakat_to_arabizi_error_by_target_length.png", "Fig. 13. Original-harakat-to-Arabizi error rate by target length."),
    ("working_files/train_arabic_wo_harakat_to_arabizi/train_arabic_wo_harakat_to_arabizi_files/train_arabic_wo_harakat_to_arabizi_files/train_arabic_wo_harakat_to_arabizi_loss_error.png", "Fig. 14. No-harakat-to-Arabizi loss and error trends."),
    ("working_files/train_arabic_wo_harakat_to_arabizi/train_arabic_wo_harakat_to_arabizi_files/train_arabic_wo_harakat_to_arabizi_files/train_arabic_wo_harakat_to_arabizi_sequence_error.png", "Fig. 15. No-harakat-to-Arabizi sequence accuracy trend."),
    ("working_files/train_arabic_wo_harakat_to_arabizi/train_arabic_wo_harakat_to_arabizi_files/train_arabic_wo_harakat_to_arabizi_files/train_arabic_wo_harakat_to_arabizi_top_k_accuracy.png", "Fig. 16. No-harakat-to-Arabizi softmax-90 top-k accuracy."),
    ("working_files/train_arabic_wo_harakat_to_arabizi/train_arabic_wo_harakat_to_arabizi_files/train_arabic_wo_harakat_to_arabizi_files/train_arabic_wo_harakat_to_arabizi_error_by_source_length.png", "Fig. 17. No-harakat-to-Arabizi error rate by source length."),
    ("working_files/train_arabic_wo_harakat_to_arabizi/train_arabic_wo_harakat_to_arabizi_files/train_arabic_wo_harakat_to_arabizi_files/train_arabic_wo_harakat_to_arabizi_error_by_target_length.png", "Fig. 18. No-harakat-to-Arabizi error rate by target length."),
    ("working_files/train_harakat_top_5_to_arabizi/train_harakat_top_5_to_arabizi_files/train_harakat_top_5_to_arabizi_files/train_harakat_top_5_to_arabizi_loss_error.png", "Fig. 19. Harakat-top-5-to-Arabizi loss and error trends."),
    ("working_files/train_harakat_top_5_to_arabizi/train_harakat_top_5_to_arabizi_files/train_harakat_top_5_to_arabizi_files/train_harakat_top_5_to_arabizi_sequence_error.png", "Fig. 20. Harakat-top-5-to-Arabizi sequence accuracy trend."),
    ("working_files/train_harakat_top_5_to_arabizi/train_harakat_top_5_to_arabizi_files/train_harakat_top_5_to_arabizi_files/train_harakat_top_5_to_arabizi_top_k_accuracy.png", "Fig. 21. Harakat-top-5-to-Arabizi softmax-90 top-k accuracy."),
    ("working_files/train_harakat_top_5_to_arabizi/train_harakat_top_5_to_arabizi_files/train_harakat_top_5_to_arabizi_files/train_harakat_top_5_to_arabizi_error_by_source_length.png", "Fig. 22. Harakat-top-5-to-Arabizi error rate by source length."),
    ("working_files/train_harakat_top_5_to_arabizi/train_harakat_top_5_to_arabizi_files/train_harakat_top_5_to_arabizi_files/train_harakat_top_5_to_arabizi_error_by_target_length.png", "Fig. 23. Harakat-top-5-to-Arabizi error rate by target length."),
    ("working_files/train_harakat_top_1_to_arabizi/train_harakat_top_1_to_arabizi_files/train_harakat_top_1_to_arabizi_loss_error.png", "Fig. 24. Harakat-top-1-to-Arabizi loss and error trends."),
    ("working_files/train_harakat_top_1_to_arabizi/train_harakat_top_1_to_arabizi_files/train_harakat_top_1_to_arabizi_sequence_error.png", "Fig. 25. Harakat-top-1-to-Arabizi sequence accuracy trend."),
    ("working_files/train_harakat_top_1_to_arabizi/train_harakat_top_1_to_arabizi_files/train_harakat_top_1_to_arabizi_top_k_accuracy.png", "Fig. 26. Harakat-top-1-to-Arabizi softmax-90 top-k accuracy."),
    ("working_files/train_harakat_top_1_to_arabizi/train_harakat_top_1_to_arabizi_files/train_harakat_top_1_to_arabizi_error_by_source_length.png", "Fig. 27. Harakat-top-1-to-Arabizi error rate by source length."),
    ("working_files/train_harakat_top_1_to_arabizi/train_harakat_top_1_to_arabizi_files/train_harakat_top_1_to_arabizi_error_by_target_length.png", "Fig. 28. Harakat-top-1-to-Arabizi error rate by target length."),
    ("working_files/train_arabic_w_levanti_harakat_to_arabizi/train_arabic_w_levanti_harakat_to_arabizi_files/train_arabic_w_levanti_harakat_to_arabizi_loss_error.png", "Fig. 29. Levanti-harakat-to-Arabizi loss and error trends."),
    ("working_files/train_arabic_w_levanti_harakat_to_arabizi/train_arabic_w_levanti_harakat_to_arabizi_files/train_arabic_w_levanti_harakat_to_arabizi_sequence_error.png", "Fig. 30. Levanti-harakat-to-Arabizi sequence accuracy trend."),
    ("working_files/train_arabic_w_levanti_harakat_to_arabizi/train_arabic_w_levanti_harakat_to_arabizi_files/train_arabic_w_levanti_harakat_to_arabizi_top_k_accuracy.png", "Fig. 31. Levanti-harakat-to-Arabizi softmax-90 top-k accuracy."),
    ("working_files/train_arabic_w_levanti_harakat_to_arabizi/train_arabic_w_levanti_harakat_to_arabizi_files/train_arabic_w_levanti_harakat_to_arabizi_error_by_source_length.png", "Fig. 32. Levanti-harakat-to-Arabizi error rate by source length."),
    ("working_files/train_arabic_w_levanti_harakat_to_arabizi/train_arabic_w_levanti_harakat_to_arabizi_files/train_arabic_w_levanti_harakat_to_arabizi_error_by_target_length.png", "Fig. 33. Levanti-harakat-to-Arabizi error rate by target length."),
]

body: list[str] = []
body.append(p("A Graduate Project White Paper on Palestinian Arabizi and Arabic Transliteration", "Title"))
body.append(p("Deep Learning 535 Final Report", "Subtitle"))
body.append(p("Executive Summary--This white paper frames Palestinian Arabizi and Arabic-script transliteration as an asymmetric problem. The Arabizi-to-Arabic direction is difficult, but it is mostly a normalization task: the model sees the Latin letters and numerals the user wrote and learns to map them back into Arabic script. The Arabic-to-Arabizi direction is the harder technical challenge because most written Arabic omits harakat: short vowels, vowel absence, consonant doubling, and other pronunciation cues. Human readers often infer those missing cues from context, but a short isolated input gives a model much less evidence. The output side is also underdetermined because Arabizi has no single standard spelling; the same Arabic word can be written several acceptable ways depending on the writer. From a practical intelligence and OSINT perspective, the most important metric is therefore not only strict single-best exact match. It is softmax-90 top-k coverage: whether the useful transliteration appears somewhere in a ranked candidate set that can support search, keyword expansion, analyst triage, or downstream Arabic NLP. The work trains one custom Arabizi-to-Arabic model and a sequence of Arabic-to-Arabizi experiments: original gold harakat, no harakat, custom harakat top-five, custom harakat top-one, and Levanti-predicted harakat. Evaluation supports the operational framing. Arabizi-to-Arabic reaches 75.8 percent test sequence accuracy and 91.7 percent top-10 coverage. On Arabic-to-Arabizi, the gold-harakat model reaches 83.1 percent exact test accuracy when pronunciation marks are visible, the no-harakat model drops to 47.4 percent exact test accuracy but still reaches 87.4 percent top-10 coverage, and the proof-of-concept custom top-five model reaches 98.3 percent top-10 coverage. These results show promise, but the system is not ready for full operational use as an app; it is a feasibility prototype that would need more data, broader testing, stronger validation, and engineering hardening before real deployment.", "Abstract"))
body.append(p("Keywords--Arabizi, Arabic transliteration, Palestinian Arabic, character-level Transformer, diacritization, Maknuune, top-k decoding, OSINT text normalization.", "Keywords"))

body.append(h("I. Introduction"))
intro = [
    "Arabizi is informal Arabic written with Latin letters and numerals. It is common in chat, social media, and situations where Arabic keyboard entry is inconvenient. The writing style is practical for users but difficult for software because it is not a strict one-to-one transliteration scheme. A single Latin character may correspond to several Arabic letters, digits may stand for Arabic sounds with no common Latin equivalent, vowels may be omitted or over-specified, and spelling conventions vary by dialect and by writer. These properties make Arabizi transliteration a sequence prediction problem rather than a simple lookup table.",
    "Arabizi exists for practical historical and social reasons. Many users began writing Arabic with Latin characters because older phones, keyboards, websites, games, and messaging systems did not always make Arabic-script input easy. Even when Arabic keyboards are available, some users continue to prefer Arabizi because it is fast, familiar within peer groups, or better suited to mixed Arabic-English online conversation. Numerals became part of the convention because some Arabic sounds do not have obvious Latin equivalents. For example, writers often use `3` for the sound ع and `7` for ح because the shapes or community conventions make those digits recognizable substitutes.",
    "This problem has practical importance beyond ordinary text normalization. Analysts working with Arabic-language material may encounter dialectal Arabic written in Arabizi form, especially in informal communications, web forums, messaging logs, and social media collections. In open-source intelligence (OSINT) workflows, organizations often monitor public sources for legally and ethically defined keywords, names, places, slogans, event terms, or emerging threat indicators. If those keywords are written in Arabizi rather than Arabic script, an Arabic-script keyword watchlist may miss them. Intelligence and security organizations that lawfully collect Arabic-language data may therefore receive material that Arabic-script search tools, machine translation systems, named-entity recognizers, and triage workflows cannot process directly. A reliable Arabizi-to-Arabic transliterator helps convert that material into a script representation compatible with downstream Arabic NLP. It does not solve translation or interpretation by itself, but it can make otherwise opaque Romanized Arabic text searchable, comparable, and ready for later human or machine analysis.",
    "The two directions are not equally difficult. Arabizi-to-Arabic begins with a user spelling that already contains many pronunciation choices: Latin vowels, doubled letters, and numerals often reveal what the writer intended. Arabic-to-Arabizi begins from ordinary Arabic script, which usually leaves short vowels and other harakat unwritten. That means the model must first infer pronunciation information that is not present in the characters. A human reader would use sentence context, morphology, dialect knowledge, and common sense to decide what the missing harakat should be. In this project, the model often receives only a word or short phrase, so that inference is the core challenge.",
    "The goal of this project is to build a proof-of-concept bidirectional transliteration model set for Palestinian Arabic. The system is intentionally small enough to study and run locally, but it still uses modern neural sequence modeling. For Arabizi-to-Arabic, one model maps Latin-script input to undiacritized Arabic. For Arabic-to-Arabizi, the system first predicts possible diacritized Arabic forms and then uses those candidate forms as structured input to the Arabizi generator. The important contribution is the modeling pipeline and its evaluation, not a finished user application.",
    "The design follows the observation that Arabic script often omits short vowels while Arabizi often expresses them. Direct Arabic-to-Arabizi mapping from unvocalized Arabic can therefore be under-specified. A bare Arabic consonantal form may have several plausible pronunciations, and each pronunciation may lead to more than one Romanized spelling. Even after pronunciation is inferred, there is usually no single correct Arabizi output because writers choose between spellings such as short or doubled vowels, numerals or letter combinations, and English-like or Arabic-sound-like conventions. The project handles this ambiguity by generating up to five harakat hypotheses before generating Arabizi candidates. This choice mirrors the practical behavior of a human reader: first infer likely vocalization, then render the result in a Latin-script convention.",
    "This white paper introduces the task and related work, describes the dataset, presents the prototype models plus the top-one and Levanti comparison models, gives the evaluation metrics and results, and discusses error behavior and limitations. The tone is intentionally project-oriented. The goal is not to claim a finished research contribution or a ready operational app, but to show a complete deep-learning feasibility study: data preparation, model design, training, evaluation, and model comparison. The main engineering theme is candidate preservation. From an intelligence-use perspective, high softmax-90 top-k coverage is the key practical result because it improves the chance that a relevant spelling appears in a search or analyst review set. Rather than forcing one answer when the data itself supports several spellings, the model returns ranked transliteration candidates. This makes the prototype more honest and more useful for recall-oriented workflows.",
]
body.extend(p(x) for x in intro)

body.append(h("II. Related Work"))
related = [
    "The core modeling approach is based on neural sequence-to-sequence learning. Sutskever, Vinyals, and Le showed that an encoder-decoder neural network could map variable-length input sequences to variable-length output sequences and achieve strong machine-translation results [1]. Bahdanau, Cho, and Bengio then introduced attention as a way for the decoder to focus on relevant source positions during generation [2]. The Transformer later replaced recurrent computation with multi-head attention and feed-forward layers, improving parallelism and becoming the standard architecture for many sequence transduction tasks [3]. This project uses character-level Transformer encoder-decoder models because transliteration is primarily about spelling and symbol sequences, and because the available data is word-like rather than sentence-scale.",
    "Arabizi transliteration has been studied as a dialectal Arabic NLP problem for more than a decade. Al-Badrashiny, Eskander, Habash, and Rambow describe Romanized dialectal Arabic as informal Latin-script writing that cannot be treated as ordinary letter substitution; their system generated transliteration candidates, filtered them, and selected likely Arabic-script forms [4]. Habash, Diab, and Rambow proposed CODA as a conventional orthography for dialectal Arabic, emphasizing that dialectal Arabic lacks a single naturally standardized written form [5]. These points matter directly here: the model is evaluated against one lexicon spelling, but real users may accept alternatives. That is one reason top-k evaluation is reported alongside strict exact-match accuracy.",
    "Arabic diacritization is also central to this system. Surveys of Arabic diacritization tools note that Arabic text is commonly written without short-vowel marks, even though those marks carry phonological, morphological, and sometimes semantic information [6]. Neural diacritization research has shown that character and sequence models can restore these marks effectively when enough clean supervision is available [7]. In this project, diacritization is not the final user-facing output in the Arabic-to-Arabizi direction. Instead, it is an intermediate representation that supplies possible pronunciations to the Romanization model.",
    "The implementation uses PyTorch for training and model serialization. PyTorch is designed around imperative Python execution while still supporting hardware acceleration, which makes it well suited for experimental deep-learning workflows [8]. The dataset foundation is Maknuune, a large open Palestinian Arabic lexicon with more than 36,000 entries that include diacritized Arabic orthography and phonological information [9].",
]
body.extend(p(x) for x in related)

body.append(h("III. Linguistic Background"))
linguistic = [
    "A key linguistic issue in this project is the difference between Arabic script with and without diacritics. Arabic diacritics, often called harakat, are small marks written above or below letters. They can indicate short vowels, vowel absence, consonant doubling, and other pronunciation information. For example, the consonantal skeleton of a word may stay the same while the harakat change how it is pronounced. In fully vocalized writing, these marks make pronunciation more explicit.",
    "In ordinary Arabic writing, however, most short-vowel diacritics are not written. Newspapers, social media posts, books for adult readers, and everyday messages usually present words mostly as consonants and long-vowel letters. Native and advanced readers infer the missing short vowels from context, grammar, word patterns, and world knowledge. This is natural for human readers because Arabic morphology is patterned and because sentence context usually narrows the possibilities. Children, language learners, religious texts, dictionaries, and teaching materials are more likely to include full or partial diacritics because those contexts require explicit pronunciation guidance.",
    "When diacritics are not included, a lot of information about pronunciation and sometimes meaning can be lost. The same undiacritized Arabic letter sequence can correspond to more than one vocalized word. A reader may infer the intended form from context, but a machine model seeing only a short isolated word has less context than a human reader. This is especially important for transliteration into Arabizi because Arabizi often writes vowels with Latin letters. If the Arabic input does not show the short vowels, the model must guess which vowels should appear in the Arabizi output.",
    "This missing-vowel problem is less severe in the reverse direction. When a user writes Arabizi, they often write at least some vowel information directly with Latin letters. The spelling may be inconsistent, but the model still receives evidence about pronunciation from the user's own Romanized form. In Arabic-to-Arabizi, by contrast, the model starts from a script that intentionally leaves many pronunciation choices implicit. The direction therefore requires reconstruction before transliteration: infer the likely spoken form, then choose how that spoken form might be written in Arabizi.",
    "This is why the Arabic-to-Arabizi side of the project uses a two-stage design. The first model predicts possible harakat forms from the plain Arabic input. The second model uses up to five of those vocalized hypotheses to generate Arabizi. The harakat stage is not an academic extra; it addresses a real missing-information problem. Without some estimate of the unwritten vowels and pronunciation marks, the Romanized output would be forced to invent vowel information from a stripped Arabic source.",
    "The second challenge is that Arabizi is not standardized. Several outputs can be reasonable for the same Arabic input. Writers may double vowels for emphasis or pronunciation, omit vowels when the word is obvious, choose numerals for emphatic or pharyngeal sounds, or use English-influenced spellings. A strict exact-match metric can therefore count an output wrong even when a human would understand it as a plausible spelling. This is why the report treats ranked top-k candidates as central evidence rather than as a convenience feature.",
    "This project also required building its own harakat adder rather than simply using a general Arabic diacritization model. Many available Arabic diacritizers are designed around Modern Standard Arabic (MSA) or formal written Arabic. That assumption is reasonable for news, books, religious text, or formal documents, but it is not the same as colloquial Palestinian Arabic. Palestinian Arabic differs from MSA in pronunciation, vocabulary, morphology, and common written forms. A model trained mainly on MSA may add diacritics that are grammatically plausible for standard Arabic but wrong for a Palestinian dialect word or phrase.",
    "The transliteration goal makes this mismatch especially costly. The harakat adder is not being used to make formal Arabic look complete; it is being used to infer pronunciation before generating Arabizi. If an outside MSA-oriented diacritizer inserts standard-Arabic vowels, the next model may produce Arabizi that reflects the wrong dialectal pronunciation. Training a project-specific harakat adder on the Maknuune-derived Palestinian data keeps the intermediate vocalization closer to the same dialectal distribution as the Arabizi targets. In other words, the custom harakat adder is a domain-adaptation step. It makes the Arabic-to-Arabizi pipeline internally consistent with colloquial Palestinian Arabic rather than forcing the pipeline through an MSA pronunciation layer.",
]
body.extend(p(x) for x in linguistic)

body.append(h("IV. Dataset and Preprocessing"))
dataset = [
    "The starting point is the cleaned Maknuune-derived file `maknuune-v1.0.1_cleaned.csv`, which contains 36,302 rows and three primary fields: `arabizi`, `arabic_harakat`, and `arabic_stripped`. Maknuune was used because it was the strongest available Palestinian Arabic resource found for this project: it is dialect-specific, includes Arabizi, includes diacritized Arabic forms, and is large enough to support supervised neural training. The dataset is still narrower than raw OSINT text, but for a Palestinian-focused transliteration prototype it is a better fit than a larger MSA corpus or a broad Arabic resource that does not preserve the Arabizi/harakat relationship needed here.",
    "The `arabizi` field is the Latin-script representation used as the source for Arabizi-to-Arabic training and as the target for Arabic-to-Arabizi training. The `arabic_harakat` field preserves Arabic script with diacritics. The `arabic_stripped` field removes those diacritics and is used as the plain Arabic target or source depending on direction. This compact schema supports the prototype model chain without requiring separate annotation passes.",
    "The cleaned data also shows why exact matching is an imperfect but still useful diagnostic. When grouped by undiacritized `arabic_stripped`, 5,117 of 27,626 unique Arabic forms, about 18.5 percent, map to more than one Arabizi spelling in the dataset itself. Most ambiguous forms have two variants, but hundreds have three or more. For example, `عقد` appears with variants such as `3aqad`, `3aqid`, `3uqad`, `3uqud`, `3akkad`, `3akkid`, `3aqqad`, and `3aqqid`; `نقر` appears as `naqar`, `naqir`, `nuqur`, `naqqar`, `naqqir`, `nuqqar`, and `nuqqur`; and `زور` appears as `zoor`, `zuur`, `zawar`, `ziwir`, `zawwar`, and `zawwir`. This means spelling variation is not just a theoretical concern; it is visible in the project data.",
    "A separate cleaning step creates `maknuune-v1.0.1_cleaned_predictions_top_5.csv`, also with 36,302 rows. It begins from harakat predictions produced by the harakat adder and extracts the most likely alternatives from `harakat_predictions_softmax_90`. The output adds `harakat_top_prediction` plus `harakat_prediction_1` through `harakat_prediction_5`. This turns the harakat-adder output into a static second-stage training input and makes the top-one and top-five Arabic-to-Arabizi experiments directly comparable.",
    "A second comparison dataset, `maknuune-v1.0.1_cleaned_levanti_harakat_predictions.csv`, stores harakat predicted by an off-the-shelf Hugging Face Levanti diacritizer. It uses the same underlying Maknuune rows but gives the Arabizi generator a different kind of pronunciation hint. This lets the project compare an outside Levanti diacritization stage against the custom harakat-adder outputs trained on the project's own Palestinian Arabic data.",
    "The source-target construction is direction-specific. For Arabizi-to-Arabic, the source is `arabizi` and the target is `arabic_stripped`. For the harakat adder, the source is `arabic_stripped` and the target is `arabic_harakat`. For Levanti-harakat-to-Arabizi, the source is `levanti_harakat_prediction` and the target is `arabizi`. For harakat-top-one-to-Arabizi, the source is `harakat_top_prediction` and the target is `arabizi`. For harakat-top-five-to-Arabizi, the source is the ordered list of up to five harakat predictions, with blanks allowed when fewer than five exist, and the target is `arabizi`. The top-five structure lets the prototype model learn from uncertainty in the preceding diacritization stage instead of pretending that only one vocalized form is possible.",
    "The data split uses 29,041 training rows, 3,630 validation rows, and 3,631 test rows for the harakat adder, with analogous splits in the transliteration experiments. Character vocabularies are learned from the training columns, and special tokens handle padding, sequence start, and sequence end. Because this is a course project focused on words and short phrases, character modeling is a practical choice. It preserves digits such as 2, 3, and 7 in Arabizi and preserves Arabic letters and combining marks in the Arabic side without requiring a large word vocabulary.",
]
body.extend(p(x) for x in dataset)

body.append(table([
    ["Artifact", "Rows", "Main columns", "Purpose"],
    ["maknuune-v1.0.1_cleaned.csv", "36,302", "arabizi; arabic_harakat; arabic_stripped", "Base paired lexicon"],
    ["maknuune-v1.0.1_cleaned_predictions_top_5.csv", "36,302", "base columns plus harakat_prediction_1..5", "Reusable input for Arabic-to-Arabizi stage two"],
    ["maknuune-v1.0.1_cleaned_levanti_harakat_predictions.csv", "36,302", "base columns plus levanti_harakat_prediction", "Levanti-diacritizer comparison input"],
    ["arabizi_to_arabic_best.pt", "--", "model state; vocabularies; config; metrics", "Best Arabizi-to-Arabic checkpoint"],
    ["harkat_adder.pt", "--", "model state; vocabularies; config; best metrics", "Arabic diacritization checkpoint"],
    ["train_arabic_w_harakat_to_arabizi_best.pt", "--", "model state; vocabularies; config; metrics", "Best gold-harakat-to-Arabizi checkpoint"],
    ["train_arabic_wo_harakat_to_arabizi_best.pt", "--", "model state; vocabularies; config; metrics", "Best no-harakat-to-Arabizi checkpoint"],
    ["train_arabic_w_levanti_harakat_to_arabizi_best.pt", "--", "model state; vocabularies; config; metrics", "Best Levanti-harakat-to-Arabizi comparison checkpoint"],
    ["train_harakat_top_1_to_arabizi_best.pt", "--", "model state; vocabularies; config; metrics", "Best top-one-to-Arabizi comparison checkpoint"],
    ["harakat_top_5_to_arabizi_best.pt", "--", "model state; vocabularies; config; metrics", "Best top-five-to-Arabizi checkpoint"],
], "DatasetArtifacts"))
body.append(caption("Table I. Principal data and model artifacts used by the system."))

body.append(table([
    ["Undiacritized Arabic", "Number of Arabizi variants", "Example Arabizi spellings"],
    ["عقد", "8", "3aqad; 3aqid; 3uqad; 3uqud; 3akkad; 3akkid; 3aqqad; 3aqqid"],
    ["نقر", "7", "naqar; naqir; nuqur; naqqar; naqqir; nuqqar; nuqqur"],
    ["زور", "6", "zoor; zuur; zawar; ziwir; zawwar; zawwir"],
    ["فرق", "6", "farq; faraq; fariq; firaq; farraq; farriq"],
    ["نفذ", "6", "nafad; nafaz; naffad; naffaz; naffid; naffiz"],
], "ArabiziVariationExamples"))
body.append(caption("Table II. Examples of multiple Arabizi spellings for the same undiacritized Arabic form in the cleaned Maknuune-derived data."))

body.append(h("V. Model Development"))
modeldev = [
    "All transliteration components are character-level encoder-decoder Transformer models. The Arabizi-to-Arabic, original-harakat-to-Arabizi, no-harakat-to-Arabizi, Levanti-harakat-to-Arabizi, harakat-top-one-to-Arabizi, and harakat-top-five-to-Arabizi models use a model dimension of 192, eight attention heads, three encoder layers, three decoder layers, a feed-forward dimension of 512, dropout of 0.15, and saved vocabularies for source and target characters. The harakat adder is slightly deeper and wider in the feed-forward block: it uses a model dimension of 192, six attention heads, four encoder layers, four decoder layers, and a feed-forward dimension of 768. This reflects the added complexity of predicting Arabic combining marks.",
    "The deep learning choice is central to the project. Arabizi transliteration has many context-dependent character mappings: `7` may represent ح, `3` may represent ع, `2` may represent a hamza-like sound, and Latin vowels can represent short vowels, long vowels, or writer-specific spelling habits. A rule system can encode common substitutions, but it struggles when multiple mappings interact across a whole word. A sequence-to-sequence neural model learns these interactions from paired examples. The encoder converts the entire source character sequence into contextual representations, and the decoder generates the target sequence one character at a time while attending to relevant source positions.",
    "The models operate at character level rather than word level because the vocabulary is small and the task is mostly orthographic. Word-level modeling would create a sparse vocabulary and would fail on unseen spellings. Character-level modeling allows the system to generalize to spelling variants that were not seen as exact tokens during training. It also lets the model preserve numerals and Arabic combining marks as first-class symbols. This is especially important in intelligence or investigative text processing, where informal data often contains rare names, abbreviations, typos, and locally meaningful spellings.",
    "The Arabizi-to-Arabic model is the custom model built for the reverse normalization direction. It trains a direct transliterator from the Latin-script `arabizi` column to unvocalized Arabic script in `arabic_stripped`. The same character-level Transformer architecture is appropriate here because the problem is a spelling-to-spelling sequence task with digits such as 2, 3, and 7 carrying sound information. At inference time, the model does not have to stop after a greedy prediction. Candidate decoding can expand possible sequences and rank the highest-likelihood options, which is useful when several Arabic strings are plausible.",
    "The Arabizi-to-Arabic model is still useful, but it is not the primary research risk in this project. It maps from an expressive informal spelling into a more compact Arabic-script form. The Arabic-to-Arabizi side has to do more: it must compensate for missing harakat, recover likely pronunciation, and then generate one or more plausible Latin-script spellings. That is why most of the model-comparison work in this report focuses on Arabic-to-Arabizi variants.",
    "The original-harakat-to-Arabizi model trains from `arabic_harakat` to `arabizi`. This is the clean pronunciation-visible version of the Arabic-to-Arabizi problem. It uses the same 192-dimensional character Transformer so that the comparison with other transliteration experiments isolates the input representation rather than changing the architecture. This model is best understood as an upper-bound style experiment: if the system already had the correct Palestinian harakat, how well could the Arabizi generator perform? Its strong result shows that the architecture can learn the Romanization task when the missing pronunciation cues are supplied.",
    "The no-harakat-to-Arabizi model trains from `arabic_stripped` to `arabizi`. This is the clean baseline for the real user input condition, where ordinary written Arabic does not include short-vowel marks. The architecture is deliberately kept the same as the original-harakat model so the performance difference measures the cost of removing harakat. It asks how much a Transformer can infer directly from bare Arabic letters without an explicit diacritization stage.",
    "The harakat-adder model trains a diacritizer from plain Arabic to diacritized Arabic. In the Arabic-to-Arabizi direction, this model is used only as an intermediate pronunciation stage. Softmax-90 means the decoder keeps generating alternatives until the retained candidates account for roughly 90 percent of the probability mass or until the configured candidate limit is reached. The goal is not to treat harakat restoration as the final task, but to pass plausible pronunciations forward to the Arabizi generator.",
    "The harakat model works by treating diacritization as character-level sequence generation. Its source string is Arabic with the harakat removed, and its target string is the same lexical item with the expected Palestinian Arabic diacritics restored. During encoding, the model builds contextual representations for the stripped Arabic characters. During decoding, it generates the vocalized output one character at a time, including ordinary Arabic letters and combining marks. This is more difficult than simply inserting vowels into fixed slots because the output sequence can include additional combining characters and because the correct marks depend on dialectal pronunciation patterns learned from the training data.",
    "The saved metrics show that the harakat adder works best on short-to-medium strings. Its best validation checkpoint is epoch 83, with validation sequence accuracy 0.4983. By length bucket, it performs best on 10-19 character strings, reaching about 0.5710 sequence accuracy. It also performs reasonably on 0-9 character strings, reaching about 0.4815. Accuracy falls sharply after that: 20-29 character strings reach about 0.0313, 30-39 character strings reach about 0.0526, and longer buckets are effectively zero in the saved validation analysis. This means the model is most useful for words and short phrases that resemble the lexicon-style dataset. It should not be treated as a reliable long-sentence diacritizer.",
    "The Levanti-harakat-to-Arabizi model is an outside-diacritizer comparison. The Levanti model was chosen because it is an existing Hugging Face model, `guymorlan/levanti_arabic2diacritics`, built for Levantine/Palestinian-style Arabic diacritization rather than formal MSA-only text. It was selected as a practical external baseline: close enough dialectally to be relevant, ready to run without training, and different from the project's custom harakat adder. The Levanti setup applies that model to `arabic_stripped`, writes `levanti_harakat_prediction`, and then trains the same character-level Arabizi generator on those predictions. This experiment asks whether a generic Levantine pronunciation hint is enough to support Arabizi generation and whether the project's custom Palestinian harakat predictions help more than an outside model.",
    "The custom harakat adder may outperform the Levanti option in this project for several practical reasons. It was trained directly on the same Maknuune-derived Palestinian data distribution used by the downstream Arabizi models, so its mistakes and vocabulary are more aligned with the final task. It also uses the same cleaning assumptions, Unicode handling, and target convention as the project pipeline. The Levanti model is valuable because it is dialect-aware and ready-made, but it was not optimized for this exact lexicon, this exact scoring setup, or the downstream Arabizi-generation objective.",
    "The harakat-top-one-to-Arabizi model is a comparison model. It uses the same cleaned harakat-prediction file, but it gives the Arabizi generator only `harakat_top_prediction`, the single most likely vocalized Arabic string from the custom harakat adder. This creates a direct ablation against the top-five design. If the top-one model performs much worse, then preserving multiple harakat hypotheses is justified. If it performs similarly, the simpler pipeline might be enough for some future use cases.",
    "The harakat-top-five-to-Arabizi model trains the second Arabic-to-Arabizi stage and is the approach selected for the proof-of-concept Arabic-to-Arabizi pipeline. It receives up to five candidate vocalized Arabic strings as the source representation and predicts Arabizi. This is more informative than using only stripped Arabic because Arabizi often contains vowel letters that correspond to short vowels or pronunciation choices. It is also more robust than a single predicted harakat string because the model can learn from several plausible pronunciations instead of depending on one upstream decision. During inference, blank slots are passed when the harakat adder produces fewer than five candidates. The second model then returns up to ten softmax-90 Arabizi outputs stacked in descending likelihood.",
    "The main technical work is therefore the model chain, the data preparation, and the evaluation of ranked candidate generation. The important question for this report is which input representation and decoding strategy best handles the Arabic-to-Arabizi ambiguity.",
]
body.extend(p(x) for x in modeldev)

body.append(table([
    ["Model", "Source", "Target", "Layers", "Heads", "Best epoch", "Validation seq. acc.", "Test seq. acc."],
    ["Harakat adder", "arabic_stripped", "arabic_harakat", "4 enc / 4 dec", "6", "83", "0.4983", "not stored in bundle"],
    ["Arabizi to Arabic", "arabizi", "arabic_stripped", "3 enc / 3 dec", "8", "89", "0.7533", "0.7580"],
    ["Original harakat to Arabizi", "arabic_harakat", "arabizi", "3 enc / 3 dec", "8", "87", "0.8280", "0.8312"],
    ["No harakat to Arabizi", "arabic_stripped", "arabizi", "3 enc / 3 dec", "8", "88", "0.4714", "0.4738"],
    ["Levanti harakat to Arabizi", "levanti_harakat_prediction", "arabizi", "3 enc / 3 dec", "8", "98", "0.4932", "0.5016"],
    ["Harakat top-1 to Arabizi", "harakat_top_prediction", "arabizi", "3 enc / 3 dec", "8", "78", "0.5138", "0.5152"],
    ["Harakat top-5 to Arabizi", "harakat_prediction_1..5", "arabizi", "3 enc / 3 dec", "8", "98", "0.6378", "0.6409"],
], "ModelSummary"))
body.append(caption("Table III. Model configurations and sequence-accuracy results from saved artifacts."))

body.append(h("Why Not One Bidirectional Model", 2))
bidirectional_rationale = [
    "A single bidirectional model was not used because the two directions are not symmetric tasks. Arabizi-to-Arabic is mostly a normalization problem from an expressive Romanized spelling into Arabic script. Arabic-to-Arabizi is a reconstruction problem followed by generation: the model must infer missing harakat and then choose among several possible Arabizi spellings. Combining both directions into one model would force one architecture and one training objective to cover two different information conditions.",
    "The Arabic-to-Arabizi side also needs an intermediate pronunciation representation. The best-performing prototype pipeline first generates multiple harakat hypotheses and then uses those hypotheses to generate Arabizi candidates. A single direct bidirectional model would either have to hide that intermediate reasoning inside one opaque network or add task-control tokens and special formatting to represent multiple candidate harakat strings. That would make training, debugging, and evaluation harder without solving the core ambiguity.",
    "Separate models also made the experiments cleaner. The project needed to compare original gold harakat, no harakat, custom top-one harakat, custom top-five harakat, and Levanti-predicted harakat. Keeping these as separate Arabic-to-Arabizi experiments made it possible to isolate exactly what each input representation contributed. A single bidirectional model would blur those ablations because errors could come from direction confusion, shared capacity limits, or the choice of input representation.",
    "Operationally, separate models give better candidate control. In intelligence-style use, the key result is whether the right spelling survives into a top-k candidate set. The Arabic-to-Arabizi pipeline can explicitly preserve multiple harakat hypotheses before final Romanization, while the Arabizi-to-Arabic model can focus on ranked Arabic-script normalization. This design is easier to explain, audit, and improve than one large bidirectional model that returns candidates without exposing where uncertainty entered the process.",
]
body.extend(p(x) for x in bidirectional_rationale)

body.append(table([
    ["Arabic-to-Arabizi experiment", "Input evidence", "Why it was trained", "Main result"],
    ["Original harakat", "Gold `arabic_harakat`", "Upper-bound style test when pronunciation cues are visible", "0.8312 test exact match"],
    ["No harakat", "`arabic_stripped`", "Baseline for ordinary written Arabic without short vowels", "0.4738 test exact; 0.874 top-10"],
    ["Custom top-5 harakat", "`harakat_prediction_1..5`", "Prototype pipeline preserving pronunciation uncertainty", "0.6409 test exact; 0.983 top-10"],
    ["Custom top-1 harakat", "`harakat_top_prediction`", "Ablation testing one custom diacritization hypothesis", "0.5152 test exact; 0.875 top-10"],
    ["Levanti harakat", "`levanti_harakat_prediction`", "External dialect-aware diacritizer comparison", "0.5016 test exact; 0.886 top-10"],
], "ArabicToArabiziExperiments"))
body.append(caption("Table IV. Arabic-to-Arabizi experiment map showing the role of each trained model."))

body.append(h("VI. Training Procedure"))
workflow = [
    "Each trainer follows the same broad pattern. First, it reads the cleaned CSV input and validates that the expected source and target columns are available. Second, it builds source and target character vocabularies, encodes strings into indexed tensors, and splits rows into training, validation, and test partitions. Third, it constructs the Transformer model from explicit configuration values and trains with validation checks after each epoch. Finally, it records metrics and plots so the final report can compare models by loss, token accuracy, sequence accuracy, top-k coverage, and length-bucket behavior.",
    "The saved model files include source and target vocabularies, padding indexes, architecture parameters, and selected metrics. This is important because a character-level transliterator is tightly coupled to its vocabulary. If a model were evaluated with a different character ordering, predictions would be meaningless even if the tensor shapes matched. Saving the vocabulary and configuration alongside the weights prevents that class of silent error and makes the reported metrics tied to the exact trained model.",
    "The training recipe was intentionally conventional so that the project would be understandable as a graduate deep-learning prototype. The experiments use seed 42, batch size 128, a maximum of 100 epochs, AdamW optimization with learning rate 3e-4 and weight decay 1e-4, cross-entropy loss with padding ignored, gradient clipping at norm 1.0, and a OneCycle learning-rate schedule. The transliteration models use early stopping patience of 12 validation-loss checks and save checkpoints every five epochs, while the harakat-adder model uses patience 10. These choices keep the training setup clear and make differences between experiments easier to attribute to input representation rather than hidden training changes.",
    "A deliberate design choice was to perform the harakat-candidate cleaning once before training the Arabic-to-Arabizi stages. The cleaned prediction file turns the output of the harakat adder into a static training input, so the top-one and top-five experiments train against the same underlying examples with different amounts of pronunciation evidence. This separation keeps the ablation clean: the top-one model tests one guessed vocalization, while the top-five model tests whether preserving several vocalization hypotheses improves Arabizi generation.",
    "The training metrics also support model selection. The best checkpoint is selected by validation sequence accuracy, not simply by the final epoch. This matters because training loss can continue to decrease while validation exact-match accuracy plateaus or declines. The Arabizi-to-Arabic model, for example, trained through epoch 100 but selected epoch 89 as the best checkpoint. The Levanti-harakat-to-Arabizi model selected epoch 98. The harakat-top-one-to-Arabizi model selected epoch 78. The harakat-top-five-to-Arabizi model selected epoch 98. The harakat adder selected epoch 83. In all cases, the report uses the saved best-model metrics rather than assuming that the last epoch was best.",
]
body.extend(p(x) for x in workflow)

body.append(h("VII. Evaluation Methodology"))
evaltxt = [
    "The experiments report token accuracy, sequence accuracy, loss, length-bucket breakdowns, and top-k candidate coverage. Token accuracy counts individual output characters. Sequence accuracy is stricter: the entire predicted output must exactly equal the reference string. In transliteration, sequence accuracy can look harsh because a single character difference, an optional vowel, or a plausible alternate spelling counts as a full error. This issue is especially important for Arabic-to-Arabizi, where there may be several acceptable Romanized spellings for the same Arabic form. For a ranked candidate list, top-k accuracy is often more meaningful because the desired answer only needs to appear among the returned choices.",
    "For practical intelligence use, softmax-90 top-k coverage is the primary operational metric in this report. A watchlist, search expansion tool, or analyst triage interface usually does not require the model's first guess to be the single exact reference spelling. It needs the relevant spelling to be present in the candidate set so it can be searched, reviewed, reranked, or passed downstream. This makes top-k coverage closer to an operational recall measure, while exact sequence accuracy is better understood as a strict diagnostic of one-best model behavior.",
    "Top-k evaluation was intentionally limited to 1,000 samples for the softmax-90 candidate generators. This was done to control processing cost. For each sampled item, the model generates candidates in descending likelihood and records whether the gold reference appears in the top 1, 3, 5, or 10. The same analysis is also broken down by source and target length, allowing the report to separate short-token performance from longer phrase behavior. The softmax-90 threshold is useful because it preserves a compact set of likely alternatives rather than returning every low-probability string the decoder could imagine.",
    "The length-based error analysis is useful because transliteration errors compound with sequence length. A five-character word may require only a few decisions, while a twenty-character phrase requires many more. Exact-match sequence accuracy therefore normally drops as length increases even when token accuracy remains high. The saved figures and CSV files confirm this pattern for both directions. This is not a failure of the plots; it is a natural property of exact sequence metrics.",
]
body.extend(p(x) for x in evaltxt)

body.append(h("VIII. Results"))
results = [
    "The harakat adder reached its best validation sequence accuracy at epoch 83. Its best validation sequence accuracy was 0.4983, with validation loss 0.2680 and training sequence accuracy 0.5466. By length, the validation accuracy was strongest for 10-19 character items at about 0.5710 and lower for very short items at about 0.4815. It dropped sharply for longer items: 20-29 character items reached 0.0313, and the longest buckets were essentially zero. This shows that the model is useful for short words and short expressions but that full exact diacritization becomes difficult as length grows.",
    "The Arabizi-to-Arabic model completed 100 epochs and selected epoch 89 as the best checkpoint by validation sequence accuracy. At that epoch, validation sequence accuracy was 0.7533, validation token accuracy was 0.9151, and validation loss was 0.2959. The saved test metrics report test loss of 0.2827, test token accuracy of 0.9183, and test sequence accuracy of 0.7580. These numbers indicate that the model usually gets most characters right and exactly matches the reference for about three quarters of held-out examples.",
    "The Arabizi-to-Arabic top-k evaluation is stronger than the single-best metric. On the 1,000-sample softmax-90 run, top-1 accuracy was 0.775, top-3 accuracy was 0.888, top-5 accuracy was 0.904, and top-10 accuracy was 0.917. In other words, when the decoder is allowed to return ten descending-likelihood Arabic candidates, the reference appears in the list for about 91.7 percent of sampled cases. These results show that the reverse normalization direction is already relatively strong compared with the Arabic-to-Arabizi side, where the system must infer unwritten pronunciation information before it can generate a Romanized spelling.",
    "The original-harakat-to-Arabizi model completed 100 epochs and selected epoch 87 as the best validation checkpoint. With gold Maknuune harakat visible, validation sequence accuracy reached 0.8280, validation token accuracy reached 0.9478, and validation loss was 0.1839. On the normal test set, the model exactly matched 2,896 of 3,484 examples, or about 0.8312 sequence accuracy. This is the strongest single-best exact-match result among the Arabic-to-Arabizi experiments and shows that the shared character Transformer can perform the Romanization task well when pronunciation information is available.",
    "The same original-harakat-trained model was also stress-tested on 1,000 stripped-Arabic inputs, even though it had been trained with harakat. That stripped-input softmax-90 test reached only 0.084 top-1, 0.168 top-3, 0.220 top-5, and 0.309 top-10 accuracy. This sharp drop is one of the clearest pieces of evidence in the project: a model trained to rely on visible harakat cannot simply be handed unvocalized Arabic at inference time and expected to recover.",
    "The no-harakat-to-Arabizi model completed 100 epochs and selected epoch 88 as the best validation checkpoint. It trained directly on `arabic_stripped`, so it is the fairest no-pronunciation baseline. Best validation sequence accuracy was 0.4714, with validation token accuracy 0.8873 and validation loss 0.3295. On the test set, it exactly matched 1,635 of 3,451 examples, or about 0.4738 sequence accuracy. Its softmax-90 candidate evaluation was much stronger than its one-best exact match: top-1 accuracy was 0.465, top-3 was 0.751, top-5 was 0.826, and top-10 was 0.874 on 1,000 sampled rows.",
    "The Levanti-harakat-to-Arabizi comparison model completed 98 epochs and selected epoch 98 as the best validation checkpoint. At that epoch, validation loss was 0.3118, validation token accuracy was 0.8944, and validation sequence accuracy was 0.4932. The test error-analysis file reports 1,731 exact matches out of 3,451 test examples, or about 0.5016 test sequence accuracy. Its softmax-90 candidate evaluation reached top-1 accuracy 0.506, top-3 accuracy 0.781, top-5 accuracy 0.840, and top-10 accuracy 0.886 on 1,000 sampled test rows.",
    "The harakat-top-one-to-Arabizi comparison model completed all 100 epochs and selected epoch 78 as the best validation checkpoint. At that epoch, validation loss was 0.3413, validation token accuracy was 0.8869, and validation sequence accuracy was 0.5138. The test error-analysis file reports 1,778 exact matches out of 3,451 test examples, or about 0.5152 test sequence accuracy. Its softmax-90 candidate evaluation reached top-1 accuracy 0.505, top-3 accuracy 0.761, top-5 accuracy 0.821, and top-10 accuracy 0.875 on 1,000 sampled test rows.",
    "The harakat-top-five-to-Arabizi model selected epoch 98 as the best validation checkpoint. At that epoch, validation loss was 0.1152, validation token accuracy was 0.9539, and validation sequence accuracy was 0.6378. The saved test metrics report test loss of 0.1184, test token accuracy of 0.9539, and test sequence accuracy of 0.6409. The top-k evaluation again shows why candidate lists are important: top-1 accuracy was 0.654, top-3 accuracy was 0.922, top-5 accuracy was 0.967, and top-10 accuracy was 0.983.",
    "The Arabic-to-Arabizi experiments form a clear ladder. Gold original harakat gives the best one-best exact match because it exposes the pronunciation information directly, but it is not the normal usable case because users usually do not provide correct harakat. No harakat is much harder, but it still produces useful candidate lists. Levanti and custom top-one predicted harakat slightly improve one-best exact match over the no-harakat baseline, but both remain single-hypothesis approaches. The selected proof-of-concept approach is the custom harakat top-five pipeline: it improves top-10 coverage to 98.3 percent and improves single-best test exact match to about 64.1 percent. This supports the pipeline decision to preserve multiple custom harakat hypotheses before generating Arabizi, while still leaving significant work before production use.",
    "The custom top-five pipeline also explains why the home-built harakat adder is more useful here than the Levanti comparison. The Levanti path gives one outside model prediction and reaches 88.6 percent top-10 coverage. The custom top-five path gives the downstream model several project-specific Palestinian harakat hypotheses and reaches 98.3 percent top-10 coverage. That gap does not mean the Levanti model is bad; it means this task benefits from a diacritizer trained and decoded in the same domain as the final Arabizi model, especially when its uncertainty is preserved rather than collapsed to one string.",
    "From the intelligence-workflow perspective, these top-k results are the headline model-analysis finding. The single-best exact-match numbers show how often the model chooses the reference spelling by itself. The softmax-90 top-k numbers show how often the model keeps the reference spelling available for a recall-oriented process. For Arabic-to-Arabizi, where missing harakat and non-standard spelling make one-best prediction inherently uncertain, top-10 coverage is a better measure of practical usefulness than top-1 exact match alone.",
]
body.extend(p(x) for x in results)

body.append(table([
    ["Model", "Top-1", "Top-3", "Top-5", "Top-10", "Sample size"],
    ["Arabizi to Arabic", "0.775", "0.888", "0.904", "0.917", "1,000"],
    ["Original harakat trained, stripped input stress test", "0.084", "0.168", "0.220", "0.309", "1,000"],
    ["No harakat to Arabizi", "0.465", "0.751", "0.826", "0.874", "1,000"],
    ["Levanti harakat to Arabizi", "0.506", "0.781", "0.840", "0.886", "1,000"],
    ["Harakat top-1 to Arabizi", "0.505", "0.761", "0.821", "0.875", "1,000"],
    ["Harakat top-5 to Arabizi", "0.654", "0.922", "0.967", "0.983", "1,000"],
], "TopK"))
body.append(caption("Table V. Softmax-90 top-k coverage for candidate-generating transliteration models."))

body.append(h("IX. Error Analysis"))
analysis = [
    "The Arabizi-to-Arabic error analysis shows a steep length effect. For source strings of 0-9 characters, sequence accuracy was about 0.8214. For 10-19 characters, it fell to about 0.4215. For 20-29 characters, it was about 0.0313. Target-length buckets show the same broad behavior: 0-9 character targets reached about 0.7996, while 10-19 character targets reached only 0.0739. This does not mean the model has no partial knowledge for longer strings; token accuracy remains high. It means that exact full-string matching becomes statistically fragile as more characters must all be correct at once.",
    "The original-harakat-to-Arabizi model shows what happens when pronunciation is visible. Source strings of 0-9 characters reached about 0.9012 sequence accuracy, and 10-19 character source strings reached about 0.6689. Target strings of 0-9 characters reached about 0.9115, while 10-19 character targets reached about 0.4272. Longer examples still failed under exact-match scoring, but the short-word results are much stronger than the no-harakat baseline.",
    "The no-harakat-to-Arabizi baseline shows the penalty for removing pronunciation cues. Source strings of 0-9 characters reached about 0.5015 sequence accuracy, while 10-19 character source strings reached only about 0.0170. Target strings of 0-9 characters reached about 0.5075, and 10-19 character targets reached about 0.3141. The model can often infer common short-word spellings, but phrase-level exact matching deteriorates quickly because each missing vowel or spelling choice compounds.",
    "The Levanti-harakat-to-Arabizi comparison model has a similar short-input profile to the custom top-one model. Source strings of 0-9 characters reached about 0.5075 sequence accuracy, and 10-19 character source strings reached about 0.5589. Target strings of 0-9 characters reached about 0.5353, while 10-19 character targets reached about 0.3482. Like the custom top-one model, it did not exactly match 20-plus-character examples in this test split. This suggests that a single predicted vocalization, whether Levanti or custom, leaves the Arabizi generator with limited evidence for longer strings.",
    "The harakat-top-one-to-Arabizi comparison model shows the cost of relying on only the highest-probability diacritization. Source strings of 0-9 characters reached about 0.5167 sequence accuracy, and 10-19 character source strings reached about 0.5524. Target strings of 0-9 characters reached about 0.5491, while 10-19 character targets reached about 0.3547. The model did not exactly match any 20-plus-character examples in this test split. These numbers are useful as an ablation baseline because the architecture is similar to the top-five model but the source evidence is narrower.",
    "The harakat-top-five-to-Arabizi model has a different length profile. It performs well on many mid-length examples because the source representation contains multiple pronunciation hypotheses. Source lengths of 10-19 characters reached about 0.7625 sequence accuracy and 20-29 characters reached about 0.5798, while 30-39 characters dropped to about 0.3793. Target-length buckets show similar degradation as outputs become longer. This suggests that the top-five harakat representation gives the model useful disambiguating information, but it cannot fully remove compounding sequence risk.",
    "A second error source is orthographic multiplicity. Arabizi has no single standard spelling. Long vowels may be doubled or not, English-inspired spellings may compete with Arabic-sound spellings, and numerals may be used inconsistently. If the reference is `salaam`, a model output of `salam` may be readable and plausible but still counted wrong. This affects Arabic-to-Arabizi more than Arabizi-to-Arabic because the target side is not a standardized script form. Likewise, Arabic output may differ in hamza choice, alif form, or final letter while remaining interpretable. Strict metrics are useful for consistent scoring, but they understate practical usefulness when the candidate list contains acceptable alternatives.",
    "The harakat-adder errors are especially consequential in the Arabic-to-Arabizi pipeline. If the first stage misses a vocalization, the second stage may never see the pronunciation needed to generate the desired Arabizi output. The top-five strategy reduces this risk but does not eliminate it. Increasing the harakat candidate count would likely improve coverage but also increase compute cost and may introduce noisy candidates. The current system chooses five as a practical compromise between preserving uncertainty and limiting noise.",
]
body.extend(p(x) for x in analysis)

body.append(table([
    ["Direction", "Length type", "0-9", "10-19", "20-29", "30-39"],
    ["Arabizi to Arabic", "source", "0.8214", "0.4215", "0.0313", "--"],
    ["Arabizi to Arabic", "target", "0.7996", "0.0739", "0.0769", "0.0000"],
    ["Original harakat to Arabizi", "source", "0.9012", "0.6689", "0.0000", "0.0000"],
    ["Original harakat to Arabizi", "target", "0.9115", "0.4272", "0.0000", "0.0000"],
    ["No harakat to Arabizi", "source", "0.5015", "0.0170", "0.0000", "0.0000"],
    ["No harakat to Arabizi", "target", "0.5075", "0.3141", "0.0000", "0.0000"],
    ["Levanti harakat to Arabizi", "source", "0.5075", "0.5589", "0.0000", "0.0000"],
    ["Levanti harakat to Arabizi", "target", "0.5353", "0.3482", "0.0000", "0.0000"],
    ["Top-1 harakat to Arabizi", "source", "0.5167", "0.5524", "0.0000", "0.0000"],
    ["Top-1 harakat to Arabizi", "target", "0.5491", "0.3547", "0.0000", "0.0000"],
    ["Top-5 harakat to Arabizi", "source", "0.6149", "0.7625", "0.5798", "0.3793"],
    ["Top-5 harakat to Arabizi", "target", "0.6290", "0.7637", "0.5709", "0.2949"],
], "LengthAccuracy"))
body.append(caption("Table VI. Selected sequence-accuracy buckets from length-based error analysis."))

body.append(h("X. Threats to Validity and Feasibility Boundaries"))
validity = [
    "Because this is a prototype, the evaluation should be read as evidence of feasibility rather than proof of readiness for operational deployment. The strongest limitation is dataset realism. Maknuune is a clean and valuable Palestinian Arabic lexical resource, and it was the best available dataset found for this project because it directly connects Palestinian Arabic, Arabizi, and diacritized Arabic forms. However, it is not the same as raw OSINT text. Public social-media Arabizi can contain full sentences, emojis, hashtags, usernames, code-switching with English, spelling mistakes, repeated letters, sarcasm, abbreviations, and irregular punctuation. The present system is strongest on words and short phrases that resemble the lexicon-style training examples. A production system would need an external test set drawn from real public Arabizi and would need preprocessing for noisy text.",
    "The OSINT framing also needs careful boundaries. This system is not an intelligence system and should not be treated as one. It does not infer intent, threat, identity, reliability, location, or meaning. It only attempts to normalize script forms so that Arabizi material can be searched or routed into Arabic-language tools. In a responsible OSINT workflow, transliteration would be one early text-processing step used with lawful collection, human review, context, and policy controls. The model can increase recall for keyword monitoring, but it cannot decide whether a keyword match matters.",
    "Another limitation is the absence of a formal baseline comparison. A full study should compare the Transformer models against simpler alternatives such as a hand-built character substitution table, finite-state transliteration rules, edit-distance lookup against the lexicon, or a nearest-neighbor retrieval method. Those baselines would answer whether the neural approach provides enough benefit to justify its complexity. This project partially motivates the neural approach by pointing to ambiguous mappings and strong top-k coverage, but it does not yet quantify gains over a simpler baseline.",
    "The model architecture was also not selected through a full hyperparameter search. The chosen Transformer dimensions are reasonable for a small character-level task, but the study does not prove that 192 hidden dimensions, three or four encoder-decoder layers, dropout 0.15, or eight attention heads are optimal. The current architecture should therefore be understood as a practical prototype configuration, not as an optimized model family. A stronger deep-learning study would train smaller and larger variants and report whether performance is limited by model capacity, data size, or decoding strategy.",
    "Exact-match metrics also have limits. Arabizi has many acceptable spellings, and Palestinian Arabic itself can have multiple plausible Arabic-script renderings. A model output can be useful to a human analyst while still failing exact sequence accuracy against a single reference. That limitation matters less for the intended application than it would for a one-shot spelling converter, because the goal is not to produce one official spelling. The goal is to help intelligence users expand Arabic keywords into searchable Arabizi variants for Boolean queries and recall-oriented triage. Top-k metrics address this problem by measuring whether the reference appears among ranked candidates, but they still rely on one gold answer. A stronger evaluation would add character error rate, edit distance, and human judgments of whether non-reference candidates are acceptable.",
    "The current top-k evaluation is informative but incomplete. It uses a 1,000-example softmax-90 sample to manage compute cost, so the reported top-k values should be treated as estimates. The report does not include confidence intervals, random-seed variation, or cross-validation. It also selects best checkpoints by validation sequence accuracy rather than by validation top-k coverage, even though the intended use case emphasizes candidate lists. Future experiments should align the selection metric more closely with the final use case.",
    "The harakat stage is a known bottleneck. Its best validation sequence accuracy is about 49.8 percent, and its performance drops sharply on strings longer than about twenty characters. The Arabic-to-Arabizi pipeline can only use the pronunciation hypotheses the harakat adder provides. If the correct vocalization is missing from the top five, the second-stage model may never generate the best Arabizi output. This is a real pipeline-complexity risk, but it is also the main reason the pipeline exists: in the absence of much larger Palestinian Arabic training data that could let a direct model learn around missing harakat, preserving several pronunciation hypotheses is the most practical way to reduce information loss. It is acceptable for a feasibility prototype focused on short inputs, but it is not sufficient for long sentence processing without segmentation, more data, or a stronger dialect-specific diacritizer.",
    "A related deep-learning issue is exposure bias. During training, the decoder learns with the correct previous target characters available through teacher-forced targets. During inference, it must condition on its own earlier predictions. A small early mistake can therefore compound, which helps explain why exact-match accuracy drops sharply as strings become longer. This is a normal challenge for autoregressive sequence models, but it should be named because it affects how the length-based results are interpreted.",
    "Unicode and normalization choices are another possible source of error. Arabic script contains multiple alif and hamza forms, ta marbuta, alif maqsura, and combining marks that can be represented in ways that look similar to a reader but differ as Unicode strings. The current project relies on the cleaned data and model vocabularies, but a full benchmark would document normalization rules explicitly and report whether evaluation was performed on raw strings, normalized strings, or both.",
    "Dialect scope is also intentionally narrow. The trained system is Palestinian-focused because the data is Palestinian-focused. Arabizi conventions vary across the Arabic-speaking world, and dialects differ in sound systems, vocabulary, morphology, and even pronunciation within the same broad dialect family. This point also explains why the custom harakat adder can outperform the Levanti comparison in the pipeline: the Levanti model is broader and useful as an external baseline, but it is not tuned to the exact Palestinian lexicon and downstream Arabizi spelling distribution used here. A Gulf, Egyptian, Iraqi, Maghrebi, or broader Levantine deployment would need dialect-specific data or a multilingual dialect design. The current model should therefore be described as Palestinian Arabic transliteration, not as a general Arabic or pan-Arabizi solution.",
    "Finally, top-k decoding trades automation for recall. Showing up to ten candidates improves the chance that a useful transliteration appears, but it also creates a selection burden for a human or downstream system. For this project that tradeoff is intentional: the target use case is intelligence keyword expansion and Boolean search support, where recall is usually more valuable than forcing a single polished consumer-facing answer. In future work, calibrated probabilities, candidate grouping, duplicate handling, and downstream reranking could make the candidate list easier to use. For this course project, top-k output is best understood as a transparent way to preserve uncertainty while demonstrating that the models have learned useful mappings.",
]
body.extend(p(x) for x in validity)

body.append(h("XI. Figures and Training Graphics"))
body.append(p("The following figures are embedded from the saved training-output folders. They document the model behavior described above: loss and error trends, learning-rate schedules, sequence accuracy, top-k coverage, and error rates by length. Including the plots in the report is useful because the CSV tables give exact values while the graphics reveal whether training converged smoothly or oscillated. The curves also show why the saved best checkpoints are not always the final epoch; validation sequence accuracy can peak before the last epoch even when training loss continues to improve."))
for i, (path_str, cap) in enumerate(figures, start=1):
    path = ROOT / path_str
    if path.exists():
        body.append(image(path, i, cap))

body.append(h("XII. Practical Interpretation"))
practical = [
    "The project should be understood as a proof of concept, not as a full-use production app. It demonstrates that a character-level Transformer pipeline can learn useful Palestinian Arabizi/Arabic mappings and that ranked candidate generation is especially valuable in the harder Arabic-to-Arabizi direction. With more time, data, engineering resources, and human evaluation, the same modeling idea could be refined into a stronger tool.",
    "The practical value of the model chain is that it can turn a single query form into multiple plausible script variants. In an intelligence keyword-search setting, this matters because a watchlist term may be missed if it is searched only in Arabic script or only in one Arabizi spelling. The model is not making intelligence judgments and is not translating meaning. It is a recall-oriented text normalization component: it proposes spellings that could be used for Boolean search expansion, analyst review, or downstream NLP.",
    "Within the prototype experiments, the best Arabic-to-Arabizi decision is the custom top-five harakat pathway. The original-harakat model is an important upper-bound experiment, and the no-harakat, Levanti, and custom top-one models are useful comparisons. But the custom harakat adder followed by the top-five-to-Arabizi model best matches the real input condition while giving the strongest top-k coverage. This should be read as a model-selection result for the proof of concept, not as a claim that the system is production-ready.",
    "Candidate lists are operationally central, not merely convenient. When Arabizi spelling is ambiguous, a single forced transliteration can hide uncertainty and cause a downstream keyword search to miss a useful spelling. Returning ranked alternatives allows a human analyst or later pipeline stage to preserve ambiguity while still seeing which candidates the model considers most likely. In that setting, the 98.3 percent top-10 result for the custom top-five Arabic-to-Arabizi stage is more important than the lower one-best exact-match score because the candidate set is what enables recall.",
]
body.extend(p(x) for x in practical)

body.append(h("XIII. Limitations and Future Work"))
limits = [
    "The most important limitation is that this is not ready as an app for full use. It is a proof of concept that demonstrates the feasibility of the modeling approach and the value of top-k candidate preservation. A full application would require more time, more representative data, a stronger evaluation set, better robustness to noisy inputs, security and privacy review, error monitoring, logging, documentation, and user testing. The current work should therefore be judged as a working research prototype, not as a finished operational system.",
    "The next major limitation is data scope. Maknuune is a high-quality Palestinian Arabic lexicon and the best available dataset identified for this Palestinian Arabizi/harakat task, but it is still primarily lexical. The current models are best for words and short expressions, not long conversational sentences. The length analysis shows why: exact sequence accuracy degrades quickly as the number of characters increases. Future work should add sentence-level data, context-aware decoding, and perhaps a language model for reranking multi-token outputs.",
    "A second limitation is evaluation against a single reference. The top-k analysis partly addresses this by showing candidate coverage, but it still checks candidates against one gold string. Human evaluation would better capture acceptable spelling variants. This is most important for Arabic-to-Arabizi because the output is a user convention rather than a standardized orthography. The cleaned dataset already demonstrates the issue: about 18.5 percent of unique undiacritized Arabic forms have multiple Arabizi spellings. The project did not have time to build a multi-reference benchmark or human acceptability study, so the report treats this as valid future work rather than pretending exact match captures the whole problem.",
    "A third limitation is model size and training budget. The models are intentionally small and use character-level vocabularies. Larger Transformers, byte-pair or unigram subword vocabularies, data augmentation, and pronunciation-aware features could improve robustness. However, larger models also increase serving cost and make error analysis harder. The present design favors a model family that is small enough to train, compare, and explain within the course project.",
    "The Arabic-to-Arabizi pipeline also depends on intermediate harakat quality. If the harakat adder fails, the second-stage transliterator receives incomplete evidence. This is the cost of using a pipeline, but it was chosen because ordinary Arabic input lacks the short-vowel information that Arabizi often needs. With more Palestinian Arabic training data, a direct no-harakat model might learn more of that missing information from distributional patterns alone. In the current data setting, the top-five harakat stage is the strongest practical compromise because it gives the Arabizi model multiple plausible pronunciations instead of a single guess. A future version could train the Arabic-to-Arabizi direction end-to-end with latent pronunciation candidates, or jointly train the two stages so that the first stage optimizes downstream Arabizi coverage rather than only diacritized Arabic exact match.",
    "A final limitation is that the current system is a transliteration model prototype rather than a complete dialectal Arabic text-processing platform. It does not segment clitics, identify named entities, normalize emojis or punctuation-heavy social media text, or condition on surrounding sentence context. Those tasks are common in real Arabizi data and would become important in intelligence, law-enforcement, humanitarian, or research settings. The present scope was narrower by design: build a transparent deep-learning pipeline from a reliable Palestinian lexicon, preserve candidate uncertainty, and evaluate the trained models carefully. The intended application is specifically intelligence-style keyword expansion and Boolean search support, so the report prioritizes recall-oriented candidate coverage over consumer-app polish or one-best spelling authority.",
    "A future version should add baseline experiments. The most useful baselines would be a rule-based Arabizi character mapper, an edit-distance lexicon retriever, and a simple lookup model. If the neural models beat those baselines on exact accuracy, top-k coverage, and edit distance, the case for deep learning would be much stronger. If a baseline performs similarly on short words, then the neural model's value may be strongest for ambiguous spellings, unseen variants, or ranked candidate generation.",
    "A future version should also include human evaluation and external data. Human bilingual or dialect-aware annotators could judge whether candidates are acceptable even when they do not match the single reference. An external OSINT-style test set, collected ethically from public examples or manually created to simulate noisy public text, would measure whether the model generalizes beyond the clean Maknuune distribution. These additions would move the work from feasibility prototype toward a more convincing applied NLP study.",
    "The highest-priority deep-learning improvements are additional ablations and stricter validation. The new Levanti and top-one experiments show that single predicted-harakat inputs are meaningfully weaker than preserving five custom hypotheses, and the no-harakat experiment already shows the cost of removing the harakat stage entirely. The decoder should also be compared under greedy decoding, ordinary beam search, and the current softmax-90 candidate strategy. Because the intelligence-use case is recall-oriented, future model selection should consider validation top-k coverage directly rather than relying only on validation sequence accuracy.",
    "Another useful future step would be a stricter data split. Random row splits may allow related forms, shared lemmas, or very similar spellings to appear in both training and validation/test sets. A lemma-aware, root-aware, or string-similarity-aware split would better test generalization. If performance drops under that stricter split, the result would not invalidate the prototype; it would clarify whether the current model is learning productive character correspondences or partly memorizing lexicon entries.",
    "Longer-term work could explore pretrained or joint models. ByT5, mT5, AraBERT-derived systems, or other pretrained sequence models might improve robustness to noisy real-world text, though they would be larger and less transparent than the current prototype. A joint Arabic-to-Arabizi model could also learn the harakat and Romanization steps together, reducing pipeline error propagation. Those directions are outside the time and scope of this course project, but they are natural next steps if the prototype is extended into a larger research effort.",
]
body.extend(p(x) for x in limits)

body.append(h("XIV. Conclusion"))
conclusion = [
    "This project produced a working proof-of-concept model pipeline for Palestinian Arabizi-Arabic transliteration built from cleaned Maknuune data, one custom Arabizi-to-Arabic model, five Arabic-to-Arabizi modeling paths, a custom harakat adder, saved metrics, and plotted diagnostics. It is not a complete app ready for full operational use; it is a feasibility demonstration that shows a promising architecture and a clear path for later refinement with more time, data, validation, and engineering resources. The white-paper argument is that the central challenge is not simply moving between two scripts. It is the Arabic-to-Arabizi direction specifically: ordinary Arabic omits many harakat and therefore hides pronunciation cues, while Arabizi itself allows many writer-dependent spellings. Arabizi-to-Arabic reaches 91.7 percent top-10 coverage on the sampled softmax-90 evaluation, but Arabic-to-Arabizi requires explicit uncertainty handling.",
    "The Arabic-to-Arabizi model sequence shows why. The original-harakat model reaches about 83.1 percent exact test accuracy when gold pronunciation marks are present, but that is an upper-bound condition rather than the normal user workflow. The no-harakat model drops to about 47.4 percent exact accuracy, proving that the missing marks matter. The Levanti and custom top-one predicted-harakat models recover some pronunciation information, but both are still limited by one guessed vocalization. The final selected proof-of-concept approach is the custom harakat top-five pipeline because it starts from ordinary Arabic input and performs best operationally by preserving multiple plausible pronunciations before generating Arabizi candidates.",
    "The report also shows why the system is architected as a pipeline. Arabic without harakat does not fully specify pronunciation, while Arabizi often represents vowels. Adding a harakat prediction stage gives the Arabic-to-Arabizi model more useful evidence, and preserving multiple harakat hypotheses is better than forcing one pronunciation too early. The no-harakat model reaches 87.4 percent top-10 coverage, the Levanti comparison reaches 88.6 percent, the custom top-one comparison reaches 87.5 percent, and the harakat-top-five-to-Arabizi stage reaches 98.3 percent. Those softmax-90 top-k results are the most important practical intelligence result in the paper because they measure whether the right candidate survives into a usable review/search set. The metrics and graphics give clear targets for future deep-learning improvement.",
]
body.extend(p(x) for x in conclusion)

body.append(h("References"))
refs = [
    "[1] I. Sutskever, O. Vinyals, and Q. V. Le, \"Sequence to Sequence Learning with Neural Networks,\" Advances in Neural Information Processing Systems, 2014.",
    "[2] D. Bahdanau, K. Cho, and Y. Bengio, \"Neural Machine Translation by Jointly Learning to Align and Translate,\" arXiv:1409.0473, 2014.",
    "[3] A. Vaswani et al., \"Attention Is All You Need,\" Advances in Neural Information Processing Systems, 2017.",
    "[4] M. Al-Badrashiny, R. Eskander, N. Habash, and O. Rambow, \"Automatic Transliteration of Romanized Dialectal Arabic,\" Proceedings of CoNLL, pp. 30-38, 2014.",
    "[5] N. Habash, M. Diab, and O. Rambow, \"Conventional Orthography for Dialectal Arabic,\" Proceedings of LREC, pp. 711-718, 2012.",
    "[6] O. Hamed and T. Zesch, \"A Survey and Comparative Study of Arabic Diacritization Tools,\" Journal for Language Technology and Computational Linguistics, vol. 32, no. 1, pp. 27-47, 2017.",
    "[7] A. Fadel, I. Tuffaha, B. Al-Jawarneh, and M. Al-Ayyoub, \"Arabic Text Diacritization Using Deep Neural Networks,\" International Conference on Computer Applications and Information Security, 2019.",
    "[8] A. Paszke et al., \"PyTorch: An Imperative Style, High-Performance Deep Learning Library,\" Advances in Neural Information Processing Systems, 2019.",
    "[9] S. Dibas et al., \"Maknuune: A Large Open Palestinian Arabic Lexicon,\" WANLP, 2022, arXiv:2210.12985.",
]
body.extend(p(r, "Reference") for r in refs)

all_text = "\n".join(re.sub("<[^>]+>", " ", x) for x in body)

content_xml = f'''<?xml version="1.0" encoding="UTF-8"?>
<office:document-content
 xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
 xmlns:style="urn:oasis:names:tc:opendocument:xmlns:style:1.0"
 xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"
 xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0"
 xmlns:draw="urn:oasis:names:tc:opendocument:xmlns:drawing:1.0"
 xmlns:fo="urn:oasis:names:tc:opendocument:xmlns:xsl-fo-compatible:1.0"
 xmlns:xlink="http://www.w3.org/1999/xlink"
 xmlns:svg="urn:oasis:names:tc:opendocument:xmlns:svg-compatible:1.0"
 office:version="1.2">
 <office:automatic-styles>
  <style:style style:name="Title" style:family="paragraph"><style:text-properties fo:font-size="18pt" fo:font-weight="bold"/><style:paragraph-properties fo:text-align="center" fo:margin-bottom="0.12in"/></style:style>
  <style:style style:name="Subtitle" style:family="paragraph"><style:text-properties fo:font-size="11pt" fo:font-style="italic"/><style:paragraph-properties fo:text-align="center" fo:margin-bottom="0.18in"/></style:style>
  <style:style style:name="Abstract" style:family="paragraph"><style:text-properties fo:font-size="9pt"/><style:paragraph-properties fo:text-align="justify" fo:margin-bottom="0.08in"/></style:style>
  <style:style style:name="Keywords" style:family="paragraph"><style:text-properties fo:font-size="9pt" fo:font-style="italic"/><style:paragraph-properties fo:margin-bottom="0.12in"/></style:style>
  <style:style style:name="Heading1" style:family="paragraph"><style:text-properties fo:font-size="11pt" fo:font-weight="bold"/><style:paragraph-properties fo:margin-top="0.16in" fo:margin-bottom="0.06in"/></style:style>
  <style:style style:name="Heading2" style:family="paragraph"><style:text-properties fo:font-size="10pt" fo:font-weight="bold"/><style:paragraph-properties fo:margin-top="0.12in" fo:margin-bottom="0.05in"/></style:style>
  <style:style style:name="P1" style:family="paragraph"><style:text-properties fo:font-size="9.5pt"/><style:paragraph-properties fo:text-align="justify" fo:margin-bottom="0.07in"/></style:style>
  <style:style style:name="Caption" style:family="paragraph"><style:text-properties fo:font-size="8.5pt" fo:font-style="italic"/><style:paragraph-properties fo:text-align="center" fo:margin-bottom="0.1in"/></style:style>
  <style:style style:name="Reference" style:family="paragraph"><style:text-properties fo:font-size="8.5pt"/><style:paragraph-properties fo:margin-bottom="0.04in"/></style:style>
  <style:style style:name="TableCell" style:family="paragraph"><style:text-properties fo:font-size="8pt"/></style:style>
  <style:style style:name="TableHeader" style:family="paragraph"><style:text-properties fo:font-size="8pt" fo:font-weight="bold"/></style:style>
 </office:automatic-styles>
 <office:body>
  <office:text>
   {"".join(body)}
   {p("Approximate body word count: " + str(words(all_text)), "Reference")}
  </office:text>
 </office:body>
</office:document-content>
'''

styles_xml = '''<?xml version="1.0" encoding="UTF-8"?>
<office:document-styles
 xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
 xmlns:style="urn:oasis:names:tc:opendocument:xmlns:style:1.0"
 xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"
 xmlns:fo="urn:oasis:names:tc:opendocument:xmlns:xsl-fo-compatible:1.0"
 office:version="1.2">
 <office:styles>
  <style:default-style style:family="paragraph">
   <style:text-properties style:font-name="Times New Roman" fo:font-family="Times New Roman" fo:font-size="9.5pt"/>
  </style:default-style>
 </office:styles>
 <office:automatic-styles>
  <style:page-layout style:name="pm1">
   <style:page-layout-properties fo:page-width="8.5in" fo:page-height="11in" fo:margin-top="0.7in" fo:margin-bottom="0.7in" fo:margin-left="0.65in" fo:margin-right="0.65in"/>
  </style:page-layout>
 </office:automatic-styles>
 <office:master-styles>
  <style:master-page style:name="Standard" style:page-layout-name="pm1"/>
 </office:master-styles>
</office:document-styles>
'''

meta_xml = '''<?xml version="1.0" encoding="UTF-8"?>
<office:document-meta xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" xmlns:meta="urn:oasis:names:tc:opendocument:xmlns:meta:1.0" office:version="1.2">
 <office:meta>
  <meta:title>A Character-Level Neural System for Palestinian Arabizi and Arabic Transliteration</meta:title>
  <meta:keyword>Arabizi, Arabic, transliteration, Transformer, Maknuune</meta:keyword>
 </office:meta>
</office:document-meta>
'''

manifest_items = [
    '<manifest:file-entry manifest:full-path="/" manifest:media-type="application/vnd.oasis.opendocument.text"/>',
    '<manifest:file-entry manifest:full-path="content.xml" manifest:media-type="text/xml"/>',
    '<manifest:file-entry manifest:full-path="styles.xml" manifest:media-type="text/xml"/>',
    '<manifest:file-entry manifest:full-path="meta.xml" manifest:media-type="text/xml"/>',
]

image_paths: list[Path] = []
for path_str, _ in figures:
    path = ROOT / path_str
    if path.exists():
        image_paths.append(path)
        media_type = mimetypes.guess_type(path.name)[0] or "image/png"
        manifest_items.append(f'<manifest:file-entry manifest:full-path="Pictures/{esc(path.name)}" manifest:media-type="{media_type}"/>')

manifest_xml = f'''<?xml version="1.0" encoding="UTF-8"?>
<manifest:manifest xmlns:manifest="urn:oasis:names:tc:opendocument:xmlns:manifest:1.0" manifest:version="1.2">
 {"".join(manifest_items)}
</manifest:manifest>
'''

if OUT.exists():
    OUT.unlink()

with zipfile.ZipFile(OUT, "w") as zf:
    zf.writestr("mimetype", "application/vnd.oasis.opendocument.text", compress_type=zipfile.ZIP_STORED)
    zf.writestr("content.xml", content_xml, compress_type=zipfile.ZIP_DEFLATED)
    zf.writestr("styles.xml", styles_xml, compress_type=zipfile.ZIP_DEFLATED)
    zf.writestr("meta.xml", meta_xml, compress_type=zipfile.ZIP_DEFLATED)
    zf.writestr("META-INF/manifest.xml", manifest_xml, compress_type=zipfile.ZIP_DEFLATED)
    for img in image_paths:
        zf.write(img, f"Pictures/{img.name}", compress_type=zipfile.ZIP_DEFLATED)

print(f"Wrote {OUT}")
print(f"Approximate body word count: {words(all_text)}")
print(f"Embedded figures: {len(image_paths)}")
