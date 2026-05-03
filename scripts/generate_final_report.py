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
    ("working_files/train_harakat_top_5_to_arabizi/train_harakat_top_5_to_arabizi_files/train_harakat_top_5_to_arabizi_files/train_harakat_top_5_to_arabizi_loss_error.png", "Fig. 9. Harakat-top-5-to-Arabizi loss and error trends."),
    ("working_files/train_harakat_top_5_to_arabizi/train_harakat_top_5_to_arabizi_files/train_harakat_top_5_to_arabizi_files/train_harakat_top_5_to_arabizi_sequence_error.png", "Fig. 10. Harakat-top-5-to-Arabizi sequence accuracy trend."),
    ("working_files/train_harakat_top_5_to_arabizi/train_harakat_top_5_to_arabizi_files/train_harakat_top_5_to_arabizi_files/train_harakat_top_5_to_arabizi_top_k_accuracy.png", "Fig. 11. Harakat-top-5-to-Arabizi softmax-90 top-k accuracy."),
    ("working_files/train_harakat_top_5_to_arabizi/train_harakat_top_5_to_arabizi_files/train_harakat_top_5_to_arabizi_files/train_harakat_top_5_to_arabizi_error_by_source_length.png", "Fig. 12. Harakat-top-5-to-Arabizi error rate by source length."),
    ("working_files/train_harakat_top_5_to_arabizi/train_harakat_top_5_to_arabizi_files/train_harakat_top_5_to_arabizi_files/train_harakat_top_5_to_arabizi_error_by_target_length.png", "Fig. 13. Harakat-top-5-to-Arabizi error rate by target length."),
]

body: list[str] = []
body.append(p("A Graduate Project White Paper on Palestinian Arabizi and Arabic Transliteration", "Title"))
body.append(p("Deep Learning 535 Final Report", "Subtitle"))
body.append(p("Executive Summary--This graduate project is framed as a prototype and feasibility study for Palestinian Arabizi and Arabic-script transliteration. It builds and evaluates a small system that converts between Palestinian Arabizi and Arabic script using the Maknuune Palestinian Arabic lexicon as a paired resource. The work trains three character-level Transformer models: an Arabizi-to-Arabic model, an Arabic-without-harakat to Arabic-with-harakat model, and a harakat-top-five-to-Arabizi model. The project is written as a classroom white paper rather than as a completed publication or operational intelligence product. Its purpose is to test whether a small deep-learning pipeline can learn useful transliteration candidates, document the data and modeling decisions, report observed results, and identify what would need to improve before real-world use. Evaluation shows that exact sequence accuracy is demanding because one input may have several plausible spellings, but top-k candidate generation is more useful for an interactive transliterator. The Arabizi-to-Arabic model reaches 75.8 percent test sequence accuracy and 91.7 percent top-10 coverage on a 1,000-example softmax-90 evaluation. The Arabic-to-Arabizi pipeline uses a diacritization stage followed by a top-five harakat candidate transliterator and reaches 98.3 percent top-10 coverage for the second stage.", "Abstract"))
body.append(p("Keywords--Arabizi, Arabic transliteration, Palestinian Arabic, character-level Transformer, diacritization, Maknuune, top-k decoding, OSINT text normalization.", "Keywords"))

body.append(h("I. Introduction"))
intro = [
    "Arabizi is informal Arabic written with Latin letters and numerals. It is common in chat, social media, and situations where Arabic keyboard entry is inconvenient. The writing style is practical for users but difficult for software because it is not a strict one-to-one transliteration scheme. A single Latin character may correspond to several Arabic letters, digits may stand for Arabic sounds with no common Latin equivalent, vowels may be omitted or over-specified, and spelling conventions vary by dialect and by writer. These properties make Arabizi transliteration a sequence prediction problem rather than a simple lookup table.",
    "Arabizi exists for practical historical and social reasons. Many users began writing Arabic with Latin characters because older phones, keyboards, websites, games, and messaging systems did not always make Arabic-script input easy. Even when Arabic keyboards are available, some users continue to prefer Arabizi because it is fast, familiar within peer groups, or better suited to mixed Arabic-English online conversation. Numerals became part of the convention because some Arabic sounds do not have obvious Latin equivalents. For example, writers often use `3` for the sound ع and `7` for ح because the shapes or community conventions make those digits recognizable substitutes.",
    "This problem has practical importance beyond ordinary text normalization. Analysts working with Arabic-language material may encounter dialectal Arabic written in Arabizi form, especially in informal communications, web forums, messaging logs, and social media collections. In open-source intelligence (OSINT) workflows, organizations often monitor public sources for legally and ethically defined keywords, names, places, slogans, event terms, or emerging threat indicators. If those keywords are written in Arabizi rather than Arabic script, an Arabic-script keyword watchlist may miss them. Intelligence and security organizations that lawfully collect Arabic-language data may therefore receive material that Arabic-script search tools, machine translation systems, named-entity recognizers, and triage workflows cannot process directly. A reliable Arabizi-to-Arabic transliterator helps convert that material into a script representation compatible with downstream Arabic NLP. It does not solve translation or interpretation by itself, but it can make otherwise opaque Romanized Arabic text searchable, comparable, and ready for later human or machine analysis.",
    "The goal of this project is to build a usable bidirectional transliterator for Palestinian Arabic. The system is intentionally small enough to run locally, but it still uses modern neural sequence modeling. The frontend lets a user choose a direction, type a short input, and receive stacked candidate outputs. The backend exposes the same behavior through a local API. For Arabizi-to-Arabic, the system runs one model that maps Latin-script input to undiacritized Arabic. For Arabic-to-Arabizi, the system first predicts possible diacritized Arabic forms and then uses those candidate forms as structured input to the Arabizi generator.",
    "The design follows the observation that Arabic script often omits short vowels while Arabizi often expresses them. Direct Arabic-to-Arabizi mapping from unvocalized Arabic can therefore be under-specified. A word such as a bare consonantal Arabic form may have several plausible pronunciations, and each pronunciation may lead to a different Romanized spelling. The project handles this ambiguity by generating up to five harakat hypotheses before generating Arabizi candidates. This choice mirrors the practical behavior of a human reader: first infer likely vocalization, then render the result in a Latin-script convention.",
    "This white paper introduces the task and related work, describes the dataset, presents the three trained models, gives the evaluation metrics and results, documents the deployment approach, and discusses error behavior and limitations. The tone is intentionally project-oriented. The goal is not to claim a finished research contribution, but to show a complete feasibility study: a deep-learning workflow from data cleaning through model training, evaluation, and demonstration. The main engineering theme is candidate preservation. Rather than forcing one answer when the data itself supports several spellings, the application exposes up to ten descending-likelihood transliterations. This makes the tool more honest and more useful for a user who can quickly select the best rendering for a specific word or phrase.",
]
body.extend(p(x) for x in intro)

body.append(h("II. Related Work"))
related = [
    "The core modeling approach is based on neural sequence-to-sequence learning. Sutskever, Vinyals, and Le showed that an encoder-decoder neural network could map variable-length input sequences to variable-length output sequences and achieve strong machine-translation results [1]. Bahdanau, Cho, and Bengio then introduced attention as a way for the decoder to focus on relevant source positions during generation [2]. The Transformer later replaced recurrent computation with multi-head attention and feed-forward layers, improving parallelism and becoming the standard architecture for many sequence transduction tasks [3]. This project uses character-level Transformer encoder-decoder models because transliteration is primarily about spelling and symbol sequences, and because the available data is word-like rather than sentence-scale.",
    "Arabizi transliteration has been studied as a dialectal Arabic NLP problem for more than a decade. Al-Badrashiny, Eskander, Habash, and Rambow describe Romanized dialectal Arabic as informal Latin-script writing that cannot be treated as ordinary letter substitution; their system generated transliteration candidates, filtered them, and selected likely Arabic-script forms [4]. Habash, Diab, and Rambow proposed CODA as a conventional orthography for dialectal Arabic, emphasizing that dialectal Arabic lacks a single naturally standardized written form [5]. These points matter directly here: the model is evaluated against one lexicon spelling, but real users may accept alternatives. That is one reason top-k evaluation is reported alongside strict exact-match accuracy.",
    "Arabic diacritization is also central to this system. Surveys of Arabic diacritization tools note that Arabic text is commonly written without short-vowel marks, even though those marks carry phonological, morphological, and sometimes semantic information [6]. Neural diacritization research has shown that character and sequence models can restore these marks effectively when enough clean supervision is available [7]. In this project, diacritization is not the final user-facing output in the Arabic-to-Arabizi direction. Instead, it is an intermediate representation that supplies possible pronunciations to the Romanization model.",
    "The implementation stack uses PyTorch for training and model serialization. PyTorch is designed around imperative Python execution while still supporting hardware acceleration, which makes it well suited for notebook-driven experimentation and saved `.pt` deployment artifacts [8]. The serving layer uses FastAPI, a Python framework for type-hinted APIs with automatic interactive documentation and production-oriented request handling [9]. The dataset foundation is Maknuune, a large open Palestinian Arabic lexicon with more than 36,000 entries that include diacritized Arabic orthography and phonological information [10].",
]
body.extend(p(x) for x in related)

body.append(h("III. Linguistic Background"))
linguistic = [
    "A key linguistic issue in this project is the difference between Arabic script with and without diacritics. Arabic diacritics, often called harakat, are small marks written above or below letters. They can indicate short vowels, vowel absence, consonant doubling, and other pronunciation information. For example, the consonantal skeleton of a word may stay the same while the harakat change how it is pronounced. In fully vocalized writing, these marks make pronunciation more explicit.",
    "In ordinary Arabic writing, however, most short-vowel diacritics are not written. Newspapers, social media posts, books for adult readers, and everyday messages usually present words mostly as consonants and long-vowel letters. Native and advanced readers infer the missing short vowels from context, grammar, word patterns, and world knowledge. This is natural for human readers because Arabic morphology is patterned and because sentence context usually narrows the possibilities. Children, language learners, religious texts, dictionaries, and teaching materials are more likely to include full or partial diacritics because those contexts require explicit pronunciation guidance.",
    "When diacritics are not included, a lot of information about pronunciation and sometimes meaning can be lost. The same undiacritized Arabic letter sequence can correspond to more than one vocalized word. A reader may infer the intended form from context, but a machine model seeing only a short isolated word has less context than a human reader. This is especially important for transliteration into Arabizi because Arabizi often writes vowels with Latin letters. If the Arabic input does not show the short vowels, the model must guess which vowels should appear in the Arabizi output.",
    "This is why the Arabic-to-Arabizi side of the project uses a two-stage design. The first model predicts possible harakat forms from the plain Arabic input. The second model uses up to five of those vocalized hypotheses to generate Arabizi. The harakat stage is not an academic extra; it addresses a real missing-information problem. Without some estimate of the unwritten vowels and pronunciation marks, the Romanized output would be forced to invent vowel information from a stripped Arabic source.",
    "This project also required building its own harakat adder rather than simply using a general Arabic diacritization model. Many available Arabic diacritizers are designed around Modern Standard Arabic (MSA) or formal written Arabic. That assumption is reasonable for news, books, religious text, or formal documents, but it is not the same as colloquial Palestinian Arabic. Palestinian Arabic differs from MSA in pronunciation, vocabulary, morphology, and common written forms. A model trained mainly on MSA may add diacritics that are grammatically plausible for standard Arabic but wrong for a Palestinian dialect word or phrase.",
    "The transliteration goal makes this mismatch especially costly. The harakat adder is not being used to make formal Arabic look complete; it is being used to infer pronunciation before generating Arabizi. If an outside MSA-oriented diacritizer inserts standard-Arabic vowels, the next model may produce Arabizi that reflects the wrong dialectal pronunciation. Training a project-specific harakat adder on the Maknuune-derived Palestinian data keeps the intermediate vocalization closer to the same dialectal distribution as the Arabizi targets. In other words, the custom harakat adder is a domain-adaptation step. It makes the Arabic-to-Arabizi pipeline internally consistent with colloquial Palestinian Arabic rather than forcing the pipeline through an MSA pronunciation layer.",
]
body.extend(p(x) for x in linguistic)

body.append(h("IV. Dataset and Preprocessing"))
dataset = [
    "The starting point is the cleaned Maknuune-derived file `maknuune-v1.0.1_cleaned.csv`, which contains 36,302 rows and three primary fields: `arabizi`, `arabic_harakat`, and `arabic_stripped`. The `arabizi` field is the Latin-script representation used as the source for Arabizi-to-Arabic training and as the target for Arabic-to-Arabizi training. The `arabic_harakat` field preserves Arabic script with diacritics. The `arabic_stripped` field removes those diacritics and is used as the plain Arabic target or source depending on direction. This compact schema supports all three models without requiring separate annotation passes.",
    "A separate cleaning notebook creates `maknuune-v1.0.1_cleaned_predictions_top_5.csv`, also with 36,302 rows. It begins from harakat predictions produced by the harakat adder and extracts the most likely alternatives from `harakat_predictions_softmax_90`. The output adds `harakat_prediction_1` through `harakat_prediction_5`. This one-time preparation is important because it avoids repeating expensive harakat-candidate parsing every time the top-five Arabizi trainer runs in Google Colab. The cleaned top-five file therefore becomes a reusable training dataset for the second-stage Arabic-to-Arabizi model.",
    "The source-target construction is direction-specific. For Arabizi-to-Arabic, the source is `arabizi` and the target is `arabic_stripped`. For the harakat adder, the source is `arabic_stripped` and the target is `arabic_harakat`. For harakat-top-five-to-Arabizi, the source is the ordered list of up to five harakat predictions, with blanks allowed when fewer than five exist, and the target is `arabizi`. The top-five structure lets the model learn from uncertainty in the preceding diacritization stage instead of pretending that only one vocalized form is possible.",
    "The data split follows the saved notebook configuration: 29,041 training rows, 3,630 validation rows, and 3,631 test rows for the harakat adder, with analogous splits in the transliteration notebooks. Character vocabularies are learned from the training columns, and special tokens handle padding, sequence start, and sequence end. Because this is a course project focused on words and short phrases, character modeling is a practical choice. It preserves digits such as 2, 3, and 7 in Arabizi and preserves Arabic letters and combining marks in the Arabic side without requiring a large word vocabulary.",
]
body.extend(p(x) for x in dataset)

body.append(table([
    ["Artifact", "Rows", "Main columns", "Purpose"],
    ["maknuune-v1.0.1_cleaned.csv", "36,302", "arabizi; arabic_harakat; arabic_stripped", "Base paired lexicon"],
    ["maknuune-v1.0.1_cleaned_predictions_top_5.csv", "36,302", "base columns plus harakat_prediction_1..5", "Reusable input for Arabic-to-Arabizi stage two"],
    ["arabizi_to_arabic_best.pt", "--", "model state; vocabularies; config; metrics", "Best Arabizi-to-Arabic checkpoint"],
    ["harkat_adder.pt", "--", "model state; vocabularies; config; best metrics", "Arabic diacritization checkpoint"],
    ["harakat_top_5_to_arabizi_best.pt", "--", "model state; vocabularies; config; metrics", "Best top-five-to-Arabizi checkpoint"],
], "DatasetArtifacts"))
body.append(caption("Table I. Principal data and model artifacts used by the system."))

body.append(h("V. Model Development"))
modeldev = [
    "All three neural components are character-level encoder-decoder Transformer models. The Arabizi-to-Arabic and harakat-top-five-to-Arabizi models use a model dimension of 192, eight attention heads, three encoder layers, three decoder layers, a feed-forward dimension of 512, dropout of 0.15, and saved vocabularies for source and target characters. The harakat adder is slightly deeper and wider in the feed-forward block: it uses a model dimension of 192, six attention heads, four encoder layers, four decoder layers, and a feed-forward dimension of 768. This reflects the added complexity of predicting Arabic combining marks.",
    "The deep learning choice is central to the project. Arabizi transliteration has many context-dependent character mappings: `7` may represent ح, `3` may represent ع, `2` may represent a hamza-like sound, and Latin vowels can represent short vowels, long vowels, or writer-specific spelling habits. A rule system can encode common substitutions, but it struggles when multiple mappings interact across a whole word. A sequence-to-sequence neural model learns these interactions from paired examples. The encoder converts the entire source character sequence into contextual representations, and the decoder generates the target sequence one character at a time while attending to relevant source positions.",
    "The models operate at character level rather than word level because the vocabulary is small and the task is mostly orthographic. Word-level modeling would create a sparse vocabulary and would fail on unseen spellings. Character-level modeling allows the system to generalize to spelling variants that were not seen as exact tokens during training. It also lets the model preserve numerals and Arabic combining marks as first-class symbols. This is especially important in intelligence or investigative text processing, where informal data often contains rare names, abbreviations, typos, and locally meaningful spellings.",
    "Training was performed in notebooks designed for Colab and Google Drive. The notebooks write progress CSV files, metrics CSV files, plots, best-model checkpoints, and `.pt` bundles into stable folders. Periodic saving matters for Colab because sessions can disconnect. The notebooks therefore save both progress and model artifacts as training proceeds. The final application loads the `.pt` files from the `models` directory and reconstructs the vocabulary and model configuration stored in each bundle.",
    "The Arabizi-to-Arabic notebook trains a direct transliterator from the Latin-script column to unvocalized Arabic script. This model is used whenever the frontend direction is `Arabizi -> Arabic`. At inference time, the backend does not stop after a greedy prediction. Instead, it expands candidate sequences and returns the highest-likelihood unique or non-unique options depending on the requested behavior. The current site preserves duplicates in some pipelines because repeated candidates may reflect distinct upstream paths.",
    "The harakat-adder notebook trains a diacritizer from plain Arabic to diacritized Arabic. In the final application, this model is used only for Arabic-to-Arabizi. The user types Arabic without harakat. The backend asks the harakat adder for up to five softmax-90 candidates. Softmax-90 means the decoder keeps generating alternatives until the retained candidates account for roughly 90 percent of the probability mass or until the configured candidate limit is reached. The goal is not to display these harakat forms to the user, but to pass plausible pronunciations forward.",
    "The harakat model works by treating diacritization as character-level sequence generation. Its source string is Arabic with the harakat removed, and its target string is the same lexical item with the expected Palestinian Arabic diacritics restored. During encoding, the model builds contextual representations for the stripped Arabic characters. During decoding, it generates the vocalized output one character at a time, including ordinary Arabic letters and combining marks. This is more difficult than simply inserting vowels into fixed slots because the output sequence can include additional combining characters and because the correct marks depend on dialectal pronunciation patterns learned from the training data.",
    "The saved metrics show that the harakat adder works best on short-to-medium strings. Its best validation checkpoint is epoch 83, with validation sequence accuracy 0.4983. By length bucket, it performs best on 10-19 character strings, reaching about 0.5710 sequence accuracy. It also performs reasonably on 0-9 character strings, reaching about 0.4815. Accuracy falls sharply after that: 20-29 character strings reach about 0.0313, 30-39 character strings reach about 0.0526, and longer buckets are effectively zero in the saved validation analysis. This means the model is most useful for the short words and phrases that match the project interface and lexicon-style dataset. It should not be treated as a reliable long-sentence diacritizer.",
    "The harakat-top-five-to-Arabizi notebook trains the second Arabic-to-Arabizi stage. It receives up to five candidate vocalized Arabic strings as the source representation and predicts Arabizi. This is more informative than using only stripped Arabic because Arabizi often contains vowel letters that correspond to short vowels or pronunciation choices. During deployment, blank slots are passed when the harakat adder produces fewer than five candidates. The second model then returns up to ten softmax-90 Arabizi outputs stacked in descending likelihood.",
    "The local application is implemented in `app.py`. It loads all three checkpoints, exposes prediction helpers, and serves the `docs/index.html` interface. In this report, the interface is treated as a demonstration layer rather than the main contribution. The main technical work is the model chain, the data preparation, the saved training outputs, and the evaluation of ranked candidate generation. Direction selection controls which model chain is executed, allowing one compact demonstration to show both transliteration directions.",
]
body.extend(p(x) for x in modeldev)

body.append(table([
    ["Model", "Source", "Target", "Layers", "Heads", "Best epoch", "Validation seq. acc.", "Test seq. acc."],
    ["Harakat adder", "arabic_stripped", "arabic_harakat", "4 enc / 4 dec", "6", "83", "0.4983", "not stored in bundle"],
    ["Arabizi to Arabic", "arabizi", "arabic_stripped", "3 enc / 3 dec", "8", "89", "0.7533", "0.7580"],
    ["Harakat top-5 to Arabizi", "harakat_prediction_1..5", "arabizi", "3 enc / 3 dec", "8", "98", "0.6378", "0.6409"],
], "ModelSummary"))
body.append(caption("Table II. Model configurations and sequence-accuracy results from saved artifacts."))

body.append(h("VI. Training Workflow and Reproducibility"))
workflow = [
    "The development process was organized around notebooks because the project required repeated inspection of data, training curves, and example outputs. The notebooks were written to be runnable in Google Colab while saving all durable artifacts to Google Drive under the shared Arabizi project folder. That storage choice is practical: Colab runtimes are temporary, and a long training run can be interrupted by a browser disconnect, hardware recycle, or idle timeout. The notebooks therefore save progress CSVs, figures, and checkpoint files as first-class outputs rather than as incidental notebook state.",
    "Each trainer follows the same broad pattern. First, it imports libraries, mounts Google Drive when running in Colab, and defines a project-specific output directory named after the notebook. Second, it reads the cleaned CSV input and validates that the expected source and target columns are available. Third, it builds source and target character vocabularies, encodes strings into indexed tensors, and splits rows into training, validation, and test partitions. Fourth, it constructs the Transformer model from explicit configuration values and trains with validation checks after each epoch. Fifth, it saves progress metrics and plots so the training run can be evaluated even if the notebook is closed later.",
    "The saved `.pt` files are more than raw neural weights. They include enough metadata for the local backend to reconstruct the exact model: source and target vocabularies, padding indexes, architecture parameters, and selected metrics. This is important because a character-level transliterator is tightly coupled to its vocabulary. If a deployed model were loaded with a different character ordering, predictions would be meaningless even if the tensor shapes matched. Saving the vocabulary and configuration alongside the weights prevents that class of silent error.",
    "The training recipe was intentionally conventional so that the project would be understandable as a graduate deep-learning prototype. The notebooks use seed 42, batch size 128, a maximum of 100 epochs, AdamW optimization with learning rate 3e-4 and weight decay 1e-4, cross-entropy loss with padding ignored, gradient clipping at norm 1.0, and a OneCycle learning-rate schedule. The transliteration notebooks use early stopping patience of 12 validation-loss checks and save checkpoints every five epochs, while the harakat-adder notebook uses patience 10. These details matter because they make the reported results traceable and because they show that the project did not depend on a hidden training procedure.",
    "The notebooks also save graphics as PNG files rather than relying only on inline notebook outputs. This proved useful for reporting. The report can embed the same loss, sequence accuracy, learning-rate, top-k, and length-analysis plots that were generated during training. It also makes the project easier to audit: the CSV files provide numerical detail, the plots show convergence behavior, and the `.pt` files provide the deployable checkpoints. Together, these artifacts form a reproducible chain from cleaned data to web demonstration.",
    "A deliberate design choice was to perform the top-five harakat cleaning once before training the final Arabic-to-Arabizi stage. Without this step, the Colab trainer would need to repeatedly parse the softmax-90 harakat prediction list. That would waste compute and make future training runs slower and less stable. The cleaned top-five file turns the output of the harakat adder into a static training input. The pipeline is therefore split into a preprocessing phase, a model-training phase, and an inference phase. This separation keeps each notebook understandable and keeps the final application fast enough for interactive use.",
    "The training metrics also support model selection. The best checkpoint is selected by validation sequence accuracy, not simply by the final epoch. This matters because training loss can continue to decrease while validation exact-match accuracy plateaus or declines. The Arabizi-to-Arabic model, for example, trained through epoch 100 but selected epoch 89 as the best checkpoint. The harakat-top-five-to-Arabizi model selected epoch 98. The harakat adder selected epoch 83. In all cases, the report uses the saved best-model metrics rather than assuming that the last epoch was best.",
    "The system can be reproduced from the repository by following the artifact chain. Start with the cleaned Maknuune CSV. Run the harakat adder trainer to produce `harkat_adder.pt` and its plots. Run the data-cleaning notebook that extracts top-five harakat candidates from the softmax-90 prediction column. Train the Arabizi-to-Arabic model from `arabizi` to `arabic_stripped`. Train the harakat-top-five-to-Arabizi model from the five candidate columns to `arabizi`. Finally, copy the best checkpoints into the `models/` directory and run the FastAPI application. The HTML frontend then calls the backend and displays ranked candidates.",
]
body.extend(p(x) for x in workflow)

body.append(h("VII. Evaluation Methodology"))
evaltxt = [
    "The notebooks report token accuracy, sequence accuracy, loss, length-bucket breakdowns, and top-k candidate coverage. Token accuracy counts individual output characters. Sequence accuracy is stricter: the entire predicted output must exactly equal the reference string. In transliteration, sequence accuracy can look harsh because a single character difference, an optional vowel, or a plausible alternate spelling counts as a full error. For a user-facing candidate list, top-k accuracy is often more meaningful because the user only needs the desired answer to appear among the returned choices.",
    "Top-k evaluation was intentionally limited to 1,000 samples for the softmax-90 candidate generators. This was done to control processing cost. For each sampled item, the notebook generates candidates in descending likelihood and records whether the gold reference appears in the top 1, 3, 5, or 10. The same analysis is also broken down by source and target length, allowing the report to separate short-token performance from longer phrase behavior.",
    "The length-based error analysis is useful because transliteration errors compound with sequence length. A five-character word may require only a few decisions, while a twenty-character phrase requires many more. Exact-match sequence accuracy therefore normally drops as length increases even when token accuracy remains high. The saved figures and CSV files confirm this pattern for both directions. This is not a failure of the plots; it is a natural property of exact sequence metrics.",
]
body.extend(p(x) for x in evaltxt)

body.append(h("VIII. Results"))
results = [
    "The harakat adder reached its best validation sequence accuracy at epoch 83. Its best validation sequence accuracy was 0.4983, with validation loss 0.2680 and training sequence accuracy 0.5466. By length, the validation accuracy was strongest for 10-19 character items at about 0.5710 and lower for very short items at about 0.4815. It dropped sharply for longer items: 20-29 character items reached 0.0313, and the longest buckets were essentially zero. This shows that the model is useful for short words and short expressions but that full exact diacritization becomes difficult as length grows.",
    "The Arabizi-to-Arabic model completed 100 epochs and selected epoch 89 as the best checkpoint by validation sequence accuracy. At that epoch, validation sequence accuracy was 0.7533, validation token accuracy was 0.9151, and validation loss was 0.2959. The saved test metrics report test loss of 0.2827, test token accuracy of 0.9183, and test sequence accuracy of 0.7580. These numbers indicate that the model usually gets most characters right and exactly matches the reference for about three quarters of held-out examples.",
    "The Arabizi-to-Arabic top-k evaluation is stronger than the single-best metric. On the 1,000-sample softmax-90 run, top-1 accuracy was 0.775, top-3 accuracy was 0.888, top-5 accuracy was 0.904, and top-10 accuracy was 0.917. In other words, when the application is allowed to show ten descending-likelihood Arabic candidates, the reference appears in the list for about 91.7 percent of sampled cases. This justifies the frontend design that stacks candidate outputs instead of presenting only one answer.",
    "The harakat-top-five-to-Arabizi model selected epoch 98 as the best validation checkpoint. At that epoch, validation loss was 0.1152, validation token accuracy was 0.9539, and validation sequence accuracy was 0.6378. The saved test metrics report test loss of 0.1184, test token accuracy of 0.9539, and test sequence accuracy of 0.6409. The top-k evaluation again shows why candidate lists are important: top-1 accuracy was 0.654, top-3 accuracy was 0.922, top-5 accuracy was 0.967, and top-10 accuracy was 0.983.",
    "The contrast between exact sequence accuracy and top-k coverage is the central result of the project. The models are not perfect one-shot normalizers, but they are strong candidate generators. This is especially appropriate for Arabizi because users may spell the same word in different ways and because the reference lexicon contains one accepted spelling even when other outputs may be linguistically plausible. A deterministic one-answer tool would hide that uncertainty. The deployed system instead treats transliteration as ranked retrieval over possible strings.",
]
body.extend(p(x) for x in results)

body.append(table([
    ["Model", "Top-1", "Top-3", "Top-5", "Top-10", "Sample size"],
    ["Arabizi to Arabic", "0.775", "0.888", "0.904", "0.917", "1,000"],
    ["Harakat top-5 to Arabizi", "0.654", "0.922", "0.967", "0.983", "1,000"],
], "TopK"))
body.append(caption("Table III. Softmax-90 top-k coverage for candidate-generating transliteration models."))

body.append(h("IX. Error Analysis"))
analysis = [
    "The Arabizi-to-Arabic error analysis shows a steep length effect. For source strings of 0-9 characters, sequence accuracy was about 0.8214. For 10-19 characters, it fell to about 0.4215. For 20-29 characters, it was about 0.0313. Target-length buckets show the same broad behavior: 0-9 character targets reached about 0.7996, while 10-19 character targets reached only 0.0739. This does not mean the model has no partial knowledge for longer strings; token accuracy remains high. It means that exact full-string matching becomes statistically fragile as more characters must all be correct at once.",
    "The harakat-top-five-to-Arabizi model has a different length profile. It performs well on many mid-length examples because the source representation contains multiple pronunciation hypotheses. Source lengths of 10-19 characters reached about 0.7625 sequence accuracy and 20-29 characters reached about 0.5798, while 30-39 characters dropped to about 0.3793. Target-length buckets show similar degradation as outputs become longer. This suggests that the top-five harakat representation gives the model useful disambiguating information, but it cannot fully remove compounding sequence risk.",
    "A second error source is orthographic multiplicity. Arabizi has no single standard spelling. Long vowels may be doubled or not, English-inspired spellings may compete with Arabic-sound spellings, and numerals may be used inconsistently. If the reference is `salaam`, a model output of `salam` may be readable and plausible but still counted wrong. Likewise, Arabic output may differ in hamza choice, alif form, or final letter while remaining interpretable. Strict metrics are therefore necessary for reproducibility, but they understate practical usefulness when the candidate list contains acceptable alternatives.",
    "The harakat-adder errors are especially consequential in the Arabic-to-Arabizi pipeline. If the first stage misses a vocalization, the second stage may never see the pronunciation needed to generate the desired Arabizi output. The top-five strategy reduces this risk but does not eliminate it. Increasing the harakat candidate count would likely improve coverage but also increase compute cost and may introduce noisy candidates. The current system chooses five as a practical compromise for local inference and Colab-trained data preparation.",
]
body.extend(p(x) for x in analysis)

body.append(table([
    ["Direction", "Length type", "0-9", "10-19", "20-29", "30-39"],
    ["Arabizi to Arabic", "source", "0.8214", "0.4215", "0.0313", "--"],
    ["Arabizi to Arabic", "target", "0.7996", "0.0739", "0.0769", "0.0000"],
    ["Top-5 harakat to Arabizi", "source", "0.6149", "0.7625", "0.5798", "0.3793"],
    ["Top-5 harakat to Arabizi", "target", "0.6290", "0.7637", "0.5709", "0.2949"],
], "LengthAccuracy"))
body.append(caption("Table IV. Selected sequence-accuracy buckets from length-based error analysis."))

body.append(h("X. Threats to Validity and Feasibility Boundaries"))
validity = [
    "Because this is a prototype, the evaluation should be read as evidence of feasibility rather than proof of readiness for operational deployment. The strongest limitation is dataset realism. Maknuune is a clean and valuable Palestinian Arabic lexical resource, but it is not the same as raw OSINT text. Public social-media Arabizi can contain full sentences, emojis, hashtags, usernames, code-switching with English, spelling mistakes, repeated letters, sarcasm, abbreviations, and irregular punctuation. The present system is strongest on words and short phrases that resemble the lexicon-style training examples. A production system would need an external test set drawn from real public Arabizi and would need preprocessing for noisy text.",
    "The OSINT framing also needs careful boundaries. This system is not an intelligence system and should not be treated as one. It does not infer intent, threat, identity, reliability, location, or meaning. It only attempts to normalize script forms so that Arabizi material can be searched or routed into Arabic-language tools. In a responsible OSINT workflow, transliteration would be one early text-processing step used with lawful collection, human review, context, and policy controls. The model can increase recall for keyword monitoring, but it cannot decide whether a keyword match matters.",
    "Another limitation is the absence of a formal baseline comparison. A full study should compare the Transformer models against simpler alternatives such as a hand-built character substitution table, finite-state transliteration rules, edit-distance lookup against the lexicon, or a nearest-neighbor retrieval method. Those baselines would answer whether the neural approach provides enough benefit to justify its complexity. This project partially motivates the neural approach by pointing to ambiguous mappings and strong top-k coverage, but it does not yet quantify gains over a simpler baseline.",
    "The model architecture was also not selected through a full hyperparameter search. The chosen Transformer dimensions are reasonable for a small character-level task, but the study does not prove that 192 hidden dimensions, three or four encoder-decoder layers, dropout 0.15, or eight attention heads are optimal. The current architecture should therefore be understood as a practical prototype configuration, not as an optimized model family. A stronger deep-learning study would train smaller and larger variants and report whether performance is limited by model capacity, data size, or decoding strategy.",
    "Exact-match metrics also have limits. Arabizi has many acceptable spellings, and Palestinian Arabic itself can have multiple plausible Arabic-script renderings. A model output can be useful to a human analyst while still failing exact sequence accuracy against a single reference. Top-k metrics address this problem by measuring whether the reference appears among ranked candidates, but they still rely on one gold answer. A stronger evaluation would add character error rate, edit distance, and human judgments of whether non-reference candidates are acceptable.",
    "The current top-k evaluation is informative but incomplete. It uses a 1,000-example softmax-90 sample to manage compute cost, so the reported top-k values should be treated as estimates. The report does not include confidence intervals, random-seed variation, or cross-validation. It also selects best checkpoints by validation sequence accuracy rather than by validation top-k coverage, even though the deployed interface emphasizes candidate lists. Future experiments should align the selection metric more closely with the final use case.",
    "The harakat stage is a known bottleneck. Its best validation sequence accuracy is about 49.8 percent, and its performance drops sharply on strings longer than about twenty characters. The Arabic-to-Arabizi pipeline can only use the pronunciation hypotheses the harakat adder provides. If the correct vocalization is missing from the top five, the second-stage model may never generate the best Arabizi output. This is acceptable for a feasibility prototype focused on short inputs, but it is not sufficient for long sentence processing without segmentation, more data, or a stronger dialect-specific diacritizer.",
    "A related deep-learning issue is exposure bias. During training, the decoder learns with the correct previous target characters available through teacher-forced targets. During inference, it must condition on its own earlier predictions. A small early mistake can therefore compound, which helps explain why exact-match accuracy drops sharply as strings become longer. This is a normal challenge for autoregressive sequence models, but it should be named because it affects how the length-based results are interpreted.",
    "Unicode and normalization choices are another possible source of error. Arabic script contains multiple alif and hamza forms, ta marbuta, alif maqsura, and combining marks that can be represented in ways that look similar to a reader but differ as Unicode strings. The current project relies on the cleaned data and model vocabularies, but a full benchmark would document normalization rules explicitly and report whether evaluation was performed on raw strings, normalized strings, or both.",
    "Dialect scope is also intentionally narrow. The trained system is Palestinian-focused because the data is Palestinian-focused. Arabizi conventions vary across the Arabic-speaking world, and dialects differ in sound systems, vocabulary, and morphology. A Gulf, Egyptian, Iraqi, Maghrebi, or broader Levantine deployment would need dialect-specific data or a multilingual dialect design. The current model should therefore be described as Palestinian Arabic transliteration, not as a general Arabic or pan-Arabizi solution.",
    "Finally, the top-k interface trades automation for recall. Showing up to ten candidates improves the chance that a useful transliteration appears, but it also creates a selection burden for a human or downstream system. In future work, calibrated probabilities, candidate grouping, duplicate handling, and downstream reranking could make the candidate list easier to use. For this course project, top-k output is best understood as a transparent way to preserve uncertainty while demonstrating that the models have learned useful mappings.",
]
body.extend(p(x) for x in validity)

body.append(h("XI. Figures and Training Graphics"))
body.append(p("The following figures are embedded from the saved training-output folders. They document the model behavior described above: loss and error trends, learning-rate schedules, sequence accuracy, top-k coverage, and error rates by length. Including the plots in the report is useful because the CSV tables give exact values while the graphics reveal whether training converged smoothly or oscillated. The curves also show why the saved best checkpoints are not always the final epoch; validation sequence accuracy can peak before the last epoch even when training loss continues to improve."))
for i, (path_str, cap) in enumerate(figures, start=1):
    path = ROOT / path_str
    if path.exists():
        body.append(image(path, i, cap))

body.append(h("XII. Operational Use and Deployment"))
deployment = [
    "The most important deployment point is that the trained models turn informal Romanized Arabic into a form that can enter an Arabic NLP workflow. In an OSINT or intelligence-analysis setting, collected public text may arrive in a mixture of Arabic script, English, numerals, emojis, and Arabizi. Keyword monitoring is often one of the first filters in such workflows: analysts maintain watchlists for names, locations, organizations, topics, or time-sensitive indicators and then review matching public posts or documents. If Arabizi is left untreated, Arabic-script keyword searches and translation tools may miss relevant material. A transliteration stage can normalize candidate strings into Arabic script so analysts can search across spelling systems, compare related messages, and route material to downstream translation or entity-extraction systems.",
    "The deployed system has three layers. The first layer is static HTML and JavaScript in `docs/index.html`. It provides direction selection, dialect selection, a short input box, action buttons, and a ranked output panel. This interface exists mainly as a demonstration and testing surface. The deeper contribution is the model-serving logic behind it: the API exposes ranked neural transliteration candidates that could be called by another application, batch-processing script, or analyst workflow.",
    "The second layer is the FastAPI backend in `app.py`. It loads the checkpoints from `models/`, rebuilds each model from saved configuration, and exposes candidate prediction behavior. For `Arabizi -> Arabic`, it calls the Arabizi-to-Arabic model and returns up to ten Arabic candidates. For `Arabic -> Arabizi`, it calls the harakat adder, keeps up to five vocalized candidates, pads the missing slots if needed, sends that structured input into the top-five-to-Arabizi model, and returns up to ten Arabizi candidates. The output is stacked in descending likelihood.",
    "The third layer is the trained model bundle itself. Each `.pt` file contains more than model weights. The bundles also carry vocabulary mappings, configuration dictionaries, and metrics. This makes inference reproducible: the backend can load the exact vocabulary and architecture used during training instead of hard-coding assumptions in the service. It also makes the report auditable because the same saved files provide the performance numbers reported here.",
    "The demonstration site is intentionally conservative. The twenty-character limit prevents slow or unpredictable local inference on long inputs. The output area is sized for candidate inspection rather than paragraph translation. The interface does not claim to translate meaning; it transliterates script. This distinction is important for operational use. Transliteration maps written forms, while translation maps semantic content across languages. In a real analytic pipeline, transliteration would be an early normalization component, not the final intelligence product.",
    "Candidate lists are also operationally useful. When collected Arabizi text is ambiguous, a single forced transliteration can hide uncertainty and may cause a downstream keyword search to miss the right spelling. Returning ranked alternatives allows a human analyst or later pipeline stage to preserve ambiguity. For example, an entity mention, place name, slogan, or colloquial phrase may have several plausible Arabic-script forms. A top-k transliterator gives downstream OSINT tools more chances to match the intended form while still ordering candidates by learned likelihood. The same logic can also work in reverse: Arabic keywords can be expanded into likely Arabizi spellings so public-source monitoring is not limited to one script.",
]
body.extend(p(x) for x in deployment)

body.append(h("XIII. Limitations and Future Work"))
limits = [
    "The most important limitation is data scope. Maknuune is a high-quality Palestinian Arabic lexicon, but it is still primarily lexical. The current models are best for words and short expressions, not long conversational sentences. The length analysis shows why: exact sequence accuracy degrades quickly as the number of characters increases. Future work should add sentence-level data, context-aware decoding, and perhaps a language model for reranking multi-token outputs.",
    "A second limitation is evaluation against a single reference. The top-k analysis partly addresses this by showing candidate coverage, but it still checks candidates against one gold string. Human evaluation would better capture acceptable spelling variants. For example, multiple Arabizi strings can reasonably map to the same Arabic form, and multiple Arabic orthographies can be readable for the same dialectal word. A future benchmark could include sets of acceptable outputs or use edit-distance thresholds alongside exact match.",
    "A third limitation is model size and training budget. The models are intentionally small and were trained in notebook workflows suitable for Colab. Larger Transformers, byte-pair or unigram subword vocabularies, data augmentation, and pronunciation-aware features could improve robustness. However, larger models also increase serving cost. The present design favors a lightweight local application that can be understood, moved, and demonstrated without a production ML platform.",
    "The Arabic-to-Arabizi pipeline also depends on intermediate harakat quality. If the harakat adder fails, the second-stage transliterator receives incomplete evidence. A future version could train the Arabic-to-Arabizi direction end-to-end with latent pronunciation candidates, or jointly train the two stages so that the first stage optimizes downstream Arabizi coverage rather than only diacritized Arabic exact match. Another useful extension would be confidence calibration, showing probabilities or visual confidence bars beside each candidate.",
    "A final limitation is that the current system is a transliteration demonstrator rather than a complete dialectal Arabic text-processing platform. It does not segment clitics, identify named entities, normalize emojis or punctuation-heavy social media text, or condition on surrounding sentence context. Those tasks are common in real Arabizi data and would become important in intelligence, law-enforcement, humanitarian, or research settings. The present scope was narrower by design: build a transparent deep-learning pipeline from a reliable Palestinian lexicon, preserve candidate uncertainty, and make the trained models usable through a simple local interface. That narrower scope makes the model behavior easier to explain and easier to improve incrementally.",
    "A future version should add baseline experiments. The most useful baselines would be a rule-based Arabizi character mapper, an edit-distance lexicon retriever, and a simple lookup model. If the neural models beat those baselines on exact accuracy, top-k coverage, and edit distance, the case for deep learning would be much stronger. If a baseline performs similarly on short words, then the neural model's value may be strongest for ambiguous spellings, unseen variants, or ranked candidate generation.",
    "A future version should also include human evaluation and external data. Human bilingual or dialect-aware annotators could judge whether candidates are acceptable even when they do not match the single reference. An external OSINT-style test set, collected ethically from public examples or manually created to simulate noisy public text, would measure whether the model generalizes beyond the clean Maknuune distribution. These additions would move the work from feasibility prototype toward a more convincing applied NLP study.",
    "The highest-priority deep-learning improvements are ablations and stricter validation. The Arabic-to-Arabizi side should be tested with no harakat stage, with only the top-1 harakat candidate, and with the current top-5 harakat candidates. That would show how much the custom diacritizer contributes. The decoder should also be compared under greedy decoding, ordinary beam search, and the current softmax-90 candidate strategy. These experiments would not require changing the basic data pipeline, but they would make the model-design argument more convincing.",
    "Another useful future step would be a stricter data split. Random row splits may allow related forms, shared lemmas, or very similar spellings to appear in both training and validation/test sets. A lemma-aware, root-aware, or string-similarity-aware split would better test generalization. If performance drops under that stricter split, the result would not invalidate the prototype; it would clarify whether the current model is learning productive character correspondences or partly memorizing lexicon entries.",
    "Longer-term work could explore pretrained or joint models. ByT5, mT5, AraBERT-derived systems, or other pretrained sequence models might improve robustness to noisy real-world text, though they would be larger and less transparent than the current prototype. A joint Arabic-to-Arabizi model could also learn the harakat and Romanization steps together, reducing pipeline error propagation. Those directions are outside the time and scope of this course project, but they are natural next steps if the prototype is extended into a larger research effort.",
]
body.extend(p(x) for x in limits)

body.append(h("XIV. Conclusion"))
conclusion = [
    "This project produced a working feasibility prototype for Palestinian Arabizi-Arabic transliteration built from cleaned Maknuune data, three character-level Transformer models, saved training notebooks, plotted diagnostics, and a compact web interface. The strongest practical result is not only the single-best accuracy but the high top-k coverage. Arabizi-to-Arabic reaches 91.7 percent top-10 coverage on the sampled softmax-90 evaluation, and the harakat-top-five-to-Arabizi stage reaches 98.3 percent top-10 coverage. These numbers support the prototype design: return ranked candidates and preserve uncertainty rather than pretending there is always one obvious answer.",
    "The report also shows why the system is architected as a pipeline. Arabic without harakat does not fully specify pronunciation, while Arabizi often represents vowels. Adding a harakat prediction stage gives the Arabic-to-Arabizi model more useful evidence. This pipeline is not perfect, but it directly addresses an ambiguity that a one-step stripped-Arabic-to-Arabizi model would have to learn implicitly. The saved artifacts, metrics, and graphics make the development process reproducible and give clear targets for future improvement.",
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
    "[9] S. Ramirez, \"FastAPI Documentation,\" FastAPI, available: https://fastapi.tiangolo.com/.",
    "[10] S. Dibas et al., \"Maknuune: A Large Open Palestinian Arabic Lexicon,\" WANLP, 2022, arXiv:2210.12985.",
    "[11] B. Talafha, A. Abuammar, and M. Al-Ayyoub, \"Atar: Attention-Based LSTM for Arabizi Transliteration,\" International Journal of Electrical and Computer Engineering, vol. 11, no. 3, pp. 2327-2334, 2021.",
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
