#################################
# app.py
# FastAPI backend for Arabizi <-> Arabic transliteration
#################################

import math
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


#################################
# basic setup
#################################

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_DIR = Path("models")
ARABIZI_TO_ARABIC_PATH = MODEL_DIR / "arabizi_to_arabic_website_bundle.pt"
ARABIC_TO_ARABIZI_PATH = MODEL_DIR / "arabic_to_arabizi_website_bundle.pt"
ARABIZI_TO_ARABIC_RERANKER_PATH = MODEL_DIR / "arabizi_to_arabic_reranker_best.pt"
ARABIC_TO_ARABIZI_RERANKER_PATH = MODEL_DIR / "arabic_to_arabizi_reranker_best.pt"

PAD = "<PAD>"
BOS = "<BOS>"
EOS = "<EOS>"
UNK = "<UNK>"

DEFAULT_CONFIG = {
    "D_MODEL": 320,
    "NHEAD": 8,
    "NUM_ENCODER_LAYERS": 5,
    "NUM_DECODER_LAYERS": 5,
    "DIM_FEEDFORWARD": 1280,
    "DROPOUT": 0.20,
    "DEFAULT_DECODER": "beam",
    "BEAM_SIZE": 20,
    "MAX_WEB_BEAM_SIZE": 10,
    "BEAM_LENGTH_PENALTY": 0.80,
    "RETURN_TOP_K_PREDICTIONS": 10,
    "NORMALIZE_UNICODE_FORM": "NFC",
    "LOWERCASE_ARABIZI": True,
    "COLLAPSE_EXTRA_WHITESPACE": True,
    "STRIP_TEXT": True,
    "STRIP_ARABIC_TATWEEL": True,
    "NORMALIZE_ARABIC_ALEF_VARIANTS": True,
    "NORMALIZE_ARABIC_YAA_VARIANTS": True,
    "NORMALIZE_ARABIC_TAA_MARBUTA": False,
    "NORMALIZE_ARABIZI_APOSTROPHES": True,
    "COLLAPSE_ARABIZI_CHAR_REPEATS_IN_NORMALIZATION": True,
    "MAX_CONSECUTIVE_ARABIZI_REPEATS": 2,
    "ARABIZI_MULTI_TOKENS": [
        "sch", "tsh",
        "sh", "kh", "gh", "th", "dh", "ch", "ph", "zh",
        "2'", "3'", "5'", "6'", "7'", "8'", "9'",
        "7a", "7o", "7e", "3a", "3o", "3e", "2a", "2i", "2o",
        "5a", "5o", "5e", "6a", "6o", "6e", "8a", "8o", "8e", "9a", "9o", "9e",
        "aa", "ii", "uu", "oo", "ee",
        "ou", "aw", "ay",
    ],
}

ARABIC_ALEF_VARIANTS = {"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا"}
ARABIC_YAA_VARIANTS = {"ى": "ي", "ئ": "ي"}
ARABIC_TATWEEL = "ـ"
ARABIZI_APOSTROPHE_VARIANTS = ["’", "‘", "`", "´", "ʾ", "ʼ", "ʻ"]


#################################
# api setup
#################################

app = FastAPI(title="Arabizi Transliteration API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TransliterationRequest(BaseModel):
    text: str
    direction: str
    dialect: str = "Palestinian Arabic"
    top_k: int = 10


class TransliterationResponse(BaseModel):
    candidates: List[str]
    status: str


#################################
# model definition from notebook
#################################

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=2048):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)

        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model,
        nhead,
        num_encoder_layers,
        num_decoder_layers,
        dim_feedforward,
        dropout,
        pad_idx,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.d_model = d_model

        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)
        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.output_layer = nn.Linear(d_model, tgt_vocab_size)

    def make_src_key_padding_mask(self, src):
        return src.eq(self.pad_idx)

    def make_tgt_key_padding_mask(self, tgt):
        return tgt.eq(self.pad_idx)

    def forward(self, src, tgt_input):
        src_padding_mask = self.make_src_key_padding_mask(src)
        tgt_padding_mask = self.make_tgt_key_padding_mask(tgt_input)

        tgt_seq_len = tgt_input.size(1)
        tgt_mask = torch.triu(
            torch.ones(tgt_seq_len, tgt_seq_len, device=tgt_input.device, dtype=torch.bool),
            diagonal=1,
        )

        src_emb = self.positional_encoding(self.src_embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.positional_encoding(self.tgt_embedding(tgt_input) * math.sqrt(self.d_model))

        hidden = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask,
        )
        return self.output_layer(hidden)


class RerankerMLP(nn.Module):
    """MLP reranker that scores candidates based on various features."""
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        # Match the exact layer structure from the saved model (net.0, net.3, net.5)
        self.net = nn.ModuleDict({
            "0": nn.Linear(input_dim, hidden_dim),
            "3": nn.Linear(hidden_dim, hidden_dim),
            "5": nn.Linear(hidden_dim, 1),
        })
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.net["0"](x))
        x = self.relu(self.net["3"](x))
        x = self.net["5"](x)
        return x.squeeze(-1)


#################################
# text helpers from notebook
#################################

def get_config(bundle: Dict[str, Any]) -> Dict[str, Any]:
    cfg = DEFAULT_CONFIG.copy()
    cfg.update(bundle.get("user_config", {}) or {})
    return cfg


def compress_repeated_characters(text: str, max_repeats: int = 2) -> str:
    if max_repeats < 1:
        return text
    pattern = re.compile(r"(.)\1{" + str(max_repeats) + r",}")
    return pattern.sub(lambda m: m.group(1) * max_repeats, text)


def normalize_text(x: str, cfg: Dict[str, Any], lowercase: bool = False) -> str:
    x = "" if x is None else str(x)

    unicode_form = cfg.get("NORMALIZE_UNICODE_FORM")
    if unicode_form:
        x = unicodedata.normalize(unicode_form, x)

    has_arabic = any("\u0600" <= ch <= "\u06FF" for ch in x)

    if has_arabic:
        if cfg.get("STRIP_ARABIC_TATWEEL", True):
            x = x.replace(ARABIC_TATWEEL, "")
        if cfg.get("NORMALIZE_ARABIC_ALEF_VARIANTS", True):
            x = "".join(ARABIC_ALEF_VARIANTS.get(ch, ch) for ch in x)
        if cfg.get("NORMALIZE_ARABIC_YAA_VARIANTS", True):
            x = "".join(ARABIC_YAA_VARIANTS.get(ch, ch) for ch in x)
        if cfg.get("NORMALIZE_ARABIC_TAA_MARBUTA", False):
            x = x.replace("ة", "ه")
    else:
        if cfg.get("NORMALIZE_ARABIZI_APOSTROPHES", True):
            for apostrophe in ARABIZI_APOSTROPHE_VARIANTS:
                x = x.replace(apostrophe, "'")

    if cfg.get("STRIP_TEXT", True):
        x = x.strip()

    if cfg.get("COLLAPSE_EXTRA_WHITESPACE", True):
        x = re.sub(r"\s+", " ", x)

    if lowercase:
        x = x.lower()

    if (
        cfg.get("COLLAPSE_ARABIZI_CHAR_REPEATS_IN_NORMALIZATION", True)
        and not any("\u0600" <= ch <= "\u06FF" for ch in x)
    ):
        x = compress_repeated_characters(
            x,
            max_repeats=int(cfg.get("MAX_CONSECUTIVE_ARABIZI_REPEATS", 2)),
        )

    return x


def arabic_char_tokenize(text: str) -> List[str]:
    return list(text)


def arabizi_aware_tokenize(text: str, cfg: Dict[str, Any]) -> List[str]:
    text = normalize_text(text, cfg, lowercase=bool(cfg.get("LOWERCASE_ARABIZI", True)))
    multi_tokens = sorted(cfg.get("ARABIZI_MULTI_TOKENS", DEFAULT_CONFIG["ARABIZI_MULTI_TOKENS"]), key=len, reverse=True)

    tokens = []
    i = 0
    while i < len(text):
        if text[i].isspace():
            tokens.append(" ")
            i += 1
            continue

        matched = False
        for chunk in multi_tokens:
            if text.startswith(chunk, i):
                tokens.append(chunk)
                i += len(chunk)
                matched = True
                break

        if not matched:
            tokens.append(text[i])
            i += 1

    return tokens


def tokenize_text_by_script(text: str, script: str, cfg: Dict[str, Any]) -> List[str]:
    if script == "arabizi":
        return arabizi_aware_tokenize(text, cfg)
    if script == "arabic":
        normalized = normalize_text(text, cfg, lowercase=False)
        return arabic_char_tokenize(normalized)
    raise ValueError(f"Unknown script: {script}")


def detokenize_text(tokens: List[str], script: str) -> str:
    if script in {"arabic", "arabizi"}:
        return "".join(tokens)
    raise ValueError(f"Unknown script: {script}")


def id_to_token(itos: Any, idx: int) -> str:
    if isinstance(itos, dict):
        return itos.get(idx, itos.get(str(idx), UNK))
    return itos[idx]


def encode_tokens(tokens: List[str], stoi: Dict[str, int], max_len: int) -> List[int]:
    encoded = [stoi[BOS]]
    encoded.extend(stoi.get(tok, stoi[UNK]) for tok in tokens)
    encoded.append(stoi[EOS])

    encoded = encoded[:max_len]
    if encoded[-1] != stoi[EOS]:
        encoded[-1] = stoi[EOS]

    if len(encoded) < max_len:
        encoded.extend([stoi[PAD]] * (max_len - len(encoded)))

    return encoded


def decode_token_ids(token_ids: List[int], tgt_itos: Any, target_script: str) -> str:
    tokens = []
    for token_id in token_ids:
        tok = id_to_token(tgt_itos, int(token_id))
        if tok == EOS:
            break
        if tok in {BOS, PAD}:
            continue
        tokens.append(tok)
    return detokenize_text(tokens, target_script)


def scripts_for_direction(direction: str):
    normalized = direction.strip().lower().replace("→", "->")
    if normalized in {"arabizi -> arabic", "arabizi_to_arabic", "arabizi-to-arabic"}:
        return "arabizi_to_arabic", "arabizi", "arabic"
    if normalized in {"arabic -> arabizi", "arabic_to_arabizi", "arabic-to-arabizi"}:
        return "arabic_to_arabizi", "arabic", "arabizi"
    raise ValueError(f"Unknown direction: {direction}")


#################################
# loading bundles and models
#################################

def load_bundle(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing model file: {path.resolve()}")
    return torch.load(path, map_location=DEVICE)


def build_model_from_bundle(bundle: Dict[str, Any]) -> Seq2SeqTransformer:
    cfg = get_config(bundle)

    pad_idx = bundle["src_stoi"][PAD]

    model = Seq2SeqTransformer(
        src_vocab_size=len(bundle["src_itos"]),
        tgt_vocab_size=len(bundle["tgt_itos"]),
        d_model=int(cfg["D_MODEL"]),
        nhead=int(cfg["NHEAD"]),
        num_encoder_layers=int(cfg["NUM_ENCODER_LAYERS"]),
        num_decoder_layers=int(cfg["NUM_DECODER_LAYERS"]),
        dim_feedforward=int(cfg["DIM_FEEDFORWARD"]),
        dropout=float(cfg["DROPOUT"]),
        pad_idx=pad_idx,
    ).to(DEVICE)

    model.load_state_dict(bundle["model_state_dict"])
    model.eval()
    return model


BUNDLES: Dict[str, Dict[str, Any]] = {}
MODELS: Dict[str, Seq2SeqTransformer] = {}
RERANKERS: Dict[str, RerankerMLP] = {}

try:
    BUNDLES["arabizi_to_arabic"] = load_bundle(ARABIZI_TO_ARABIC_PATH)
    BUNDLES["arabic_to_arabizi"] = load_bundle(ARABIC_TO_ARABIZI_PATH)

    MODELS["arabizi_to_arabic"] = build_model_from_bundle(BUNDLES["arabizi_to_arabic"])
    MODELS["arabic_to_arabizi"] = build_model_from_bundle(BUNDLES["arabic_to_arabizi"])

    # Load rerankers
    if ARABIZI_TO_ARABIC_RERANKER_PATH.exists():
        reranker_bundle = torch.load(ARABIZI_TO_ARABIC_RERANKER_PATH, map_location=DEVICE)
        reranker = RerankerMLP(
            input_dim=reranker_bundle["input_dim"],
            hidden_dim=reranker_bundle.get("hidden_dim", 128)
        ).to(DEVICE)
        reranker.load_state_dict(reranker_bundle["model_state_dict"])
        reranker.eval()
        RERANKERS["arabizi_to_arabic"] = reranker

    if ARABIC_TO_ARABIZI_RERANKER_PATH.exists():
        reranker_bundle = torch.load(ARABIC_TO_ARABIZI_RERANKER_PATH, map_location=DEVICE)
        reranker = RerankerMLP(
            input_dim=reranker_bundle["input_dim"],
            hidden_dim=reranker_bundle.get("hidden_dim", 128)
        ).to(DEVICE)
        reranker.load_state_dict(reranker_bundle["model_state_dict"])
        reranker.eval()
        RERANKERS["arabic_to_arabizi"] = reranker

except Exception as exc:
    # Keep the import error readable in the terminal.
    raise RuntimeError(
        "Could not load model bundles. Make sure these files exist:\n"
        f"  {ARABIZI_TO_ARABIC_PATH}\n"
        f"  {ARABIC_TO_ARABIZI_PATH}\n"
        f"Original error: {exc}"
    ) from exc


#################################
# decoding
#################################

def prepare_source_ids(text: str, bundle: Dict[str, Any], source_script: str) -> torch.Tensor:
    cfg = get_config(bundle)
    source_tokens = tokenize_text_by_script(text, source_script, cfg)
    source_ids = encode_tokens(source_tokens, bundle["src_stoi"], int(bundle["max_src_len"]))
    return torch.tensor([source_ids], dtype=torch.long, device=DEVICE)


@torch.no_grad()
def greedy_decode(model: Seq2SeqTransformer, text: str, bundle: Dict[str, Any], source_script: str, target_script: str) -> str:
    src_ids = prepare_source_ids(text, bundle, source_script)
    max_tgt_len = int(bundle["max_tgt_len"])
    bos_id = bundle["tgt_stoi"][BOS]
    eos_id = bundle["tgt_stoi"][EOS]

    generated = [bos_id]

    for _ in range(max_tgt_len - 1):
        tgt_input = torch.tensor([generated], dtype=torch.long, device=DEVICE)
        logits = model(src_ids, tgt_input)
        next_id = int(logits[0, -1].argmax().item())
        generated.append(next_id)
        if next_id == eos_id:
            break

    return decode_token_ids(generated, bundle["tgt_itos"], target_script)


@torch.no_grad()
def beam_search_decode_nbest(
    model: Seq2SeqTransformer,
    text: str,
    bundle: Dict[str, Any],
    source_script: str,
    target_script: str,
    beam_size: int = 20,
    length_penalty: float = 0.80,
    top_k: int = 10,
) -> List[str]:
    src_ids = prepare_source_ids(text, bundle, source_script)
    max_tgt_len = int(bundle["max_tgt_len"])
    bos_id = bundle["tgt_stoi"][BOS]
    eos_id = bundle["tgt_stoi"][EOS]

    beams = [([bos_id], 0.0)]
    finished = []

    for _ in range(max_tgt_len - 1):
        candidates = []

        for token_ids, score in beams:
            if token_ids[-1] == eos_id:
                finished.append((token_ids, score))
                continue

            tgt_input = torch.tensor([token_ids], dtype=torch.long, device=DEVICE)
            logits = model(src_ids, tgt_input)
            log_probs = F.log_softmax(logits[0, -1], dim=-1)

            values, indices = torch.topk(log_probs, k=min(beam_size, log_probs.size(0)))
            for value, idx in zip(values.tolist(), indices.tolist()):
                candidates.append((token_ids + [int(idx)], score + float(value)))

        if not candidates:
            break

        def normalized_score(item):
            token_ids, raw_score = item
            length = max(len(token_ids) - 1, 1)
            return raw_score / (length ** length_penalty)

        candidates.sort(key=normalized_score, reverse=True)
        beams = candidates[:beam_size]

        if len(finished) >= top_k and all(b[0][-1] == eos_id for b in beams):
            break

    finished.extend(beams)

    # Sort, detokenize, and remove duplicate strings while preserving rank.
    def normalized_score(item):
        token_ids, raw_score = item
        length = max(len(token_ids) - 1, 1)
        return raw_score / (length ** length_penalty)

    finished.sort(key=normalized_score, reverse=True)

    outputs = []
    seen = set()
    for token_ids, _ in finished:
        text_out = decode_token_ids(token_ids, bundle["tgt_itos"], target_script).strip()
        if text_out and text_out not in seen:
            outputs.append(text_out)
            seen.add(text_out)
        if len(outputs) >= top_k:
            break

    return outputs


def extract_reranker_features(
    source: str,
    candidates: List[tuple],
    task_name: str,
    bundle: Dict[str, Any],
) -> torch.Tensor:
    """Extract features for the reranker model."""
    import math

    source_len = len(source)
    source_digit_count = sum(c.isdigit() for c in source)
    source_space_count = sum(c.isspace() for c in source)
    source_vowel_like_count = sum(c in "aeiouAEIOUàáâãäåæèéêëìíîïòóôõöøùúûüýÿ" for c in source)
    source_has_apostrophe = "'" in source
    source_has_space = " " in source

    # Get beam scores from candidates (they're already sorted by beam score)
    beam_scores = [score for _, score in candidates]

    # Normalize beam scores
    if beam_scores:
        max_score = max(beam_scores)
        min_score = min(beam_scores)
        score_range = max_score - min_score if max_score != min_score else 1.0
    else:
        max_score = min_score = score_range = 0

    features = []
    for rank, (token_ids, score) in enumerate(candidates):
        # Decode candidate to get text
        candidate = decode_token_ids(token_ids, bundle["tgt_itos"], "arabic" if task_name == "arabizi_to_arabic" else "arabizi")

        candidate_len = len(candidate)
        candidate_digit_count = sum(c.isdigit() for c in candidate)
        candidate_space_count = sum(c.isspace() for c in candidate)
        candidate_vowel_like_count = sum(c in "aeiouAEIOUàáâãäåæèéêëìíîïòóôõöøùúûüýÿ" for c in candidate)
        candidate_has_space = " " in candidate

        # Lexical matches (simple character overlap)
        source_chars = set(source.lower())
        candidate_chars = set(candidate)
        arabic_lex_match = sum(1 for c in candidate if "\u0600" <= c <= "\u06FF") / max(len(candidate), 1)
        arabizi_lex_match = sum(1 for c in candidate if c.isascii() and c.isalpha()) / max(len(candidate), 1)

        # Target frequency (placeholder - would need vocab stats)
        log_target_frequency = 0.0

        # Beam rank features
        beam_rank = rank + 1
        score_gap_from_best = max_score - score
        normalized_score_gap_from_best = score_gap_from_best / score_range if score_range > 0 else 0

        # Contains digits
        contains_2 = "2" in candidate
        contains_3 = "3" in candidate
        contains_5 = "5" in candidate
        contains_6 = "6" in candidate
        contains_7 = "7" in candidate
        contains_8 = "8" in candidate
        contains_9 = "9" in candidate

        feat = [
            score,  # candidate_score
            (score - min_score) / score_range if score_range > 0 else 0,  # candidate_normalized_score
            source_len,  # source_len
            candidate_len,  # candidate_len
            candidate_len - source_len,  # candidate_len_minus_source_len
            candidate_len / source_len if source_len > 0 else 0,  # candidate_len_ratio_to_source
            source_digit_count,  # source_digit_count
            candidate_digit_count,  # candidate_digit_count
            abs(candidate_digit_count - source_digit_count),  # abs_digit_count_diff
            source_space_count,  # source_space_count
            candidate_space_count,  # candidate_space_count
            abs(candidate_space_count - source_space_count),  # abs_space_count_diff
            source_vowel_like_count,  # source_vowel_like_count
            candidate_vowel_like_count,  # candidate_vowel_like_count
            candidate_vowel_like_count / source_vowel_like_count if source_vowel_like_count > 0 else 0,  # candidate_vowel_ratio_to_source
            arabic_lex_match,  # arabic_lex_match
            arabizi_lex_match,  # arabizi_lex_match
            log_target_frequency,  # log_target_frequency
            beam_rank,  # beam_rank
            score_gap_from_best,  # score_gap_from_best
            normalized_score_gap_from_best,  # normalized_score_gap_from_best
            float(contains_2),  # contains_2
            float(contains_3),  # contains_3
            float(contains_5),  # contains_5
            float(contains_6),  # contains_6
            float(contains_7),  # contains_7
            float(contains_8),  # contains_8
            float(contains_9),  # contains_9
            float(source_has_apostrophe),  # source_has_apostrophe
            float(source_has_space),  # source_has_space
            float(candidate_has_space),  # candidate_has_space
        ]
        features.append(feat)

    return torch.tensor(features, dtype=torch.float32, device=DEVICE)


def rerank_candidates(
    source: str,
    candidates: List[tuple],
    task_name: str,
    bundle: Dict[str, Any],
    reranker: RerankerMLP,
) -> List[str]:
    """Rerank candidates using the MLP reranker."""
    if not candidates:
        return []

    # Extract features
    features = extract_reranker_features(source, candidates, task_name, bundle)

    # Get reranker scores
    with torch.no_grad():
        rerank_scores = reranker(features)

    # Combine beam score with reranker score (weighted)
    rerank_scores = rerank_scores.cpu().tolist()

    # Re-rank by combining beam score (normalized) and reranker score
    beam_scores = [score for _, score in candidates]
    max_beam = max(beam_scores) if beam_scores else 1
    min_beam = min(beam_scores) if beam_scores else 0
    beam_range = max_beam - min_beam if max_beam != min_beam else 1

    reranked = []
    for i, (token_ids, beam_score) in enumerate(candidates):
        normalized_beam = (beam_score - min_beam) / beam_range if beam_range > 0 else 0
        combined_score = 0.5 * normalized_beam + 0.5 * rerank_scores[i]
        text = decode_token_ids(token_ids, bundle["tgt_itos"], "arabic" if task_name == "arabizi_to_arabic" else "arabizi")
        reranked.append((text, combined_score))

    # Sort by combined score
    reranked.sort(key=lambda x: x[1], reverse=True)

    # Return unique candidates
    outputs = []
    seen = set()
    for text, _ in reranked:
        if text and text not in seen:
            outputs.append(text)
            seen.add(text)

    return outputs


def predict_candidates(text: str, direction: str, top_k: int = 10, use_reranker: bool = True) -> List[str]:
    task_name, source_script, target_script = scripts_for_direction(direction)
    bundle = BUNDLES[task_name]
    model = MODELS[task_name]
    cfg = get_config(bundle)

    top_k = max(1, min(int(top_k), 25))
    decoder = cfg.get("DEFAULT_DECODER", "beam")

    if decoder == "beam":
        search_k = top_k * 2 if use_reranker and task_name in RERANKERS else top_k
        beam_size = min(
            int(cfg.get("BEAM_SIZE", 20)),
            int(cfg.get("MAX_WEB_BEAM_SIZE", 10)),
        )
        candidates_with_scores = beam_search_decode_nbest_with_scores(
            model=model,
            text=text,
            bundle=bundle,
            source_script=source_script,
            target_script=target_script,
            beam_size=max(1, beam_size),
            top_k=search_k,
        )

        reranker = RERANKERS.get(task_name)
        if use_reranker and reranker and candidates_with_scores:
            return rerank_candidates(text, candidates_with_scores, task_name, bundle, reranker)[:top_k]

        outputs = []
        seen = set()
        for token_ids, _ in candidates_with_scores:
            text_out = decode_token_ids(token_ids, bundle["tgt_itos"], target_script).strip()
            if text_out and text_out not in seen:
                outputs.append(text_out)
                seen.add(text_out)
            if len(outputs) >= top_k:
                break

        return outputs

    return [greedy_decode(model, text, bundle, source_script, target_script)]


def beam_search_decode_nbest_with_scores(
    model: Seq2SeqTransformer,
    text: str,
    bundle: Dict[str, Any],
    source_script: str,
    target_script: str,
    beam_size: int = 20,
    length_penalty: float = 0.80,
    top_k: int = 10,
) -> List[tuple]:
    """Beam search that returns candidates with their scores for reranking."""
    src_ids = prepare_source_ids(text, bundle, source_script)
    max_tgt_len = int(bundle["max_tgt_len"])
    bos_id = bundle["tgt_stoi"][BOS]
    eos_id = bundle["tgt_stoi"][EOS]

    beams = [([bos_id], 0.0)]
    finished = []

    for _ in range(max_tgt_len - 1):
        candidates = []

        for token_ids, score in beams:
            if token_ids[-1] == eos_id:
                finished.append((token_ids, score))
                continue

            tgt_input = torch.tensor([token_ids], dtype=torch.long, device=DEVICE)
            logits = model(src_ids, tgt_input)
            log_probs = F.log_softmax(logits[0, -1], dim=-1)

            values, indices = torch.topk(log_probs, k=min(beam_size, log_probs.size(0)))
            for value, idx in zip(values.tolist(), indices.tolist()):
                candidates.append((token_ids + [int(idx)], score + float(value)))

        if not candidates:
            break

        def normalized_score(item):
            token_ids, raw_score = item
            length = max(len(token_ids) - 1, 1)
            return raw_score / (length ** length_penalty)

        candidates.sort(key=normalized_score, reverse=True)
        beams = candidates[:beam_size]

        if len(finished) >= top_k and all(b[0][-1] == eos_id for b in beams):
            break

    finished.extend(beams)

    # Sort by normalized score
    finished.sort(key=lambda x: normalized_score(x), reverse=True)

    # Remove duplicates but keep scores
    outputs = []
    seen = set()
    for token_ids, score in finished:
        text_out = decode_token_ids(token_ids, bundle["tgt_itos"], target_script).strip()
        if text_out and text_out not in seen:
            outputs.append((token_ids, score))
            seen.add(text_out)
        if len(outputs) >= top_k:
            break

    return outputs


#################################
# routes
#################################

@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Arabizi transliteration backend is running.",
        "device": DEVICE,
    }


@app.post("/transliterate", response_model=TransliterationResponse)
def transliterate(request: TransliterationRequest):
    text = request.text.strip()
    if not text:
        return TransliterationResponse(candidates=[], status="Please enter some text first.")

    try:
        candidates = predict_candidates(
            text=text,
            direction=request.direction,
            top_k=request.top_k,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return TransliterationResponse(candidates=candidates, status="Done.")


#################################
# optional direct run
#################################

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
