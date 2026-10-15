"""Pinned public cross-encoder assets shared by native and browser loaders."""

SEMANTIC_REPOSITORY = "Xenova/ms-marco-MiniLM-L-6-v2"
SEMANTIC_REVISION = "a09144355adeed5f58c8ed011d209bf8ee5a1fec"
SEMANTIC_FILES = {
    "onnx/model.onnx": "c623d0bcb99f4622beb413eaef00cfbe5db20df9f1dd982da4b4f26022881870",
    "tokenizer.json": "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
    "config.json": "d827779a72d27ae68cf878a6fc2e954542663fe21ca515d9f4783fc96be2d37e",
    "tokenizer_config.json": "0b29c7bfc889e53b36d9dd3e686dd4300f6525110eaa98c76a5dafceb2029f53",
    "special_tokens_map.json": "b6d346be366a7d1d48332dbc9fdf3bf8960b5d879522b7799ddba59e76237ee3",
}
MODEL_NOTICE = f"""DailyDigest local semantic matching model

Upstream model: cross-encoder/ms-marco-MiniLM-L-6-v2
Model card: https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2
The upstream model card identifies the license as Apache-2.0.
License text: https://www.apache.org/licenses/LICENSE-2.0

ONNX conversion published by Xenova: {SEMANTIC_REPOSITORY}
Pinned conversion revision: {SEMANTIC_REVISION}
Conversion model card: https://huggingface.co/{SEMANTIC_REPOSITORY}/tree/{SEMANTIC_REVISION}

Scores are raw relevance logits, not probabilities or evidence of eligibility.
"""
