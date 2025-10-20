# ============================================
# 🧠 Urdu Text-Infilling Chatbot (Streamlit)
# ============================================

import streamlit as st
import torch
import os
from model import (
    Transformer,
    TokenizerWrapper,
    normalize_urdu_text,
    corrupt_text_span,
    CONFIG,
    SENTENCEPIECE_AVAILABLE,
)

# ----------------------------------------------------
# 🌐 Page setup
# ----------------------------------------------------
st.set_page_config(page_title="Urdu Chatbot (Transformer)", page_icon="💬", layout="centered")

st.title("💬 Urdu Text Chatbot")
st.markdown("Chat naturally in Urdu!")

# ----------------------------------------------------
# ⚙️ Device & paths
# ----------------------------------------------------
MODEL_PATH = "best_urdu_transformer_final.pt"
CORPUS_PATH = "urdu_corpus.txt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------------------------
# 🧩 Tokenizer + Model Loaders
# ----------------------------------------------------
@st.cache_resource
def load_tokenizer(corpus_path, vocab_size, use_sentencepiece):
    return TokenizerWrapper(corpus_path, vocab_size=vocab_size, use_sentencepiece=use_sentencepiece)

@st.cache_resource
def build_model_and_load_weights(_tokenizer, model_path, device):
    model = Transformer(
        vocab_size=_tokenizer.vocab_size(),
        embed_dim=CONFIG["embed_dim"],
        num_enc_layers=CONFIG["num_encoder_layers"],
        num_dec_layers=CONFIG["num_decoder_layers"],
        num_heads=CONFIG["num_heads"],
        ff_dim=CONFIG["ff_dim"],
        dropout=CONFIG["dropout"],
        max_len=CONFIG["max_len"],
        pad_idx=_tokenizer.pad_id()
    ).to(device)

    if not os.path.exists(model_path):
        return model, "missing", None

    try:
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state, strict=False)
        return model, "loaded", None
    except Exception as e:
        return model, "failed", str(e)

# ----------------------------------------------------
# Load components with fixed defaults (no sidebar)
# ----------------------------------------------------
USE_SP = SENTENCEPIECE_AVAILABLE
TEMPERATURE = 0.9
MAX_LEN = 60
CORRUPT_FRACTION = float(CONFIG["corruption_fraction"])

tokenizer = load_tokenizer(CORPUS_PATH, CONFIG["vocab_size"], USE_SP)
model, load_status, info = build_model_and_load_weights(tokenizer, MODEL_PATH, DEVICE)
model.eval()

if load_status == "missing":
    st.error(f"❌ Model file not found at `{MODEL_PATH}`.")
elif load_status == "failed":
    st.error(f"❌ Failed to load model checkpoint.\n\n{info}")
else:
    st.success("✅ Model loaded successfully!")

# ----------------------------------------------------
# 💬 Chat state management
# ----------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def add_message(role, text):
    st.session_state.chat_history.append({"role": role, "text": text})

# ----------------------------------------------------
# 🧠 Chat interaction
# ----------------------------------------------------
st.divider()
st.subheader("🗨️ Chat in Urdu")

user_input = st.text_input("آپ کا پیغام لکھیں (Enter your Urdu message):", "")

col1, col2 = st.columns([1, 0.4])
with col1:
    send_btn = st.button("📤 Send")
with col2:
    clear_btn = st.button("🧹 Clear Chat")

if clear_btn:
    st.session_state.chat_history = []
    st.experimental_rerun()

if send_btn and user_input.strip():
    # Normalize and corrupt text
    norm = normalize_urdu_text(user_input)
    corrupted, target = corrupt_text_span(norm, corruption_fraction=CORRUPT_FRACTION)

    src_ids = [tokenizer.bos_id()] + tokenizer.encode(corrupted) + [tokenizer.eos_id()]
    src_tensor = torch.LongTensor(src_ids).unsqueeze(0).to(DEVICE)

    with st.spinner("🤖 Generating response..."):
        try:
            response = model.generate(src_tensor, tokenizer, DEVICE, max_len=MAX_LEN, temperature=TEMPERATURE)
        except TypeError:
            response = model.generate(src_tensor, tokenizer, DEVICE, max_len=MAX_LEN)

    # Update chat
    add_message("user", user_input)
    add_message("bot", response)

# ----------------------------------------------------
# 🪶 Chat display (RTL Urdu style, black text)
# ----------------------------------------------------
st.markdown("---")
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(
            f"<div style='direction:rtl; background:#e8f0fe; padding:10px; border-radius:12px; margin:4px 0; text-align:right; font-size:18px; color:black;'>🧍‍♂️ {msg['text']}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div style='direction:rtl; background:#d1ffd6; padding:10px; border-radius:12px; margin:4px 0; text-align:right; font-size:18px; color:black;'>🤖 {msg['text']}</div>",
            unsafe_allow_html=True,
        )

# ----------------------------------------------------
# 📊 Sidebar Info
# ----------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.write(f"**Tokenizer type:** {'SentencePiece' if tokenizer.use_sentencepiece else 'Simple'}")
st.sidebar.write(f"**Vocab size:** {tokenizer.vocab_size()}")
st.sidebar.write(f"**Device:** `{DEVICE}`")
st.sidebar.write(f"**Parameters:** {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
st.sidebar.info("Developed with ❤️ using PyTorch + Streamlit")
