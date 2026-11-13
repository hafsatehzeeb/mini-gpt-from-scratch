import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F


# Device
device = "cuda" if torch.cuda.is_available() else "cpu"


# Load saved model data
checkpoint = torch.load(
    "mini_gpt.pth",
    map_location=device,
    weights_only=False
)

vocab_size = checkpoint["vocab_size"]
n_embd = checkpoint["n_embd"]
block_size = checkpoint["block_size"]

stoi = checkpoint["stoi"]
itos = checkpoint["itos"]


# Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_heads, block_size):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=n_embd,
            num_heads=n_heads,
            batch_first=True
        )

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

        self.feed_forward = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd)
        )

        mask = torch.triu(
            torch.ones(block_size, block_size, dtype=torch.bool),
            diagonal=1
        )

        self.register_buffer("mask", mask)

    def forward(self, x):
        T = x.size(1)

        attn_out, _ = self.attention(
            x, x, x,
            attn_mask=self.mask[:T, :T],
            need_weights=False
        )

        x = x + self.ln1(attn_out)
        x = x + self.ln2(self.feed_forward(x))

        return x


# Mini GPT Model
class MiniGPT(nn.Module):
    def __init__(self):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)

        self.transformer = TransformerBlock(
            n_embd, 4, block_size
        )

        self.final_norm = nn.LayerNorm(n_embd)
        self.output_layer = nn.Linear(n_embd, vocab_size)

    def forward(self, idx):
        B, T = idx.shape

        tok_emb = self.token_embedding(idx)

        pos = torch.arange(T, device=idx.device)
        pos_emb = self.position_embedding(pos)

        x = tok_emb + pos_emb

        x = self.transformer(x)
        x = self.final_norm(x)

        logits = self.output_layer(x)

        return logits


# Initialize and load trained weights
model = MiniGPT().to(device)

model.load_state_dict(checkpoint["model_state_dict"])

model.eval()


# Text generation
@torch.no_grad()
def generate_text(prompt, max_tokens):

    encoded = [stoi[ch] for ch in prompt if ch in stoi]

    if not encoded:
        return "Please enter valid text."

    context = torch.tensor(
        [encoded],
        dtype=torch.long,
        device=device
    )

    for _ in range(int(max_tokens)):

        idx_cond = context[:, -block_size:]

        logits = model(idx_cond)

        logits = logits[:, -1, :]

        probs = F.softmax(logits, dim=-1)

        next_token = torch.multinomial(probs, 1)

        context = torch.cat((context, next_token), dim=1)

    return "".join(
        itos[int(i)] for i in context[0]
    )


# Website interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:

    gr.Markdown("# Mini GPT – Shakespeare AI")

    gr.Markdown(
        "A character-level GPT Transformer built from scratch using PyTorch."
    )

    prompt = gr.Textbox(
        label="Enter your prompt",
        placeholder="Example: ROMEO:",
        lines=3
    )

    length = gr.Slider(
        100, 500, value=300, step=100,
        label="Generation length"
    )

    button = gr.Button("Generate Text")

    output = gr.Textbox(
        label="AI Generated Text",
        lines=15
    )

    button.click(
        generate_text,
        inputs=[prompt, length],
        outputs=output
    )


demo.launch()
