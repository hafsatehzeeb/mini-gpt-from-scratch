let session;
let vocab;

async function loadModel() {
    try {
        session = await ort.InferenceSession.create("mini_gpt.onnx");

        const response = await fetch("vocab.json");
        vocab = await response.json();

        console.log("Mini GPT loaded successfully!");
    } catch (error) {
        console.error("Model loading failed:", error);
    }
}

loadModel();
async function generateText() {

    const prompt = document.getElementById("prompt").value;

    if (!session || !vocab) {
        alert("AI model is still loading. Please wait!");
        return;
    }

    if (!prompt) {
        alert("Please enter a prompt!");
        return;
    }

    const stoi = vocab.stoi;
    const itos = vocab.itos;

    let tokens = [];

    for (let char of prompt) {
        if (stoi[char] !== undefined) {
            tokens.push(stoi[char]);
        }
    }

    if (tokens.length === 0) {
        alert("No valid characters found!");
        return;
    }

    const blockSize = vocab.block_size;

    for (let i = 0; i < 100; i++) {

        let context = tokens.slice(-blockSize);

        let input = new BigInt64Array(blockSize);

        for (let j = 0; j < context.length; j++) {
            input[j] = BigInt(context[j]);
        }

        const tensor = new ort.Tensor(
            "int64",
            input,
            [1, blockSize]
        );

        const result = await session.run({
            input_ids: tensor
        });

        const logits = result.logits.data;

        const position = context.length - 1;
        const start = position * vocab.vocab_size;

        let scores = Array.from(
            logits.slice(start, start + vocab.vocab_size)
        );

        let maxScore = Math.max(...scores);

        let probabilities = scores.map(
            x => Math.exp(x - maxScore)
        );

        let total = probabilities.reduce((a, b) => a + b, 0);

        probabilities = probabilities.map(x => x / total);

        let random = Math.random();
        let cumulative = 0;
        let nextToken = 0;

        for (let k = 0; k < probabilities.length; k++) {
            cumulative += probabilities[k];

            if (random <= cumulative) {
                nextToken = k;
                break;
            }
        }

        tokens.push(nextToken);
    }
    document.getElementById("generate").addEventListener(
    "click",
    generateText
);

    document.getElementById("output").value =
        tokens.map(token => itos[String(token)]).join("");
}
