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
