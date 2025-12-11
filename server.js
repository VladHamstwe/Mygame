import express from "express";
import fs from "fs";
import cors from "cors";

const app = express();
app.use(express.json());
app.use(cors());

// ===== LOAD EMBEDDINGS =====
let words = [];
let vectors = [];

console.log("Loading embeddings...");
const lines = fs.readFileSync("embeddings.jsonl", "utf8").trim().split("\n");

for (let line of lines) {
    const obj = JSON.parse(line);
    words.push(obj.word);
    vectors.push(obj.embedding);
}
console.log("Loaded", words.length, "embeddings.");

// ===== COSINE SIMILARITY =====
function cosine(a, b) {
    let dot = 0, na = 0, nb = 0;
    for (let i = 0; i < a.length; i++) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

// ===== API: GET WORD RANK =====
app.post("/guess", (req, res) => {
    const guess = req.body.word?.toLowerCase();

    if (!guess) {
        return res.json({ error: "No word" });
    }

    const index = words.indexOf(guess);

    if (index === -1) {
        return res.json({ error: "Word not found in dictionary" });
    }

    // Calculate similarity to the secret word
    const secretIndex = 0; // secret = words[0]
    const secretVec = vectors[secretIndex];

    // Compare the guessed word
    const similarity = cosine(vectors[index], secretVec);

    // Rank = how many words have higher similarity
    let rank = 1;
    for (let i = 0; i < vectors.length; i++) {
        if (i !== index) {
            const s = cosine(vectors[i], secretVec);
            if (s > similarity) rank++;
        }
    }

    res.json({ word: guess, rank: rank });
});

// ===== START SERVER =====
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log("Contexto server running on port", PORT));
