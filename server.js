// === AI SERVER FOR ROBLOX CONTEXTO ===
// by ChatGPT — stable version

import express from "express";
import fs from "fs";

const app = express();
app.use(express.json());

// ================= LOAD EMBEDDINGS ====================
let words = [];
let vectors = [];
console.log("Loading embeddings...");

const lines = fs.readFileSync("embeddings.jsonl", "utf8").trim().split("\n");
for (let line of lines) {
    const o = JSON.parse(line);
    words.push(o.word);
    vectors.push(o.embedding);
}

console.log("Loaded", words.length, "embeddings");

// =============== COSINE SIMILARITY =====================
function cosine(a, b) {
    let dot = 0, na = 0, nb = 0;
    for (let i = 0; i < a.length; i++) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

// =================== API ===============================
app.get("/", (req, res) => {
    res.send(`
        <h1>AI SERVER WORKING ✅</h1>
        <p>Your Roblox game can now send requests to /guess</p>
    `);
});

app.post("/guess", (req, res) => {
    const guess = req.body.word?.toLowerCase();

    if (!guess) return res.json({ error: "no_word" });

    const index = words.indexOf(guess);
    if (index === -1) return res.json({ error: "word_not_in_dictionary" });

    // SECRET WORD = FIRST WORD IN FILE
    const secretIndex = 0;
    const secretVec = vectors[secretIndex];

    const simGuess = cosine(vectors[index], secretVec);

    // Calculate rank = how many have higher similarity
    let rank = 1;
    for (let i = 0; i < words.length; i++) {
        if (i !== index) {
            if (cosine(vectors[i], secretVec) > simGuess) {
                rank++;
            }
        }
    }

    res.json({
        word: guess,
        rank: rank
    });
});

// ================ RUN SERVER ===========================
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log("AI Server running on port", PORT));

