// index.js
import express from "express";
import fs from "fs";

const app = express();
app.use(express.json());

const EMB_FILE = process.env.EMB_FILE || "./embeddings.jsonl";
console.log("Loading embeddings from", EMB_FILE);

// load embeddings into arrays
const lines = fs.readFileSync(EMB_FILE, "utf8").split(/\r?\n/).filter(Boolean);
const words = [];
let vectors = null;
let dims = null;
for (let i=0;i<lines.length;i++){
  const o = JSON.parse(lines[i]);
  if (i===0) { dims = o.vector.length; vectors = new Float32Array(lines.length * dims); }
  words.push(o.word);
  for (let j=0;j<dims;j++) vectors[i*dims + j] = o.vector[j];
}
console.log(`Loaded ${words.length} words, dims=${dims}`);

function dot(i, guessVec) {
  let s = 0;
  const base = i*dims;
  for (let k=0;k<dims;k++) s += vectors[base + k] * guessVec[k];
  return s;
}

app.post("/guess", async (req, res) => {
  try {
    const guess = (req.body.word || "").toString().trim().toLowerCase();
    if (!guess) return res.status(400).json({ error: "no_word" });

    // if guess exists in dictionary, use its vector directly
    let guessVec = null;
    const idxInDict = words.indexOf(guess);
    if (idxInDict !== -1) {
      guessVec = new Array(dims);
      for (let d=0; d<dims; d++) guessVec[d] = vectors[idxInDict*dims + d];
    } else {
      // If guess not in dict, we could call OpenAI to get embedding.
      // For simplicity: return position by computing similarity to all words using OpenAI embedding.
      // But we will require OPENAI_KEY env var if needed. Here we respond with error to avoid surprise.
      return res.status(400).json({ error: "word_not_in_dictionary" });
    }

    // compute similarity to all words (dot product since vectors are normalized)
    const sims = new Float32Array(words.length);
    for (let i=0;i<words.length;i++) sims[i] = dot(i, guessVec);

    // create index array and sort top
    const idxs = Array.from({length: words.length}, (_,i) => i);
    idxs.sort((a,b) => sims[b] - sims[a]);

    // find rank of guess
    const rank = idxs.indexOf(idxInDict) + 1;

    // top N to return
    const topN = Number(req.query.top || 20);
    const top = idxs.slice(0, topN).map((i, r) => ({ rank: r+1, word: words[i], similarity: sims[i] }));

    res.json({ word: guess, position: rank, top });
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: e.message || e });
  }
});

app.get("/", (req, res) => res.send("Contexto AI server running"));
app.listen(process.env.PORT || 3000, () => console.log("Server started"));
