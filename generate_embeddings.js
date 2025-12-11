// generate_embeddings.js
// Usage:
//   (mac/linux)  export OPENAI_KEY="sk-..."
//   (windows cmd) set OPENAI_KEY=sk-...
//   node generate_embeddings.js
//
// Output: embeddings.jsonl (one JSON per line: { "word": "...", "vector": [...] })

import fs from "fs";
import readline from "readline";
import OpenAI from "openai";
import pLimit from "p-limit";

const MODEL = process.env.EMB_MODEL || "text-embedding-3-large"; // or "text-embedding-3-small" if you prefer cheaper
const BATCH = Number(process.env.BATCH) || 32;
const CONCURRENCY = Number(process.env.CONCURRENCY) || 2;

if (!process.env.OPENAI_KEY) {
  console.error("ERROR: set OPENAI_KEY environment variable before running.");
  process.exit(1);
}

const client = new OpenAI({ apiKey: process.env.OPENAI_KEY });

async function generate(inPath = "words.txt", outPath = "embeddings.jsonl") {
  const inStream = fs.createReadStream(inPath);
  const rl = readline.createInterface({ input: inStream, crlfDelay: Infinity });

  // load already saved words to allow resume
  const done = new Set();
  if (fs.existsSync(outPath)) {
    const lines = fs.readFileSync(outPath, "utf8").split(/\r?\n/).filter(Boolean);
    for (const line of lines) {
      try { done.add(JSON.parse(line).word); } catch {}
    }
    console.log("Resuming, already have:", done.size);
  }

  const outStream = fs.createWriteStream(outPath, { flags: "a" });
  const limit = pLimit(CONCURRENCY);
  let batch = [];
  let total = done.size;

  function flushBatchNow(batchWords) {
    return limit(async () => {
      try {
        const resp = await client.embeddings.create({ model: MODEL, input: batchWords });
        for (let i = 0; i < batchWords.length; i++) {
          const word = batchWords[i];
          const vector = resp.data[i].embedding;
          // normalize vector to unit length to make later dot product = cosine
          let n = 0;
          for (let k=0;k<vector.length;k++) n += vector[k]*vector[k];
          n = Math.sqrt(n) || 1;
          for (let k=0;k<vector.length;k++) vector[k] = vector[k]/n;
          outStream.write(JSON.stringify({ word, vector }) + "\n");
          total++;
        }
        console.log(`Saved total: ${total}`);
      } catch (e) {
        console.error("Batch error:", e?.message || e);
        // rethrow or wait? we wait briefly
        await new Promise(r => setTimeout(r, 2000));
        throw e;
      }
    });
  }

  for await (const raw of rl) {
    const w = raw.trim();
    if (!w) continue;
    if (done.has(w)) continue;
    batch.push(w);
    if (batch.length >= BATCH) {
      // send and reset
      try {
        await flushBatchNow(batch);
      } catch (e) {
        console.error("Retry after error...");
        await new Promise(r => setTimeout(r, 5000));
        // try one more time
        await flushBatchNow(batch);
      }
      batch = [];
    }
  }

  if (batch.length > 0) {
    try {
      await flushBatchNow(batch);
    } catch (e) {
      console.error("Final batch error:", e?.message || e);
    }
  }

  outStream.end();
  console.log("All done. Embeddings saved to", outPath);
}

generate();
