// tokenizer-bpe.js
// BPE-enhanced tokenizer for CLI
// Simple, educational, reasonably efficient BPE implementation in JS

const fs = require('fs');

class BPE_Tokenizer {
  constructor(opts = {}) {
    const {
      lower = true,
      specialTokens = ['<PAD>', '<UNK>', '<BOS>', '<EOS>', '<SEP>'],
      merges = [],
      vocab = {},
    } = opts;

    this.lower = lower;
    this.specialTokens = specialTokens.slice();

    // vocabulary maps token -> id
    this.tokenToId = new Map();
    this.idToToken = new Map();

    // merges is list of pairs merged in order: [['a','b'], ['c','d'], ...]
    this.merges = merges.slice();

    // base counts (only used during training)
    this.counts = new Map();

    // initialize special tokens
    this._initSpecials();

    // if initial vocab provided, load it
    Object.entries(vocab || {}).forEach(([tok, id]) => {
      const n = Number(id);
      if (!Number.isNaN(n)) {
        this.tokenToId.set(tok, n);
        this.idToToken.set(n, tok);
      }
    });
  }

  _initSpecials() {
    this.tokenToId.clear();
    this.idToToken.clear();
    let idx = 0;
    for (const st of this.specialTokens) {
      this.tokenToId.set(st, idx);
      this.idToToken.set(idx, st);
      idx++;
    }
    this.nextId = idx;
  }

  // Helper: split word into characters with a special word-start marker '▁'
  // e.g. 'hello' -> ['▁h','e','l','l','o']
  _wordToChars(word) {
    const chars = Array.from(word);
    if (chars.length === 0) return [];
    const out = [];
    out.push('▁' + chars[0]);
    for (let i = 1; i < chars.length; i++) out.push(chars[i]);
    return out;
  }

  // Join token pieces to a printable word: reverse of _wordToChars
  _piecesToWord(pieces) {
    let s = '';
    for (const p of pieces) {
      if (p.startsWith('▁')) s += (s.length === 0 ? '' : '') + p.slice(1);
      else s += p;
    }
    return s;
  }

  // Train BPE merges from a corpus text
  trainFromText(text, {minFrequency = 2, mergeOps = 10000, verbose = false} = {}) {
    this.counts.clear();

    const lines = String(text).split(/\r?\n/);
    for (const l of lines) {
      const line = this.lower ? l.normalize('NFC').toLowerCase() : l;
      const words = line.split(/\s+/).filter(Boolean);
      for (const w of words) {
        const pieces = this._wordToChars(w);
        const key = pieces.join(' ');
        this.counts.set(key, (this.counts.get(key) || 0) + 1);
      }
    }

    // compute merges
    this.merges = [];

    // build initial vocab counts of pairs
    for (let iter = 0; iter < mergeOps; iter++) {
      const pairCounts = new Map();

      for (const [wordPiecesStr, cnt] of this.counts.entries()) {
        if (cnt < minFrequency) continue; // skip rare
        const pieces = wordPiecesStr.split(' ');
        for (let i = 0; i < pieces.length - 1; i++) {
          const pair = pieces[i] + ' ' + pieces[i + 1];
          pairCounts.set(pair, (pairCounts.get(pair) || 0) + cnt);
        }
      }

      if (pairCounts.size === 0) break; // nothing to merge

      // find best pair (highest count)
      let bestPair = null;
      let bestCount = 0;
      for (const [p, c] of pairCounts.entries()) {
        if (c > bestCount) {
          bestCount = c;
          bestPair = p;
        }
      }

      if (!bestPair || bestCount < minFrequency) break;

      const [a, b] = bestPair.split(' ');
      const merged = a + b; // new token
      this.merges.push([a, b]);

      // apply merge to counts: replace occurrences of [a b] with [ab]
      const newCounts = new Map();
      for (const [wordPiecesStr, cnt] of this.counts.entries()) {
        const pieces = wordPiecesStr.split(' ');
        const out = [];
        let i = 0;
        while (i < pieces.length) {
          if (i < pieces.length - 1 && pieces[i] === a && pieces[i + 1] === b) {
            out.push(merged);
            i += 2;
          } else {
            out.push(pieces[i]);
            i += 1;
          }
        }
        const nk = out.join(' ');
        newCounts.set(nk, (newCounts.get(nk) || 0) + cnt);
      }

      this.counts = newCounts;
      if (verbose && iter % 100 === 0) {
        process.stderr.write(`iter=${iter} best=${a}+${b} count=${bestCount}\n`);
      }
    }

    // after merges, collect final vocabulary tokens
    // tokens are all unique pieces across counts + special tokens
    const tokenSet = new Set(this.specialTokens);
    for (const k of this.counts.keys()) {
      for (const p of k.split(' ')) tokenSet.add(p);
    }

    // assign ids starting after specials
    this._initSpecials();
    for (const tok of tokenSet) {
      if (this.tokenToId.has(tok)) continue;
      this.tokenToId.set(tok, this.nextId);
      this.idToToken.set(this.nextId, tok);
      this.nextId++;
    }

    // ensure deterministic order for merges (already stored in this.merges)
  }

  // Save vocab + merges + meta
  save(path) {
    const obj = {
      meta: {lower: this.lower, specialTokens: this.specialTokens},
      vocab: Object.fromEntries(this.tokenToId),
      merges: this.merges,
    };
    fs.writeFileSync(path, JSON.stringify(obj, null, 2), 'utf8');
  }

  load(path) {
    const raw = fs.readFileSync(path, 'utf8');
    const obj = JSON.parse(raw);
    this.lower = obj.meta.lower;
    this.specialTokens = obj.meta.specialTokens;
    this._initSpecials();
    this.tokenToId = new Map(Object.entries(obj.vocab));
    // convert ids to numbers in idToToken
    this.idToToken = new Map([...this.tokenToId.entries()].map(([t, id]) => [Number(id), t]));
    this.nextId = Math.max(...[...this.idToToken.keys()]) + 1;
    this.merges = obj.merges || [];
  }

  // Apply merges to a word (greedy merges following learned merge order)
  _applyBPEToWord(word) {
    // initial pieces
    let pieces = this._wordToChars(word);
    if (pieces.length === 0) return [];

    // For fast pair lookup, create a set of allowed merges (strings 'a b') with priority order
    const mergeMap = new Map();
    for (let i = 0; i < this.merges.length; i++) {
      const [a, b] = this.merges[i];
      mergeMap.set(a + ' ' + b, i);
    }

    // We'll loop greedily: at each step find the leftmost highest-priority merge
    let done = false;
    while (!done) {
      done = true;
      let bestPos = -1;
      let bestRank = Infinity; // lower is better

      for (let i = 0; i < pieces.length - 1; i++) {
        const key = pieces[i] + ' ' + pieces[i + 1];
        if (mergeMap.has(key)) {
          const rank = mergeMap.get(key);
          if (rank < bestRank) {
            bestRank = rank;
            bestPos = i;
            done = false;
          }
        }
      }

      if (!done && bestPos >= 0) {
        const merged = pieces[bestPos] + pieces[bestPos + 1];
        pieces.splice(bestPos, 2, merged);
      }
    }

    return pieces;
  }

  // Tokenize text into BPE pieces (list of tokens for entire text)
  tokenize(text) {
    const src = this.lower ? text.normalize('NFC').toLowerCase() : text;
    const words = src.split(/(\s+)/); // keep whitespace tokens to preserve spacing
    const out = [];
    for (const w of words) {
      if (/^\s+$/.test(w)) {
        out.push(w); // whitespace preserved
      } else if (w.length === 0) {
        // skip
      } else {
        const pieces = this._applyBPEToWord(w);
        for (const p of pieces) out.push(p);
      }
    }
    return out;
  }

  // Encode text to ids (preserve whitespace as separate tokens or join with special marker)
  encode(text, {addBosEos = true, keepSpaces = true} = {}) {
    const pieces = this.tokenize(text);
    const ids = [];
    if (addBosEos) ids.push(this.tokenToId.get('<BOS>'));

    for (const p of pieces) {
      if (/^\s+$/.test(p)) {
        if (keepSpaces) {
          const key = '<SPACE>' + p.length; // variable space token; likely not in vocab
          ids.push(this.tokenToId.get(key) ?? this.tokenToId.get('<UNK>'));
        }
        continue;
      }
      const id = this.tokenToId.get(p);
      ids.push(id ?? this.tokenToId.get('<UNK>'));
    }

    if (addBosEos) ids.push(this.tokenToId.get('<EOS>'));
    return ids;
  }

  // Very simple decode: join tokens, replace word-start marker with nothing but add spaces
  decode(ids, {skipSpecials = true} = {}) {
    const parts = [];
    for (const idRaw of ids) {
      const id = Number(idRaw);
      const tok = this.idToToken.get(id) ?? '<UNK>';
      if (skipSpecials && this.specialTokens.includes(tok)) continue;
      parts.push(tok);
    }

    // Reconstruct text: when token starts with '▁' treat as word-start
    let out = '';
    for (const p of parts) {
      if (p.startsWith('▁')) {
        // new word
        out += (out.length === 0 ? '' : '') + p.slice(1);
      } else {
        out += p;
      }
    }
    return out;
  }
}

module.exports = BPE_Tokenizer;
