#!/usr/bin/env node
// cli.js - wrapper for the BPE tokenizer
const fs = require('fs');
const path = require('path');
const BPE_Tokenizer = require('./tokenizer-bpe');

function help() {
  console.log(`
BPE Tokenizer CLI

Commands:
  train <corpus.txt> <out_vocab.json> [--minfreq N] [--ops N]    Train BPE and save vocab+merges
  encode <vocab.json> "<text>"                                  Encode text -> ids (prints space-separated ids)
  decode <vocab.json> <ids...>                                  Decode ids -> text
  tokenize <vocab.json> "<text>"                                Show token pieces for text
  info <vocab.json>                                              Show vocab/merges info
`);
}

function parseArgs(argv) {
  const res = {cmd: null, args: [], flags: {}};
  if (argv.length === 0) return res;
  res.cmd = argv[0];
  for (let i = 1; i < argv.length; i++) {
    const a = argv[i];
    if (a.startsWith('--')) {
      const [k,v] = a.slice(2).split('=');
      res.flags[k] = v === undefined ? true : v;
    } else {
      res.args.push(a);
    }
  }
  return res;
}

async function main() {
  const raw = process.argv.slice(2);
  const parsed = parseArgs(raw);
  const cmd = parsed.cmd;

  if (!cmd) return help();

  if (cmd === 'train') {
    const corpus = parsed.args[0];
    const out = parsed.args[1];
    if (!corpus || !out) return help();
    const minfreq = Number(parsed.flags.minfreq || 2);
    const ops = Number(parsed.flags.ops || 10000);
    const text = fs.readFileSync(corpus, 'utf8');
    const t = new BPE_Tokenizer();
    t.trainFromText(text, {minFrequency: minfreq, mergeOps: ops, verbose: true});
    t.save(out);
    console.log(`Saved BPE vocab+merges to ${out}`);
  }

  else if (cmd === 'encode') {
    const vocab = parsed.args[0];
    const text = parsed.args.slice(1).join(' ');
    if (!vocab || !text) return help();
    const t = new BPE_Tokenizer();
    t.load(vocab);
    const ids = t.encode(text);
    console.log(ids.join(' '));
  }

  else if (cmd === 'decode') {
    const vocab = parsed.args[0];
    const ids = parsed.args.slice(1).map(x => parseInt(x,10)).filter(x=>!Number.isNaN(x));
    if (!vocab || ids.length===0) return help();
    const t = new BPE_Tokenizer();
    t.load(vocab);
    console.log(t.decode(ids));
  }

  else if (cmd === 'tokenize') {
    const vocab = parsed.args[0];
    const text = parsed.args.slice(1).join(' ');
    if (!vocab || !text) return help();
    const t = new BPE_Tokenizer();
    t.load(vocab);
    console.log(t.tokenize(text).join(' | '));
  }

  else if (cmd === 'info') {
    const vocab = parsed.args[0];
    if (!vocab) return help();
    const t = new BPE_Tokenizer();
    t.load(vocab);
    console.log('Vocab size (tokens):', [...t.tokenToId.keys()].length);
    console.log('Merges count:', t.merges.length);
    console.log('First 30 merges:', t.merges.slice(0,30).map(m=>m.join('+')).join(', '));
  }

  else help();
}

main();
