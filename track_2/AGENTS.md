# GENOME codebase agent instructions

Read these before editing:

1. `docs/track_2_genome/AGENTS.md`
2. `docs/track_2_genome/00_README.md`
3. `README.md`
4. `IMPLEMENTATION_STATUS.md`
5. `POLYPYTHIA_ROUND1.md`

## Immediate mission

Run PolyPythia 14M Round One. Keep the deterministic MGP Runtime, learned Neural Genome Decoder, and learned GENOME Compiler separate.

## Sealed split

- training: seed0 through seed7;
- development: seed8;
- hidden: seed9.

Do not change this split after training begins.

## Hidden endpoint rule

Before the hidden genome is predicted and sealed:

- seed9 W0 is allowed;
- seed9 WT and every later seed9 checkpoint are forbidden;
- no early training prefix is allowed;
- no repair or polishing is allowed in the primary result.

## Rules

- Never let the compiler read WT values, endpoint hashes, fitted endpoint codes, or intermediate weights.
- Train the compiler through the frozen decoder against endpoint Delta-T and functional losses. Do not force it to reproduce an arbitrary latent code.
- Treat seeds as independent model lives. Do not treat checkpoints from one trajectory as independent lives.
- Keep every Hugging Face revision, LFS hash, dataset revision, and tokenizer identity immutable.
- Do not download intermediate checkpoints unless an implemented experiment consumes them.
- Never describe G0 fitted reconstruction as hidden endpoint prediction.
- Evaluate model behavior. Parameter error alone is not success.
- Count target genome bytes, shared decoder bytes, W0 bytes, fitting time, compiler time, decode time, and evaluation time separately.
- Keep primary one-shot results separate from any later repair or polishing result.
- Preserve immutable output directories. Do not overwrite a completed artifact.
- Do not train on the local Track 1 endpoint during PolyPythia Round One.
- Run the complete test suite after changes to MGP, decoder, compiler, source planning, or hidden evaluation code.
