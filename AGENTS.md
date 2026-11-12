# AGENTS.md

Repository rules for agents working on `vllm-riva-frontend`, the Riva and Speech NIM compatibility
frontend for vLLM.

## Sources

- Cite only public sources: this repository, vLLM and its fork branches named in the docs, the
  public Riva protocol and client repositories (`nvidia-riva/common`, `nvidia-riva/python-clients`),
  public NIM documentation, Hugging Face model cards, and our own measurements.
- Never add code, names or descriptions taken from non-public sources.

## Contributions

- Commits are signed and carry a DCO `Signed-off-by` line.
- Keep existing comments when editing code; amend them only as the change requires.

## LID

- Mode: Full
- Version: 1.3.0

## LID Tooling

- The high-level design is `docs/high-level-design.md`; leaf designs and specs are under
  `docs/intent/`. Spec IDs use the `ING` prefix with a facet segment (`ING-GRPC-…`); each leaf's
  front matter lists the facets it owns.
- Code and tests cite specs with `@spec` comments. Every cited ID resolves in this repository or
  in the model plugin repository (`pdtgct/vllm-nemotron-asr-streaming-plugin`, prefixes `REG`,
  `SESS`, `FILE`).
