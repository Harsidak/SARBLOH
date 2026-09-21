# sarbloh/

The package. One responsibility per module; the contracts are in `CLAUDE.md` section 3 and are binding.

Dependency direction is one-way:

```
harness -> agent -> jugat -> parkh -> worldmodel -> memory -> surat
```

A module may import from modules to its right. Never to its left. `legacy/` is imported by nothing.

Rule that cuts across all of them: **certification precedes commitment.** `jugat` may not plan against a model
`parkh` has not certified, and `memory` may not promote a fact `parkh` has not confirmed.
