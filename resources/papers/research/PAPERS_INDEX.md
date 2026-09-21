# Research Paper Quick-Reference Index
# ARC-AGI-3 Competition — Papers to Read

## MUST-READ (Directly impacts our agent design)

### 1. Executable World Models for ARC-AGI-3
- **Title**: "Executable World Models for ARC-AGI-3 in the Era of Coding Agents"
- **Authors**: Sergey Rodionov
- **arXiv**: https://arxiv.org/abs/2605.05138
- **PDF**: https://arxiv.org/pdf/2605.05138
- **HTML**: https://arxiv.org/html/2605.05138v2
- **Code**: https://github.com/astroseger/arc-3-agents-baseline1
- **Why**: SOTA on public games (58% RHAE). Defines the observe-model-verify-refactor-plan-execute loop.
  The architecture we should converge toward.

### 2. ARC-AGI-3 Benchmark Specification
- **Title**: "ARC-AGI-3: A New Challenge for Frontier Agentic Intelligence"
- **Authors**: ARC Prize Foundation
- **arXiv**: https://arxiv.org/abs/2603.24621
- **PDF**: https://arxiv.org/pdf/2603.24621
- **Why**: Official rules. RHAE formula. Core Knowledge priors. Human baseline methodology.

### 3. Code World Models for General Game Playing
- **Title**: "Code World Models for General Game Playing"
- **Authors**: Google DeepMind (Dainese, Merler, Alakuijala, Marttinen)
- **arXiv**: https://arxiv.org/abs/2510.04542
- **PDF**: https://arxiv.org/pdf/2510.04542
- **Why**: The theoretical foundation for code-as-world-model + MCTS planning.
  Outperformed Gemini 2.5 Pro in 9/10 games.

### 4. Graph-Based Exploration for ARC-AGI-3
- **Title**: "Graph-Based Exploration for ARC-AGI-3 Interactive Reasoning Tasks"
- **Authors**: Evgenii Rudakov, Jonathan Shock, Benjamin Ultan Cowley
- **Published**: December 2025
- **Search on arXiv**: "Rudakov graph-based exploration ARC-AGI-3"
- **Why**: Training-free! No neural net. Ranked 3rd in preview.
  Directly validates our ExecPlanner T2 approach.

## SHOULD-READ (Techniques we can borrow)

### 5. WorldCoder
- **Title**: "WorldCoder: Model-Based LLM Agent Framework"
- **Venue**: NeurIPS 2024
- **Search on arXiv**: "WorldCoder model-based LLM agent Python program"
- **Why**: Optimistic planning with code world models. 
  Sample-efficient. Transfer across environments.

### 6. Tiny Recursive Models (TRM)
- **Title**: "Less is More: Recursive Reasoning with Tiny Networks"
- **Authors**: Alexia Jolicoeur-Martineau (Samsung SAIL Montreal)
- **arXiv**: https://arxiv.org/abs/2510.04871
- **PDF**: https://arxiv.org/pdf/2510.04871
- **Code**: https://github.com/SamsungSAILMontreal/TinyRecursiveModels
- **Why**: 7M params, 45% on ARC-AGI-1. Shows tiny nets + recursion work.
  The data augmentation strategy is gold.

### 7. Active Inference as Test-Time Scaling Law
- **Title**: "Active Inference as the Test-Time Scaling Law for Physical AI Agents"
- **Authors**: Hashash, Thomas, Saad, Debbah, Friston, Razi
- **arXiv**: https://arxiv.org/abs/2606.22813
- **PDF**: https://arxiv.org/pdf/2606.22813
- **Why**: Dynamic policy updating at test time. 36% improvement in inference efficiency.
  Relevant for our online adaptation strategy.

### 8. Generating Code World Models with MCTS
- **Title**: "Generating Code World Models with Large Language Models Guided by Monte Carlo Tree Search"
- **arXiv**: https://arxiv.org/abs/2405.15383
- **PDF**: https://arxiv.org/pdf/2405.15383
- **Why**: MCTS-guided code generation for world models. 
  Directly relevant to planning through executable models.

## BACKGROUND (Contextual understanding)

### 9. Grid-JEPA
- **Source**: HuggingFace community
- **Search**: "Grid-JEPA JEPA world model grid environment"
- **Why**: Latent-space prediction for grid worlds. RSSM + JEPA.
  Interesting for frame prediction without pixel reconstruction.

### 10. ARC Prize 2025 Technical Report
- **Source**: arcprize.org
- **Search**: "ARC Prize 2025 technical report winners"
- **Why**: Context on ARC-AGI-2 solutions (evolutionary program synthesis,
  test-time training, NVARC ensemble approach).

## GitHub Repositories to Clone/Study

1. https://github.com/astroseger/arc-3-agents-baseline1 (EWM agent)
2. https://github.com/arcprize/ARC-AGI-3-Kaggle-Starter (official starter)
3. https://github.com/Tufalabs/duck-harness (The Duck harness)
4. https://github.com/tufa-labs/arc-agi-3 (StochasticGoose + The Duck)
5. https://github.com/SamsungSAILMontreal/TinyRecursiveModels (TRM)
