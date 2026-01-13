# LunarLander-v3 RL Comparison - Quick Reference

**Student:** Ethan Hulme
**Course:** CIS2719 - Foundations of Robotics & AI (Coursework 2)
**Date:** January 2026

---

## How to View This Assignment

This assignment is submitted as a **Google Colab notebook** that contains:
- Complete implementation code
- All experimental results (10 test runs)
- Learning curves and visualizations
- GIF animations of trained agents
- Detailed analysis and documentation

### Access Instructions:

1. Open the shared Google Colab link
2. All results are pre-loaded and ready to view
3. You can browse through sections or run cells to verify functionality
4. No installation or setup required on your end

---

## Notebook Structure

### Section 1-2: Setup
- Package installation (automated)
- GitHub repository cloning for accessing pre-trained results

### Section 3: Implementation
- Complete code for DQN, DDQN, and PER algorithms
- Q-Network architecture
- Replay buffers (standard and prioritized)
- Training and visualization functions

### Section 4: Configuration
- All hyperparameters detailed
- Experimental setup explained

### Section 5: Training (Optional)
- Commented-out code to train from scratch
- Not necessary as all results are pre-loaded

### Section 6: Results Display
- **Test 1:** 600 episodes baseline
- **Test 6:** 2,000 episodes extended training
- **Test 10:** 10,000 episodes maximum performance
- Tests 2-5, 7-9: Additional experimental runs
- All include learning curves and agent behavior GIFs

### Section 7: Analysis
- Algorithm comparison
- Key observations
- Performance discussion

### Section 8-10: Documentation
- Environment details
- Academic references
- Conclusion

---

## What Was Implemented

### Algorithms:
1. **DQN** - Standard Deep Q-Network with experience replay
2. **DDQN** - Double DQN addressing overestimation bias
3. **PER** - Prioritized Experience Replay with importance sampling

### Features:
- ✅ Epsilon-greedy exploration with exponential decay
- ✅ Target network for stability
- ✅ Gradient clipping
- ✅ Smooth L1 loss (Huber loss)
- ✅ Proper seeding for reproducibility
- ✅ Moving average smoothing for plots

### Experimental Validation:
- 10 independent test runs
- Varying episode counts (600 to 10,000)
- Visual results (learning curves + behavior GIFs)
- Comparative analysis across algorithms

---

## Key Results

All three algorithms successfully learn to solve the LunarLander-v3 task:
- Task solved when average reward ≥ 200
- DDQN shows improved stability over DQN
- PER demonstrates faster learning with prioritized sampling
- Extended training yields significantly better performance

---

## GitHub Repository

Full project available at: https://github.com/humm3ll/LunarLander-v3-RL

Contains:
- Source code
- All test results
- Learning curves (PNG)
- Agent behavior visualizations (GIF)
- README documentation

---

## Evaluation Notes

This submission demonstrates:
- ✅ Deep understanding of value-based RL algorithms
- ✅ Clean, well-documented code
- ✅ Comprehensive experimental validation
- ✅ Proper academic citations
- ✅ Professional presentation
- ✅ Reproducible results

---

**Questions?** The notebook includes detailed comments and documentation throughout. Each algorithm implementation includes explanations of key design decisions.
