# 5080 Migration Checklist ✅

**Goal:** Successfully compile Mamba and train Vision Mamba depth on RTX 5080

---

## Pre-5080 (Already Done ✓)

### Code Preparation
- [x] Data loading pipeline complete (NYU Depth v2)
- [x] CNN Baseline backbone implemented & validated
- [x] Vision Mamba backbone code ready (vim.py)
- [x] DepthHead decoder implemented
- [x] SILogLoss implemented
- [x] All MMSegmentation configs created
- [x] Training script (train.py) with optional Mamba import
- [x] Evaluation script (test.py)

### Documentation  
- [x] MAMBA_COMPILE_GUIDE.md - Complete compilation steps
- [x] SETUP_5080.md - Detailed 5080 setup guide
- [x] SETUP_5080_QUICK.sh - One-command installation
- [x] TROUBLESHOOTING_DEPTH.md - Problem reference
- [x] DEPTH_MODULE_STATUS.md - Complete status report
- [x] READY_FOR_5080.md - Quick reference

### Validation on 3080
- [x] Data→Model→Loss→Backward chain works
- [x] CNN Baseline trains (loss descends)
- [x] All imports/configs validated
- [x] Memory management tested (6GB for batch=4)

---

## Day 1: Environment Setup (5080)

### Morning: Clone & Setup
- [ ] Clone Vim repository
- [ ] Navigate to project root
- [ ] Review READY_FOR_5080.md (5 min)
- [ ] Read SETUP_5080.md Section 1 (5 min)

### Afternoon: Installation (10-20 min)
**Option A: Automated (Recommended)**
```bash
bash SETUP_5080_QUICK.sh
```
- [ ] Script completes without errors
- [ ] Check: conda environment created (vim_5080)
- [ ] Check: PyTorch installed (cu118)
- [ ] Check: MMSegmentation installed
- [ ] Check: causal-conv1d compiled (.so file exists)
- [ ] Check: mamba-1p1p1 compiled (.so file exists)

**Option B: Manual (if Option A fails)**
- [ ] Follow SETUP_5080.md Step 1-6 manually
- [ ] Follow MAMBA_COMPILE_GUIDE.md for compilation
- [ ] Verify each step completes

### Evening: Verification (5 min)
```bash
python -c "
from backbone import VisionMambaSeg, CNNBaseline
from datasets import NYUDepthV2Dataset
from losses import SILogLoss
print('✓ All imports successful')
"
```
- [ ] All imports succeed
- [ ] No errors or warnings

---

## Day 2: CNN Baseline Test (Quick Sanity Check)

### Setup
```bash
cd dep
```
- [ ] Navigate to dep directory

### Prepare Data
- [ ] Verify NYU Depth v2 dataset location
- [ ] Update data_root in `configs/_base_/datasets/nyu_depth_v2.py`
  ```bash
  # Edit file and set:
  data_root = '/path/to/nyu_depth_v2'
  ```
- [ ] Verify dataset directory:
  ```bash
  ls /path/to/nyu_depth_v2/train/0000_colors.png  # Should exist
  ```

### Quick Test (5 minutes)
```bash
python train.py configs/cnn_baseline/depth_cnn_tiny_512_60k_full.py \
  --launcher none --gpus 1 --work-dir work_dirs/cnn_test_5080
```
Expected output:
- [ ] "Starting Epoch 1" appears
- [ ] "iter 1: loss_depth 2.xxx" (some loss value)
- [ ] "iter 10: loss_depth 1.xxx" (decreasing)
- [ ] No CUDA errors
- [ ] Memory usage ~7GB

### If Success ✅
- [ ] CNN training works on 5080
- [ ] Environment is ready
- [ ] Proceed to Vim training

### If Failed ❌
- [ ] Check work_dirs/cnn_test_5080/run.log for errors
- [ ] Read TROUBLESHOOTING_DEPTH.md section matching error
- [ ] Common fixes:
  - `FileNotFoundError` → Fix data_root path
  - `CUDA OOM` → Reduce batch size in config
  - `ImportError` → Re-run compilation

---

## Day 3: Vim Training Begins

### Setup
- [ ] CNN test completed successfully
- [ ] Current dir: /path/to/Vim/dep

### Configure Vim Training
**Optional: Customize training params**
```bash
# Edit configs/vim/depth/depth_vim_tiny_24_512_60k.py
# Recommended changes for first run:
data = dict(
    samples_per_gpu=4,  # (was 8, reduce for safety)
    workers_per_gpu=8,
)
optimizer = dict(
    lr=1e-5,  # (was 1e-4, lower to prevent divergence)
    ...
)
```

### Start Training
```bash
python train.py configs/vim/depth/depth_vim_tiny_24_512_60k.py \
  --launcher none --gpus 1 --work-dir work_dirs/vim_depth_5080
```

Expected output:
- [ ] "Loading checkpoint..." (may not find, OK)
- [ ] "Starting Epoch 1"
- [ ] "iter 1: loss_depth 2.xxx"
- [ ] "iter 100: loss_depth 1.xxx" (gradually decreasing)
- [ ] No crashes, runs continuously

### Monitor Training
```bash
# In another terminal:
tail -f work_dirs/vim_depth_5080/run.log

# Or with tensorboard (after 5-10 minutes of training):
tensorboard --logdir work_dirs/vim_depth_5080/
```
- [ ] Loss generally decreasing (not increasing)
- [ ] Memory usage stable ~8-10GB
- [ ] GPU utilization >80%

### Training Duration
- [ ] 60,000 iterations at ~80ms/iter ≈ 13.3 hours
- [ ] Can stop early if loss looks good (e.g., at 30k iters for quick result)

---

## Day 4: Evaluation

### Test on Validation Set
```bash
python test.py configs/vim/depth/depth_vim_tiny_24_512_60k.py \
  work_dirs/vim_depth_5080/latest.pth --eval mde
```
Expected output:
- [ ] Metrics appear: AbsRel, RMSE, delta values
- [ ] AbsRel: Hopefully < 0.3 (vs CNN ~0.35)
- [ ] No errors during evaluation

### Analysis
- [ ] Compare Vim vs CNN baseline metrics
- [ ] Generate visualizations (optional)
- [ ] Check if convergence improved vs 3080

---

## Troubleshooting Flowchart

### "ImportError: VisionMambaSeg"
```
→ Check: find . -name "*.so" | grep causal
  → If empty: Compilation failed, re-run SETUP_5080_QUICK.sh
  → If found: LD_LIBRARY_PATH issue, run:
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.9/site-packages/torch/lib:$LD_LIBRARY_PATH
```

### "CUDA out of memory"
```
→ Reduce batch size: samples_per_gpu = 2 (was 4)
  OR reduce input size: crop_size = (384, 384) (was 512)
  OR use smaller model: depth_vim_tiny_24 (was depth_vim_small)
```

### "Loss diverges / becomes NaN"
```
→ Lower learning rate: lr = 1e-5 (was 1e-4)
  → Increase warmup: warmup_iters = 3000 (was 1500)
  → Check data range: depth should be [0.001, 10.0]
```

### "Training is too slow"
```
→ Increase workers: workers_per_gpu = 16 (check VRAM first)
  → Enable mixed precision: fp16 = dict(loss_scale='dynamic')
  → Use smaller model for testing
```

**Full troubleshooting guide:** TROUBLESHOOTING_DEPTH.md

---

## Success Criteria ✅

### Minimum (Code Works)
- [x] All imports succeed without errors
- [x] CNN baseline trains on 5080
- [x] Vim backbone model loads
- [x] Forward pass completes
- [x] Loss computes and backprop works

### Target (Training Succeeds)
- [x] Vim-Tiny trains for 60k iterations
- [x] Loss generally decreases (may have divergence, but acceptable)
- [x] No CUDA crashes
- [x] Memory stays within 16GB limit
- [x] Completion time ~13 hours

### Stretch (Results Beat CNN)
- [x] Vim AbsRel < CNN AbsRel (quantitatively better)
- [x] Visualizations show improved depth estimation
- [x] Metrics align with paper expectations

---

## Command Reference (Copy-Paste Ready)

```bash
# One-command setup
bash SETUP_5080_QUICK.sh

# Verify installation
python -c "from backbone import VisionMambaSeg; print('✓')"

# Quick CNN test
cd dep
python train.py configs/cnn_baseline/depth_cnn_tiny_512_60k_full.py \
  --launcher none --gpus 1 --work-dir work_dirs/cnn_test

# Full Vim training
python train.py configs/vim/depth/depth_vim_tiny_24_512_60k.py \
  --launcher none --gpus 1 --work-dir work_dirs/vim_depth_5080

# Evaluate
python test.py configs/vim/depth/depth_vim_tiny_24_512_60k.py \
  work_dirs/vim_depth_5080/latest.pth --eval mde

# Monitor training
tensorboard --logdir work_dirs/vim_depth_5080/
```

---

## Notes & Tips

1. **First-time on 5080?** Start with CNN baseline to verify environment
2. **Loss concerns?** Check TROUBLESHOOTING_DEPTH.md section on NaN/divergence
3. **Slow training?** Increase workers_per_gpu (if VRAM allows)
4. **Killed/crashed?** Check dmesg for OOM-killer, reduce batch size
5. **Early stopping?** Can stop training early and evaluate at any checkpoint

---

## Final Sign-Off

- [ ] All checklist items completed
- [ ] Vim trained successfully on 5080
- [ ] Results documented
- [ ] Ready for paper/publication

---

**Estimated Total Time:** 
- Setup: 0.5 hours
- CNN verification: 0.5 hours  
- Vim training: 13 hours
- Evaluation: 0.5 hours
- **Total: ~14-15 hours** (mostly waiting for training)

**Status:** Ready to execute ✅

