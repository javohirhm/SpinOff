# 📚 Week 2 Documentation Index

## 🚀 Start Here

**New to the project? Start with:**
1. **[GETTING_STARTED.md](GETTING_STARTED.md)** ⭐ - Quick 3-step guide to get training
2. **[WEEK2_SUMMARY.md](WEEK2_SUMMARY.md)** - Overview of what's been built

## 📖 Full Documentation

### For Implementation Details
- **[week2_guide.md](week2_guide.md)** - Complete Week 2 guide with:
  - Task breakdowns with time estimates
  - Detailed explanations of each component
  - Code templates and examples
  - Common issues and solutions
  - Success criteria

### For Quick Reference
- **[WEEK2_QUICKSTART.md](SpinOff/WEEK2_QUICKSTART.md)** - Step-by-step tutorial:
  - Environment setup
  - Data preparation
  - Training instructions
  - Troubleshooting guide
  - Week 3 preparation

### For Project Overview
- **[README.md](SpinOff/README.md)** - Project overview:
  - Objectives and goals
  - Timeline and milestones
  - Technical specifications
  - Team information

## 💻 Code Structure

```
SpinOff/
├── data/                  # Data processing modules
│   ├── preprocessing.py   # NIfTI → numpy conversion
│   ├── degradation.py     # Low-dose simulation
│   ├── dataset.py         # PyTorch Dataset
│   └── transforms.py      # Augmentation
├── models/
│   └── unet.py           # U-Net architecture
├── utils/
│   └── metrics.py        # PSNR, SSIM evaluation
├── scripts/
│   └── preprocess_ixi.py # Dataset preprocessing
└── train_unet.py         # Main training script
```

## 🎯 Quick Links

### Essential Commands
```bash
# Install
pip install -r requirements.txt

# Preprocess
python scripts/preprocess_ixi.py --input_dir data/IXI-T1 --output_dir data/processed

# Train
python train_unet.py --data_dir data/processed --epochs 100
```

### Key Files to Read
1. `GETTING_STARTED.md` - If you want to start immediately
2. `week2_guide.md` - If you want to understand everything
3. `WEEK2_QUICKSTART.md` - If you need step-by-step help
4. `WEEK2_SUMMARY.md` - If you want an overview

### Key Files to Run
1. `scripts/preprocess_ixi.py` - Prepare your data
2. `train_unet.py` - Train the model
3. Test scripts in quickstart guide - Verify everything works

## 📊 What You'll Get

After completing Week 2:
- ✅ Preprocessed MRI dataset (~18,000 slices)
- ✅ Trained U-Net model
- ✅ Training curves and metrics
- ✅ Baseline performance results

## 🔗 Navigation

**Want to...**
- **Start coding now?** → [GETTING_STARTED.md](GETTING_STARTED.md)
- **Understand the theory?** → [week2_guide.md](week2_guide.md)
- **Follow a tutorial?** → [WEEK2_QUICKSTART.md](SpinOff/WEEK2_QUICKSTART.md)
- **See what's built?** → [WEEK2_SUMMARY.md](WEEK2_SUMMARY.md)
- **Check project info?** → [README.md](SpinOff/README.md)

## 📝 File Descriptions

| File | Purpose | Read Time |
|------|---------|-----------|
| **GETTING_STARTED.md** | Quick start guide | 5 min |
| **WEEK2_SUMMARY.md** | What's been built | 10 min |
| **week2_guide.md** | Complete implementation guide | 30 min |
| **WEEK2_QUICKSTART.md** | Detailed tutorial | 20 min |
| **README.md** | Project overview | 10 min |

## ⏱️ Time Estimates

- **Setup:** 5-15 minutes
- **Data preprocessing:** 1-2 hours
- **Training (100 epochs):** 4-6 hours (GPU) or 24-48 hours (CPU)
- **Total Week 2:** ~8-12 hours

## 🆘 Need Help?

1. Check [GETTING_STARTED.md](GETTING_STARTED.md) for common issues
2. Read [WEEK2_QUICKSTART.md](SpinOff/WEEK2_QUICKSTART.md) for detailed troubleshooting
3. Review error messages (they're informative!)
4. Try reducing batch size if memory issues

## ✅ Week 2 Checklist

- [ ] Read GETTING_STARTED.md
- [ ] Install dependencies
- [ ] Download IXI dataset
- [ ] Run preprocessing
- [ ] Test data pipeline
- [ ] Test U-Net model
- [ ] Start training (even 2 epochs is fine!)
- [ ] Review training curves

**You're ready to go! 🚀**
