# Training Pipeline Review — `training/training.ipynb`

---

## CRITICAL BUGS

### 1. Early Stopping Is Completely Broken
**Location:** Training loop — `counter` is initialized to `0` but **never incremented or reset**.  
**Problem:** `if counter >= patience: break` will never fire. The early stopping mechanism is dead code that gives a false sense of overfitting protection.  
**Fix:**
```python
best_acc = 0
patience = 3
counter = 0

# Inside the epoch loop, after computing val acc:
if acc > best_acc:
    best_acc = acc
    counter = 0
    torch.save(model.state_dict(), "outputs/model/best_model.pth")
    print("Saved best model")
else:
    counter += 1
    if counter >= patience:
        print("Early stopping triggered")
        break
```

---

### 2. Best Model Is Never Saved During Training
**Location:** Training loop.  
**Problem:** The training output shows `"✅ Saved best model"` printed on multiple epochs, but **there is no `torch.save` call inside the training loop**. The model is only saved after all epochs complete — meaning the final-epoch model is saved, not the best validation model. If the model overfits in later epochs, the best checkpoint is permanently lost.  
**Fix:** Add the save inside the loop as shown in issue #1 above.

---

### 3. `RandomHorizontalFlip` Is Semantically Wrong for Traffic Signs
**Location:** `train_transform`.  
**Problem:** Horizontally flipping traffic signs **changes their meaning**. A "Turn Right Ahead" sign flipped looks like "Turn Left Ahead", but the label stays "Turn Right Ahead". This injects mislabeled training data and directly degrades model accuracy on directional classes.  
**Fix:** Remove `RandomHorizontalFlip()`. Safe augmentations for traffic signs are small rotations (`±15°`), brightness/contrast jitter, and zoom — which are already present.

---

### 4. Experiment Config Logs Wrong Learning Rate
**Location:** Cell 29 — `"learning_rate": 0.001`.  
**Problem:** The actual optimizer LR is `3e-4 = 0.0003`, not `0.001`. The config file used for experiment reproducibility contains incorrect data. Anyone trying to reproduce the experiment from `config.json` will use the wrong LR.  
**Fix:**
```python
config = {
    "learning_rate": 3e-4,  # must match the Adam call
    ...
}
```

---

## HIGH ISSUES

### 5. `weights` Variable Is Shadowed
**Location:** Cell 7 defines `weights = 1. / class_counts` (NumPy array for class weighting). Cell 11 redefines `weights = MobileNet_V2_Weights.DEFAULT`.  
**Problem:** If cell 11 is re-run or cells are run out of order, the `class_weights` tensor (derived from the original `weights`) may be computed from the wrong variable. The shadowing also makes the code confusing to read.  
**Fix:** Rename to avoid collision:
```python
# Cell 7
inv_class_weights = 1. / class_counts
sample_weights = [inv_class_weights[label] for label in labels]
class_weights = torch.tensor(inv_class_weights, dtype=torch.float32).to(DEVICE)

# Cell 11
pretrained_weights = MobileNet_V2_Weights.DEFAULT
model = mobilenet_v2(weights=pretrained_weights)
```

---

### 6. Optimizer Is Created Over All Parameters, Including Frozen Ones
**Location:** Cell 15 — `optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)`.  
**Problem:** The backbone is frozen at this point, but the optimizer still holds references to all backbone parameters. This wastes memory and makes the optimizer state dict larger than necessary. When the backbone is unfrozen at epoch 5, the Adam moment estimates for backbone params are stale (initialized at zero but never updated), which can cause a spike in effective LR.  
**Fix:** Initialize the optimizer with only the trainable parameters, then recreate or update param groups on unfreeze:
```python
# Initial: only classifier
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4
)

# At epoch 5 unfreeze, add backbone with a lower LR
for param in model.features.parameters():
    param.requires_grad = True
optimizer.add_param_group({"params": model.features.parameters(), "lr": 3e-5})
```

---

### 7. `num_workers=2` Causes Issues on Windows in Jupyter
**Location:** `train_loader` — `num_workers=2`.  
**Problem:** On Windows, PyTorch multiprocessing for DataLoader requires the entry point to be guarded by `if __name__ == '__main__':`, which doesn't apply in Jupyter notebooks. This causes random `BrokenPipeError` or hangs, especially in interactive sessions.  
**Fix:** Set `num_workers=0` in notebooks on Windows, or use `persistent_workers=True` with `num_workers=2` for PyTorch >= 1.7:
```python
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
    num_workers=0  # safe for Windows Jupyter
)
```

---

### 8. Training Curves Plot Mixes Incompatible Metrics
**Location:** Cell 19 — plots `train_losses` and `val_accuracies` on the same axis.  
**Problem:** Loss (unbounded, typically 0–3+) and accuracy (0–1) have different scales and different meanings. Plotting them together on one y-axis produces a misleading chart where scale differences distort the visual relationship.  
**Fix:** Track `val_losses` as well, then plot loss curves together and accuracy separately, or use a dual y-axis (`ax.twinx()`):
```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(train_losses, label="Train Loss")
ax1.plot(val_losses, label="Val Loss")
ax2.plot(val_accuracies, label="Val Accuracy")
```

---

### 9. Confusion Matrix Is Unreadable
**Location:** Cell 21 — `sns.heatmap(cm)` with no annotations, no axis labels, no class names.  
**Problem:** A 43×43 unlabeled heatmap conveys almost no diagnostic information. You cannot tell which classes are confused with which.  
**Fix:**
```python
class_labels = [train_dataset.classes[i] for i in range(NUM_CLASSES)]
plt.figure(figsize=(20, 18))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_labels,
            yticklabels=class_labels, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
```

---

## MEDIUM ISSUES

### 10. StepLR `gamma=0.3` Is Excessively Aggressive
**Location:** Cell 15 — `StepLR(step_size=3, gamma=0.3)`.  
**Problem:** LR decays by 70% every 3 epochs. After 6 epochs the LR is `3e-4 × 0.09 = 2.7e-5`, which is very low for fine-tuning an unfrozen backbone (which happens at epoch 5). The model may not learn effectively after unfreeze.  
**Recommendation:** Use `gamma=0.5` or switch to `CosineAnnealingLR` which is more principled for transfer learning:
```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
```

---

### 11. No Differential Learning Rates on Backbone Unfreeze
**Location:** Training loop, epoch 5 unfreeze.  
**Problem:** When the backbone is unfrozen, it receives the same learning rate as the classifier head. Best practice in transfer learning is to use a 10x lower LR for the pretrained backbone to avoid destroying pretrained features.  
**Fix:** See issue #6 — use `add_param_group` with `lr=3e-5` for the backbone.

---

### 12. Class Index Ordering May Not Match Original Class IDs
**Location:** All evaluation cells.  
**Problem:** `ImageFolder` assigns class indices by **alphabetical sort** of the folder names (cleaned names like `Ahead_only`, `Ahead_or_left`, …). The original GTSRB class IDs (0=Speed limit 20, 1=Speed limit 30, …) are completely different. The `classification_report` uses ImageFolder indices (0–42), not original IDs.  
**Impact:** The `class_to_idx` mapping is saved in the checkpoint, but any downstream code that assumes label `0 = Speed limit 20` will produce wrong results.  
**Fix:** Always use `train_dataset.class_to_idx` or `train_dataset.classes` when interpreting predictions. Document this clearly or add a reverse-mapping utility.

---

### 13. `optimizer.zero_grad()` Placement Is Non-Standard
**Location:** Training loop — `zero_grad()` is called after computing the loss, not before the forward pass.  
**Current order:** forward → loss → zero_grad → backward → step  
**Standard order:** zero_grad → forward → loss → backward → step  
While functionally equivalent for a single-loss scenario, the non-standard order is confusing and can cause subtle bugs if gradient accumulation is added later.  
**Fix:**
```python
optimizer.zero_grad()
outputs = model(images)
loss = criterion(outputs, labels)
loss.backward()
optimizer.step()
```

---

### 14. No GPU Memory / Training Time Logging
**Problem:** There is no timing per epoch, no GPU memory usage tracking, and no log of the learning rate at each step. This makes post-hoc debugging and reproducibility analysis difficult.  
**Fix:** Add at minimum:
```python
import time
start = time.time()
# ... epoch ...
print(f"Epoch {epoch+1} | Loss: {epoch_loss:.4f} | Val Acc: {acc:.4f} | "
      f"LR: {scheduler.get_last_lr()[0]:.6f} | Time: {time.time()-start:.1f}s")
```

---

## SUMMARY TABLE

| # | Severity | Issue |
|---|----------|-------|
| 1 | Critical | Early stopping counter never incremented — mechanism is dead code |
| 2 | Critical | Best model checkpoint never saved inside the training loop |
| 3 | Critical | `RandomHorizontalFlip` corrupts directional sign labels |
| 4 | Critical | Config logs LR as `0.001` but actual LR is `3e-4` |
| 5 | High | `weights` variable shadowing between cell 7 and cell 11 |
| 6 | High | Optimizer holds frozen params — inefficient and causes stale Adam moments |
| 7 | High | `num_workers=2` causes Windows/Jupyter multiprocessing errors |
| 8 | High | Train loss and val accuracy plotted on same axis — misleading scale |
| 9 | High | Confusion matrix has no annotations or class labels |
| 10 | Medium | `StepLR gamma=0.3` too aggressive after backbone unfreeze |
| 11 | Medium | No differential LR for backbone vs. classifier on unfreeze |
| 12 | Medium | ImageFolder alphabetical indices vs. original GTSRB class IDs |
| 13 | Medium | `zero_grad()` in non-standard position |
| 14 | Medium | No per-epoch LR / timing logs |
