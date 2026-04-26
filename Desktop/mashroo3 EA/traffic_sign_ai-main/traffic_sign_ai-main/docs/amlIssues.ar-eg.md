# مراجعة بايبلاين التدريب — `training/training.ipynb`

---

## أخطاء حرجة (CRITICAL)

### 1) الإيقاف المبكر (Early Stopping) بايظ تمامًا
**المكان:** لوب التدريب — المتغير `counter` بيتعمله init بـ `0` بس **عمره ما بيتزوّد ولا بيتصفّر**.
**المشكلة:** الشرط `if counter >= patience: break` مش هيشتغل أبدًا. كده ميكانيزم الـearly stopping بقى كود ميّت وبيعمل إحساس مزيف إن فيه حماية من الـoverfitting.
**الحل:**
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

### 2) أفضل موديل مش بيتحفظ أثناء التدريب
**المكان:** لوب التدريب.
**المشكلة:** ناتج التدريب بيبين إن رسالة `"✅ Saved best model"` بتتطبع في كذا epoch، لكن **مفيش أي `torch.save` جوه لوب التدريب**. الموديل بيتحفظ بس بعد ما كل الـepochs تخلص — يعني بيتحفظ موديل آخر epoch مش أفضل موديل على الـvalidation. لو حصل overfitting في الآخر، أحسن checkpoint بيتفقد للأبد.
**الحل:** ضيف الحفظ جوه اللوب زي ما هو موضح في المشكلة رقم #1.

---

### 3) `RandomHorizontalFlip` غلط معنويًا على إشارات المرور
**المكان:** `train_transform`.
**المشكلة:** قلب إشارات المرور أفقيًا **بيغيّر معناها**. مثال: لافتة "Turn Right Ahead" لو اتقلبت بتبقى شبه "Turn Left Ahead"، لكن الليبل لسه "Turn Right Ahead". ده بيدخل بيانات تدريب mislabeled وبيبوّظ الدقة خصوصًا في الكلاسات الاتجاهية.
**الحل:** شيل `RandomHorizontalFlip()`. الـaugmentations الآمنة لإشارات المرور غالبًا: لف بسيط (`±15°`)، تغيير بسيط في السطوع/الكونتراست، وزووم — ودي موجودة بالفعل.

---

### 4) لوج إعدادات التجربة مسجّل Learning Rate غلط
**المكان:** Cell 29 — `"learning_rate": 0.001`.
**المشكلة:** الـLR الحقيقي للـoptimizer هو `3e-4 = 0.0003` مش `0.001`. كده ملف الإعدادات اللي المفروض يساعد على إعادة التجربة `config.json` فيه بيانات غلط. أي حد هيحاول يكرر التجربة من `config.json` هيستخدم LR غلط.
**الحل:**
```python
config = {
    "learning_rate": 3e-4,  # must match the Adam call
    ...
}
```

---

## مشاكل عالية (HIGH)

### 5) المتغير `weights` معمول له Shadowing
**المكان:** Cell 7 بتعرّف `weights = 1. / class_counts` (NumPy array لوزن الكلاسات). Cell 11 بتعرّف تاني `weights = MobileNet_V2_Weights.DEFAULT`.
**المشكلة:** لو Cell 11 اتشغّلت تاني أو الـcells اتنفذت بترتيب مختلف، الـ`class_weights` tensor (المشتق من `weights` الأولاني) ممكن يتحسب من متغير غلط. كمان الـshadowing بيخلّي الكود مش واضح.
**الحل:** غيّر أسماء المتغيرات عشان مفيش تصادم:
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

### 6) الـOptimizer بيتعمل على كل الـParameters حتى المتجمّدة
**المكان:** Cell 15 — `optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)`.
**المشكلة:** الـbackbone متجمّد وقتها، بس الـoptimizer لسه ماسك مراجع لكل باراميترز الـbackbone. ده بيهدر ميموري وبيكبّر الـstate dict بتاع الـoptimizer بدون داعي. ولما تفك تجميد الـbackbone في epoch 5، قيم Adam moments للـbackbone بتكون stale (اتعملت init بصفر بس ما اتحدّثتش)، وده ممكن يعمل spike في الـeffective LR.
**الحل:** اعمل optimizer على الـtrainable params بس، وبعدين حدّث/ضيف param group وقت الـunfreeze:
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

### 7) `num_workers=2` بيعمل مشاكل على Windows جوه Jupyter
**المكان:** `train_loader` — `num_workers=2`.
**المشكلة:** على Windows، multiprocessing بتاع PyTorch DataLoader محتاج entry point يتغلف بـ `if __name__ == '__main__':`، وده مش بينطبق في Jupyter notebooks. ده بيعمل `BrokenPipeError` عشوائي أو تهنيج، خصوصًا في سيشن تفاعلي.
**الحل:** استخدم `num_workers=0` في notebooks على Windows، أو استخدم `persistent_workers=True` مع `num_workers=2` لو PyTorch >= 1.7:
```python
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
    num_workers=0  # safe for Windows Jupyter
)
```

---

### 8) رسم منحنيات التدريب بيخلط Metrics مش قابلة للمقارنة
**المكان:** Cell 19 — بيرسم `train_losses` و`val_accuracies` على نفس المحور.
**المشكلة:** الـloss (ممكن يبقى 0–3+ ومش محدود) والـaccuracy (0–1) مقياسهم مختلف ومعناهم مختلف. رسمهم على نفس الـy-axis بيطلع شكل مُضلّل لأن فرق المقياس بيشوّه العلاقة.
**الحل:** سجّل `val_losses` كمان، وارسم خسارة التدريب/الـval سوا، والدقة لوحدها، أو استخدم محورين (`ax.twinx()`):
```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(train_losses, label="Train Loss")
ax1.plot(val_losses, label="Val Loss")
ax2.plot(val_accuracies, label="Val Accuracy")
```

---

### 9) الـConfusion Matrix مش مقروءة
**المكان:** Cell 21 — `sns.heatmap(cm)` من غير أرقام جوه الخلايا، ومن غير labels للمحاور، ومن غير أسماء الكلاسات.
**المشكلة:** Heatmap 43×43 من غير تسميات تقريبًا ملهاش قيمة تشخيصية. مش هتعرف أي كلاس بيتلخبط مع أنهي.
**الحل:**
```python
class_labels = [train_dataset.classes[i] for i in range(NUM_CLASSES)]
plt.figure(figsize=(20, 18))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_labels,
            yticklabels=class_labels, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
```

---

## مشاكل متوسطة (MEDIUM)

### 10) StepLR بـ `gamma=0.3` قاسي جدًا
**المكان:** Cell 15 — `StepLR(step_size=3, gamma=0.3)`.
**المشكلة:** الـLR بينزل 70% كل 3 epochs. بعد 6 epochs الـLR يبقى `3e-4 × 0.09 = 2.7e-5`، وده قليل جدًا خصوصًا لما تفك تجميد الـbackbone في epoch 5. ممكن الموديل ما يتعلمش كويس بعد الـunfreeze.
**التوصية:** استخدم `gamma=0.5` أو بدّل لـ `CosineAnnealingLR` (عادةً أنسب في transfer learning):
```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
```

---

### 11) مفيش Differential Learning Rates وقت فك تجميد الـBackbone
**المكان:** لوب التدريب، epoch 5 unfreeze.
**المشكلة:** لما الـbackbone يتفك تجميده، بياخد نفس الـLR بتاع classifier head. الأفضل في transfer learning إن الـbackbone يكون LR أقل بحوالي 10× عشان ما يبوّظش الـfeatures المتعلّمة.
**الحل:** زي المشكلة #6 — استخدم `add_param_group` مع `lr=3e-5` للـbackbone.

---

### 12) ترتيب فهارس الكلاسات ممكن ما يطابقش Class IDs الأصلية
**المكان:** كل خلايا التقييم.
**المشكلة:** `ImageFolder` بيعيّن indices للكلاسات حسب **الترتيب الأبجدي** لأسماء الفولدرات (أسماء متضبطة زي `Ahead_only`, `Ahead_or_left`, …). ده مختلف تمامًا عن GTSRB class IDs الأصلية (0=Speed limit 20, 1=Speed limit 30, …). `classification_report` بيستخدم indices بتاعة ImageFolder (0–42) مش IDs الأصلية.
**الأثر:** الـ`class_to_idx` mapping بيتحفظ في الـcheckpoint، بس أي كود لاحق بيفترض إن label `0 = Speed limit 20` هيطلع نتائج غلط.
**الحل:** لازم دايمًا تستخدم `train_dataset.class_to_idx` أو `train_dataset.classes` في تفسير الـpredictions. ووثّق النقطة دي بوضوح أو اعمل utility للـreverse-mapping.

---

### 13) مكان `optimizer.zero_grad()` مش ستاندرد
**المكان:** لوب التدريب — `zero_grad()` بيتنادى بعد حساب الـloss، مش قبل الـforward pass.
**الترتيب الحالي:** forward → loss → zero_grad → backward → step
**الترتيب المعتاد:** zero_grad → forward → loss → backward → step
وظيفيًا ممكن يبقى نفس النتيجة لو فيه loss واحدة، لكن الترتيب ده مُربك وممكن يعمل bugs لو هتضيف gradient accumulation بعدين.
**الحل:**
```python
optimizer.zero_grad()
outputs = model(images)
loss = criterion(outputs, labels)
loss.backward()
optimizer.step()
```

---

### 14) مفيش Logging للوقت أو GPU memory أو الـLR
**المشكلة:** مفيش زمن لكل epoch، ولا تتبّع لاستهلاك GPU memory، ولا تسجيل للـlearning rate مع كل خطوة. ده بيصعّب الـdebugging والتحليل وإعادة الإنتاج.
**الحل (الحد الأدنى):**
```python
import time
start = time.time()
# ... epoch ...
print(f"Epoch {epoch+1} | Loss: {epoch_loss:.4f} | Val Acc: {acc:.4f} | "
      f"LR: {scheduler.get_last_lr()[0]:.6f} | Time: {time.time()-start:.1f}s")
```

---

## جدول الملخص

| # | الشدة | المشكلة |
|---|-------|---------|
| 1 | حرج | عدّاد الـEarly stopping عمره ما بيتزوّد — كود ميّت |
| 2 | حرج | أفضل checkpoint للموديل مش بيتحفظ جوه لوب التدريب |
| 3 | حرج | `RandomHorizontalFlip` بيبوّظ ليبلات الإشارات الاتجاهية |
| 4 | حرج | `config.json` مسجّل LR = `0.001` لكن الحقيقي `3e-4` |
| 5 | عالي | Shadowing لمتغير `weights` بين Cell 7 وCell 11 |
| 6 | عالي | الـOptimizer ماسك params متجمّدة — هدر ومشاكل Adam stale |
| 7 | عالي | `num_workers=2` يعمل مشاكل multiprocessing على Windows/Jupyter |
| 8 | عالي | رسم loss مع accuracy على نفس المحور — شكل مضلل |
| 9 | عالي | Confusion matrix من غير labels/annotations |
| 10 | متوسط | `StepLR gamma=0.3` شديد بعد unfreeze |
| 11 | متوسط | مفيش LR مختلف للـbackbone vs head وقت unfreeze |
| 12 | متوسط | indices بتاعة ImageFolder مختلفة عن GTSRB IDs الأصلية |
| 13 | متوسط | `zero_grad()` في مكان غير ستاندرد |
| 14 | متوسط | مفيش logging للـLR/الوقت/الميموري |
