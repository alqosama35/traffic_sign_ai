# مراجعة بايبلاين البيانات — `setup_dataset.py` و `EDA/EDA.ipynb`

---

## setup_dataset.py

### أخطاء حرجة (CRITICAL)

#### 1) `class_id` بيتسجّل Float في ملف الـAnnotations CSV
**المكان:** `build_annotation()` — لوب الـreverse-mapping بيبدأ بـ `class_id = None` كقيمة افتراضية.
**المشكلة:** لو أي `class_id` فضل `None`، Pandas بتغيّر نوع العمود كله لـ `float64`، فكل الـIDs بتبقى زي `35.0` بدل `35`. ده ظاهر فعلًا في ناتج الـEDA (`class_id: 35.0`).
**الحل:**
```python
# After the reverse-mapping loop, raise an error instead of silently writing None
if class_id is None:
    print(f"WARNING: no class_id found for folder '{class_name}' — skipping")
    continue
```
وكمان اعمل cast صريح قبل الحفظ:
```python
df["class_id"] = df["class_id"].astype(int)
```

---

#### 2) فقدان بيانات بشكل صامت عند أخطاء الصور
**المكان:** `resize_image()` — `except Exception as e: print(...)` بيبلع كل الأخطاء.
**المشكلة:** أي صورة بايظة/مفقودة بتتخطّى بصمت، وده بيسيب بيانات ناقصة من غير ما تعرف كام صورة ضاعت.
**الحل:** اجمع الفشل واطبع ملخص في الآخر:
```python
failed = []
# inside resize_image, append to failed instead of just printing
# after the loop: if failed: print(f"WARNING: {len(failed)} images failed")
```

---

#### 3) `shutil.rmtree("dataset")` من غير أي حماية
**المكان:** `prepare_data()` سطر 189.
**المشكلة:** بيحذف مجلد `dataset/` بالكامل بدون أي تأكيد. إعادة تشغيل السكربت بتدمّر داتا مُعالجة قبل كده من غير تحذير أو backup.
**الحل:** ضيف guard flag أو prompt:
```python
if os.path.exists("dataset"):
    print("WARNING: 'dataset/' already exists and will be deleted. Press Ctrl+C to cancel.")
    import time; time.sleep(5)
    shutil.rmtree("dataset")
```

---

### مشاكل عالية (HIGH)

#### 4) مسارات Hardcoded وبحروف كبيرة مخصوص لـWindows — حسّاسة للـOS
**المكان:** `prepare_data()` — بيستخدم `"Train"` و`"Test"` بحروف كبيرة عشان يطابق شكل Kaggle download.
**المشكلة:** على Linux/macOS (أو لو Kaggle غيّر الـlayout) المسارات ممكن تفشل/تبوّظ.
**الحل:** استخدم `os.path.join` مع بحث case-insensitive أو وثّق بوضوح شكل الفولدرات المتوقع.

---

#### 5) `build_annotation` مفيهوش فلتر لامتدادات الملفات
**المكان:** `build_annotation()` — بيلف على `os.listdir(class_path)` من غير فلترة.
**المشكلة:** أي ملف مش صورة (`.DS_Store`, `thumbs.db`, ملفات مؤقتة) هيتضاف كسطر في `annotations.csv`.
**الحل:**
```python
VALID_EXTS = {".png", ".jpg", ".jpeg", ".ppm"}
for img in os.listdir(class_path):
    if Path(img).suffix.lower() not in VALID_EXTS:
        continue
```

---

#### 6) `clean_name` ممكن يطلع Collisions
**المكان:** `clean_name()`.
**المشكلة:** `"No passing >3.5t"` → `"No_passing_3.5t"` و(لو فيه) كلاس `"No passing 3.5t"` هتطلع نفس الاسم. دلوقتي الـclass list مفيهاش collision، بس الدالة هشة ومفيش assertion.
**الحل:** ضيف assertion في البداية إن مفيش اسمين بينتجوا نفس الـcleaned name:
```python
cleaned = [clean_name(v) for v in CLASS_NAMES.values()]
assert len(cleaned) == len(set(cleaned)), "Cleaned name collision detected"
```

---

#### 7) متغيرات مش مستخدمة في `split_train_val`
**المكان:** سطور 69–70 — `train_out` و`val_out` بيتعرّفوا بس مش بيتستخدموا؛ المسارات بتتكوّن تاني جوه اللوب عن طريق string concatenation.
**الحل:** يا إمّا تشيل `train_out`/`val_out`، يا إمّا تستخدمهم بشكل ثابت جوه اللوب.

---

### مشاكل متوسطة (MEDIUM)

#### 8) مفيش Progress Bar للـResize Loop
**المكان:** `split_train_val()` و`organize_test()`.
**المشكلة:** تغيير حجم ~50k صورة من غير أي feedback بيخلّي السكربت يبان كأنه معلّق.
**الحل:** استخدم `tqdm`:
```python
from tqdm import tqdm
for img in tqdm(img_list, desc=f"{split}/{class_name}"):
```

---

#### 9) استخدام `print()` بدل `logging`
**المشكلة:** كل الرسائل بـ `print()`، فمفيش تحكم في الـverbosity ولا سهل تكتب لوج في ملف.
**الحل:** بدّل لـ `logging.getLogger(__name__)` وخلي level يتحدد من config.

---

#### 10) ترقيم أقسام مكرر
**المكان:** الكومنتات بتسمّي `build_annotation` و`prepare_data` الاتنين `# 5)`.
**الحل:** عدّلهم لـ `# 5)` و`# 6)`.

---

#### 11) `os.path.basename` في `organize_test` هِش
**المكان:** `organize_test()` — `img_name = os.path.basename(row["Path"])`.
**المشكلة:** بيفترض إن عمود `Path` في CSV دايمًا فيه اسم ملف بس. لو Kaggle غيّر الصيغة لـمسار نسبي (`Test/00000.png`) فـ`basename` هيشتغل، بس لو فيه prefix/duplicate dirs ممكن الملف مايتلقاش في `os.path.join(test_path, img_name)`.
**الحل:** اتأكد إن الملف موجود قبل `resize_image`:
```python
if not os.path.exists(src):
    print(f"Missing: {src}")
    continue
```

---

## EDA/EDA.ipynb

### أخطاء حرجة (CRITICAL)

#### 12) مسار مطلق Hardcoded: `D:/AML_Project`
**المكان:** Cell 17 — `BASE_DIR = Path(r"D:/AML_Project")`.
**المشكلة:** النوتبوك مش portable خالص. أي حد يشغّله على جهاز/مسار مختلف هيقابل `FileNotFoundError` و`cv2.imread` هيفشل.
**الحل:** استنتج الـbase path ديناميكيًا:
```python
BASE_DIR = Path.cwd().parent  # or use __file__ if converted to .py
```
أو اقراه من environment variable / config.

---

#### 13) `cv2.imread` ممكن يرجّع `None` → كراش في `show_samples`
**المكان:** `show_samples()` — `img = cv2.imread(row.image_path)` وبعدها مباشرة `cv2.cvtColor(img, ...)`.
**المشكلة:** لو المسار غلط (مثلاً بسبب hardcoding بتاع `D:/AML_Project`)، `imread` بترجع `None` والسطر اللي بعده بيطلع `cv2.error: NULL pointer`.
**الحل:**
```python
img = cv2.imread(row.image_path)
if img is None:
    print(f"Cannot read: {row.image_path}")
    continue
```

---

### مشاكل عالية (HIGH)

#### 14) `df.sample()` من غير `random_state` — غير قابل لإعادة الإنتاج
**المكان:** `show_samples(df)` وكمان `df["image_path"].sample(500)` في خلية توزيع الأحجام.
**المشكلة:** كل restart للكيرنل بيطلع sample مختلف، فالنوتبوك مش reproducible.
**الحل:** مرّر `random_state=42` في الاتنين.

---

#### 15) حل مؤقت Deduplication للمسارات بطريقة هشة
**المكان:** Cell 17 — `df["image_path"].str.replace("dataset/dataset", "dataset")`.
**المشكلة:** ده workaround لمشكلة في `image_path` داخل `annotations.csv` (تكرار `dataset/` من `build_annotation`). إصلاح السبب الجذري هيخلّي الهَك ده مش محتاج.
**الإصلاح الجذري في `build_annotation`:** استخدم مسارات نسبية (زي `split/class_name/img`) وخلي المستهلك يحوّلها لمسار مطلق.

---

### مشاكل متوسطة (MEDIUM)

#### 16) تحليل عدم التوازن (Imbalance) ناقص
**المشكلة:** النوتبوك بيقول إن فيه عدم توازن 11x، بس مفيش تحليل يساعدك تعمل mitigation (مين الكلاسات الأقل؟ هل الـsplit محافظ على النِسب؟ هل ينفع oversampling/augmentation؟). كده النتيجة مش actionable.

---

#### 17) معظم الرسومات مش بتتسجل على الديسك
**المشكلة:** معظم الخلايا بتعمل `plt.show()` بس. للتوثيق وإعادة الإنتاج محتاج تحفظ artifacts.
**الحل:** ضيف `plt.savefig("outputs/eda/<name>.png", dpi=150, bbox_inches="tight")` قبل `plt.show()`.

---

#### 18) مفيش Findings مكتوبة — بس عناوين
**المشكلة:** أغلب خلايا الماركداون مجرد عناوين (زي `# Top 10 Classes`) من غير ملاحظات أو استنتاجات أو ملاحظات جودة بيانات. الـEDA المفروض يوثّق إيه اللي اتشاف مش بس يرسم.
