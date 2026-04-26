import cv2
import numpy as np
import os
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.utils import shuffle
from skimage.feature import local_binary_pattern

# --- MODELLER ---
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

# --- FEATURE SELECTION ---
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.feature_selection import SelectKBest, chi2, SelectFromModel
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# =============================================================================
# AYARLAR
# =============================================================================
DATA_PATH = "Dataset"  # Bu klasörün var olduğundan emin olun
IMG_SIZE = 128
LBP_RADIUS = 3
LBP_POINTS = 8 * LBP_RADIUS
TARGET_NAMES = ['Positive (Happy)', 'Surprise', 'Negative (Bad)']

# =============================================================================
# 1. TEMEL FONKSİYONLAR
# =============================================================================
def get_label_from_folder(folder_name):
    name = folder_name.lower()
    if 'happiness' in name:
        return 0
    elif 'surprise' in name:
        return 1
    elif any(x in name for x in ['disgust', 'repression', 'sadness', 'fear', 'anger']):
        return 2
    else:
        return -1

def extract_lbp(image):
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    lbp = local_binary_pattern(gray, LBP_POINTS, LBP_RADIUS, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, LBP_POINTS + 3), range=(0, LBP_POINTS + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

# =============================================================================
# 2. VERİ YÜKLEME
# =============================================================================
print(f"--- MİKRO-İFADE ANALİZİ ---")
data_by_class = {0: [], 1: [], 2: []}

if not os.path.exists(DATA_PATH):
    print(f"HATA: '{DATA_PATH}' klasörü bulunamadı! Lütfen yolu kontrol edin.")
    exit()

print("Veriler yükleniyor...")
for folder_name in os.listdir(DATA_PATH):
    folder_full_path = os.path.join(DATA_PATH, folder_name)
    if os.path.isdir(folder_full_path):
        label = get_label_from_folder(folder_name)
        if label != -1:
            image_files = glob.glob(os.path.join(folder_full_path, "*"))
            for img_path in image_files:
                if img_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    img = cv2.imread(img_path)
                    if img is not None:
                        data_by_class[label].append(extract_lbp(img))

# Dengeleme kontrolü
if len(data_by_class[0]) == 0 or len(data_by_class[1]) == 0 or len(data_by_class[2]) == 0:
    print("HATA: Sınıflardan birinde hiç veri bulunamadı. Klasör yapısını kontrol edin.")
    exit()

min_len = min(len(data_by_class[0]), len(data_by_class[1]), len(data_by_class[2]))
print(f"   -> Dengeleme: Her sınıftan {min_len} örnek alınıyor.")

X = []
y = []
for label in [0, 1, 2]:
    X.extend(data_by_class[label][:min_len])
    y.extend([label] * min_len)

X = np.array(X)
y = np.array(y)
X, y = shuffle(X, y, random_state=42)

# Orijinal Veri (Test/Train)
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"   -> Orijinal Özellik Sayısı: {X_train_raw.shape[1]}")

# =============================================================================
# 3. SELECTOR ve MODEL TANIMLARI
# =============================================================================
selectors = {
    '1. Original (None)': None,
    '2. PCA (10 Bileşen)': PCA(n_components=10),
    '3. PCA (20 Bileşen)': PCA(n_components=20),
    '4. PCA (%95 Varyans)': PCA(n_components=0.95),
    '5. LDA': LDA(n_components=2),
    '6. Ki-Kare (Top 20)': SelectKBest(score_func=chi2, k=20),
    '7. RF-Importance': SelectFromModel(RandomForestClassifier(n_estimators=50, random_state=42), threshold="mean")
}

models = {
    'SVM': GridSearchCV(SVC(random_state=42), {'C': [1, 10, 100], 'kernel': ['rbf']}, cv=3),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'k-NN': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Naive Bayes': GaussianNB()
}

results_log = []
best_acc = 0
best_model_name = ""
best_selector = ""
best_pred = []

# =============================================================================
# 4. TEST DÖNGÜSÜ
# =============================================================================
print("\nTESTLER BAŞLIYOR...")

for sel_name, selector in selectors.items():
    print(f"\n   >>> YÖNTEM: {sel_name}")
    try:
        # Seçici (Feature Selection) Uygulama
        if selector is None:
            X_train_curr, X_test_curr = X_train_raw, X_test_raw
        else:
            # PCA ve diğerleri ayrımı
            if "LDA" in sel_name or "Ki-Kare" in sel_name or "RF" in sel_name:
                # Supervised yöntemler y_train ister
                X_train_curr = selector.fit_transform(X_train_raw, y_train)
                X_test_curr = selector.transform(X_test_raw)
            else:
                # Unsupervised (PCA) sadece X ile çalışır
                X_train_curr = selector.fit_transform(X_train_raw)
                X_test_curr = selector.transform(X_test_raw)
    except Exception as e:
        print(f"      Hata (Feature Selection): {e}")
        continue

    n_feats = X_train_curr.shape[1]
    print(f"      (Özellik Sayısı: {n_feats})")

    for model_name, model in models.items():
        try:
            model.fit(X_train_curr, y_train)
            y_pred = model.predict(X_test_curr)
            acc = accuracy_score(y_test, y_pred)

            results_log.append({
                'Selection Method': sel_name,
                'Model': model_name,
                'Features': n_feats,
                'Accuracy': acc
            })
            print(f"      -> {model_name:<15}: %{acc * 100:.2f}")

            # En iyi modeli kaydet
            if acc > best_acc:
                best_acc = acc
                best_model_name = model_name
                best_selector = sel_name
                best_pred = y_pred
        except Exception as e:
            print(f"      Hata (Model): {model_name} -> {e}")

# =============================================================================
# 5. SONUÇLAR VE RAPORLAMA
# =============================================================================
print("\n" + "=" * 80)
print(f"{'ULTIMATE KARŞILAŞTIRMA TABLOSU':^80}")
print("=" * 80)

if not results_log:
    print("Hiçbir sonuç alınamadı. Lütfen veri setini ve kod akışını kontrol edin.")
    exit()

df_res = pd.DataFrame(results_log)
df_res = df_res.sort_values(by='Accuracy', ascending=False)
print(df_res.to_string(index=False))

# --- PCA Karşılaştırma Grafiği ---
pca_results = df_res[df_res['Selection Method'].str.contains("PCA")]
if not pca_results.empty:
    plt.figure(figsize=(10, 6))
    sns.barplot(data=pca_results, x='Model', y='Accuracy', hue='Selection Method')
    plt.title("PCA Performans Karşılaştırması (10 vs 20 vs %95)")
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.show()

# --- Şampiyon Model Detayları ---
print("\n" + "-" * 80)
print(f" >>> ŞAMPİYON: {best_selector} + {best_model_name}")
print(f" >>> SKOR: %{best_acc * 100:.2f}")
print("-" * 80)

print(f"\nŞAMPİYON MODELİN DETAYLI RAPORU:\n")
print(classification_report(y_test, best_pred, target_names=TARGET_NAMES))

# --- Şampiyon Confusion Matrix ---
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, best_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=TARGET_NAMES, yticklabels=TARGET_NAMES)
plt.title(f'KAZANAN: {best_selector} + {best_model_name}\n(Accuracy: %{best_acc * 100:.1f})')
plt.ylabel('Gerçek')
plt.xlabel('Tahmin')
plt.tight_layout()
plt.show()

# --- Genel Heatmap (Tüm Yöntemler vs Modeller) ---
try:
    pivot_table = df_res.pivot(index='Selection Method', columns='Model', values='Accuracy')
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_table, annot=True, cmap='RdYlGn', fmt=".2f", linewidths=.5)
    plt.title("Hangi Yöntem Hangi Model ile En İyi Çalıştı? (Accuracy)")
    plt.ylabel("Öznitelik Seçimi Yöntemi")
    plt.xlabel("Yapay Zeka Modeli")
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"Heatmap oluşturulurken hata: {e}")
