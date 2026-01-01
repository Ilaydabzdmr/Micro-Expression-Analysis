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

# --- FEATURE SELECTION & BOYUT İNDİRGEME ---
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.feature_selection import SelectKBest, chi2, SelectFromModel, mutual_info_classif, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# =============================================================================
# AYARLAR
# =============================================================================
DATA_PATH = "Dataset"
IMG_SIZE = 128
LBP_RADIUS = 3
LBP_POINTS = 8 * LBP_RADIUS
TARGET_NAMES = ['Positive (Happy)', 'Surprise', 'Negative (Bad)']


# =============================================================================
# 1. VERİ İŞLEME VE OKUMA
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


print(f"--- MİKRO-İFADE PROJESİ: 6 SELECTION x 5 MODELS (ULTIMATE) ---")
print(f"1. Veri seti yükleniyor...")

data_by_class = {0: [], 1: [], 2: []}

if not os.path.exists(DATA_PATH):
    print("HATA: Dataset klasörü bulunamadı!")
    exit()

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

min_len = min(len(data_by_class[0]), len(data_by_class[1]), len(data_by_class[2]))
print(f"   -> Veri Dengeleniyor: Her sınıftan {min_len} örnek.")

X = []
y = []
for label in [0, 1, 2]:
    X.extend(data_by_class[label][:min_len])
    y.extend([label] * min_len)

X = np.array(X)
y = np.array(y)
X, y = shuffle(X, y, random_state=42)

# Orijinal (Ham) Veri Ayrımı
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"   -> Ham Özellik Sayısı: {X_train_raw.shape[1]}")

# =============================================================================
# 2. SELECTION VE MODEL TANIMLARI (BURASI GÜNCELLENDİ)
# =============================================================================

# A. FEATURE SELECTION YÖNTEMLERİ (6 ADET)
selectors = {
    # 1. BASELINE: Hiçbir şey yapma
    '1. Original (None)': None,

    # 2. DIMENSIONALITY REDUCTION (Unsupervised)
    '2. PCA (Varyans)': PCA(n_components=0.95),

    # 3. DIMENSIONALITY REDUCTION (Supervised)
    '3. LDA (Ayırt Edici)': LDA(n_components=2),

    # 4. FILTER METHOD (Statistical)
    '4. Chi-Square': SelectKBest(score_func=chi2, k=20),

    # 5. FILTER METHOD (Information Theory) - YENİ EKLENDİ
    '5. Mutual Info': SelectKBest(score_func=mutual_info_classif, k=20),

    # 6. WRAPPER METHOD (Recursive) - YENİ EKLENDİ (En güçlülerinden)
    # RFE yavaştır, bu yüzden LogisticRegression gibi hızlı bir modelle çalıştırıyoruz.
    '6. RFE (Recursive)': RFE(estimator=LogisticRegression(max_iter=1000), n_features_to_select=20, step=1)
}

# B. YAPAY ZEKA MODELLERİ (5 ADET)
models = {
    'SVM': GridSearchCV(SVC(random_state=42), {'C': [1, 10, 100], 'kernel': ['rbf']}, cv=3),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'k-NN': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Naive Bayes': GaussianNB()
}

results_log = []
best_global_acc = 0
best_global_model_name = ""
best_global_selector_name = ""
best_global_pred = []

# =============================================================================
# 3. BÜYÜK YARIŞMA (30 KOMBİNASYON)
# =============================================================================
print("\n2. BÜYÜK YARIŞMA BAŞLIYOR (6 Selectors x 5 Models = 30 Tests)...")

for sel_name, selector in selectors.items():
    print(f"\n   >>> SEÇİM YÖNTEMİ: {sel_name}")

    try:
        if selector is None:
            X_train_curr = X_train_raw
            X_test_curr = X_test_raw
        else:
            # RFE ve LDA gibi yöntemler Y etiketlerini de ister
            # PCA sadece X ister ama y vermenin zararı yoktur (ignore eder)
            X_train_curr = selector.fit_transform(X_train_raw, y_train)
            X_test_curr = selector.transform(X_test_raw)

    except Exception as e:
        print(f"      [ATLANDI] Bu yöntem hata verdi: {e}")
        continue

    n_feats = X_train_curr.shape[1]
    print(f"      (Özellik sayısı: {n_feats})")

    # Modelleri Yarıştır
    for model_name, model_inst in models.items():
        # Eğit
        model_inst.fit(X_train_curr, y_train)

        # Test Et
        y_pred = model_inst.predict(X_test_curr)
        acc = accuracy_score(y_test, y_pred)

        # Logla
        results_log.append({
            'Selection Method': sel_name,
            'Model': model_name,
            'Features': n_feats,
            'Accuracy': acc
        })

        print(f"      -> {model_name:<15}: %{acc * 100:.2f}")

        if acc > best_global_acc:
            best_global_acc = acc
            best_global_model_name = model_name
            best_global_selector_name = sel_name
            best_global_pred = y_pred

# =============================================================================
# 4. RAPORLAMA
# =============================================================================
print("\n" + "=" * 80)
print(f"{'ULTIMATE KARŞILAŞTIRMA TABLOSU':^80}")
print("=" * 80)

df_res = pd.DataFrame(results_log)
df_res = df_res.sort_values(by='Accuracy', ascending=False)

print(df_res.to_string(index=False))

print("-" * 80)
print(f" >>> ŞAMPİYON: {best_global_selector_name} + {best_global_model_name}")
print(f" >>> SKOR: %{best_global_acc * 100:.2f}")
print("-" * 80)

print(f"\nŞAMPİYON MODELİN DETAYLI RAPORU:\n")
print(classification_report(y_test, best_global_pred, target_names=TARGET_NAMES))

plt.figure(figsize=(10, 7))
cm = confusion_matrix(y_test, best_global_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=TARGET_NAMES, yticklabels=TARGET_NAMES)
plt.title(f'KAZANAN: {best_global_selector_name} + {best_global_model_name}\n(Accuracy: %{best_global_acc * 100:.1f})')
plt.ylabel('Gerçek')
plt.xlabel('Tahmin')
plt.show()

# Tüm Sonuçları Gösteren Isı Haritası (Hocanın en çok seveceği grafik)
try:
    pivot_table = df_res.pivot(index='Selection Method', columns='Model', values='Accuracy')
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_table, annot=True, cmap='RdYlGn', fmt=".2f", linewidths=.5)
    plt.title("Hangi Yöntem Hangi Model ile En İyi Çalıştı?")
    plt.ylabel("Öznitelik Seçimi Yöntemi")
    plt.xlabel("Yapay Zeka Modeli")
    plt.show()
except:
    pass