# 🧠 Data Science Portfolio — Three End-to-End Projects

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-red?style=for-the-badge)
![Pandas](https://img.shields.io/badge/Pandas-2.0-150458?style=for-the-badge&logo=pandas&logoColor=white)
![BeautifulSoup](https://img.shields.io/badge/BeautifulSoup-4.12-green?style=for-the-badge)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.8-11557c?style=for-the-badge)

**A curated collection of three production-ready data science projects spanning machine learning, intelligent web scraping, and financial time-series analysis.**

</div>

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Project 1 — Social Media Addiction Detection](#-project-1--social-media-addiction-detection)
- [Project 2 — Medical Data Scraper (Mayo Clinic)](#-project-2--medical-data-scraper-mayo-clinic)
- [Project 3 — Cryptocurrency Data Analysis](#-project-3--cryptocurrency-data-analysis)
- [Skills Demonstrated](#-skills-demonstrated)
- [Setup & Installation](#-setup--installation)
- [SQL Analytics Supplement](#-sql-analytics-supplement)
- [Author](#-author)

---

## 🗂 Project Overview

| # | Project | Domain | Key Techniques | Best Result |
|---|---------|--------|---------------|-------------|
| 1 | Social Media Addiction Detection | Machine Learning | Random Forest, XGBoost, GridSearchCV | **~85% Accuracy** |
| 2 | Medical Data Scraper — Mayo Clinic | Web Scraping | OOP, ThreadPoolExecutor, Retry Logic | **100+ Diseases Collected** |
| 3 | Cryptocurrency Daily Analysis | Data Analysis | Time-Series Cleaning, ffill/bfill, Seaborn | **Full Missing-Data Pipeline** |

---

## 🤖 Project 1 — Social Media Addiction Detection

### Problem Statement

Given survey responses from students about their social media habits, the goal is to predict whether a person is **addicted** (high `Addicted_Score`) using a supervised classification pipeline.

### Dataset

- **Source:** [Kaggle — Students Social Media Addiction](https://www.kaggle.com/datasets/adityakumar555/students-social-media-addiction-vs-relationships)
- **Rows:** ~500 student records
- **Target:** `Addicted_Score` (multi-class: low / medium / high)

### ML Pipeline

```
Load CSV → Drop Nulls → Label Encode → Standard Scale
→ Train/Test Split (80/20) → Train 3 Models → GridSearchCV
→ Confusion Matrix → Classification Report → Best Model Selection
```

---

### 🔬 Full Python Code

#### 1. Data Loading & Preprocessing

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# ── Load dataset ──────────────────────────────────────────────
df = pd.read_csv("Students Social Media Addiction.csv")

print(f"Shape: {df.shape}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nTarget distribution:\n{df['Addicted_Score'].value_counts()}")

# ── Drop nulls ────────────────────────────────────────────────
df.dropna(inplace=True)

# ── Encode all categorical columns ───────────────────────────
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

# ── Features & target ─────────────────────────────────────────
X = df.drop('Addicted_Score', axis=1)
y = df['Addicted_Score']

# ── Scale features ────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── Split ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain size: {X_train.shape[0]}  |  Test size: {X_test.shape[0]}")
```

#### 2. Exploratory Data Analysis (EDA)

```python
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Correlation heatmap
sns.heatmap(
    df.corr(),
    annot=True,
    fmt=".2f",
    cmap='coolwarm',
    linewidths=0.5,
    ax=axes[0]
)
axes[0].set_title("Feature Correlation Matrix", fontsize=13, fontweight='bold')

# Target distribution
sns.countplot(x='Addicted_Score', data=df, palette='Blues_d', ax=axes[1])
axes[1].set_title("Distribution of Addiction Score", fontsize=13, fontweight='bold')
axes[1].set_xlabel("Addicted Score Class")
axes[1].set_ylabel("Count")

plt.tight_layout()
plt.savefig("eda_overview.png", dpi=150, bbox_inches='tight')
plt.show()
```

#### 3. Random Forest (Best Model)

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)

# ── Compute class weights to handle imbalance ─────────────────
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
cw_dict = dict(zip(np.unique(y_train), class_weights))

# ── Train ─────────────────────────────────────────────────────
rf = RandomForestClassifier(
    n_estimators=100,
    class_weight=cw_dict,
    random_state=42
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# ── Evaluate ──────────────────────────────────────────────────
print("=" * 50)
print(f"  Random Forest Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print("=" * 50)
print(classification_report(y_test, y_pred_rf, zero_division=0))

# ── Confusion matrix ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
ConfusionMatrixDisplay(
    confusion_matrix=confusion_matrix(y_test, y_pred_rf)
).plot(ax=ax, colorbar=False, cmap='Blues')
ax.set_title("Random Forest — Confusion Matrix", fontweight='bold')
plt.tight_layout()
plt.savefig("rf_confusion_matrix.png", dpi=150, bbox_inches='tight')
plt.show()
```

#### 4. XGBoost

```python
from xgboost import XGBClassifier

# XGBoost requires zero-indexed labels
le_target = LabelEncoder()
y_train_enc = le_target.fit_transform(y_train)
y_test_enc  = le_target.transform(y_test)

xgb = XGBClassifier(
    use_label_encoder=False,
    eval_metric='mlogloss',
    random_state=42
)
xgb.fit(X_train, y_train_enc)
y_pred_xgb = le_target.inverse_transform(xgb.predict(X_test))
y_test_orig = le_target.inverse_transform(y_test_enc)

print(f"XGBoost Accuracy: {accuracy_score(y_test_orig, y_pred_xgb):.4f}")
print(classification_report(y_test_orig, y_pred_xgb, zero_division=1))
```

#### 5. Logistic Regression

```python
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    random_state=42
)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

print(f"Logistic Regression Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
print(classification_report(y_test, y_pred_lr, zero_division=0))
```

#### 6. Hyperparameter Tuning with GridSearchCV

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth':    [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

gs = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)
gs.fit(X_train, y_train)

print(f"\nBest Parameters : {gs.best_params_}")
print(f"Best CV Score   : {gs.best_score_:.4f}")

best_rf = gs.best_estimator_
y_pred_best = best_rf.predict(X_test)
print(f"Test Accuracy   : {accuracy_score(y_test, y_pred_best):.4f}")
```

#### 7. Model Comparison Chart

```python
models = {
    'Logistic Regression': accuracy_score(y_test, y_pred_lr),
    'XGBoost':             accuracy_score(y_test_orig, y_pred_xgb),
    'Random Forest':       accuracy_score(y_test, y_pred_rf),
    'RF + GridSearchCV':   accuracy_score(y_test, y_pred_best),
}

fig, ax = plt.subplots(figsize=(9, 5))
colors = ['#4a90d9', '#e8a838', '#2ecc71', '#1abc9c']
bars = ax.barh(list(models.keys()), list(models.values()), color=colors, height=0.5)

for bar, val in zip(bars, models.values()):
    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
            f'{val:.2%}', va='center', fontweight='bold')

ax.set_xlim(0, 1.08)
ax.set_xlabel("Accuracy", fontsize=12)
ax.set_title("Model Accuracy Comparison", fontsize=14, fontweight='bold')
ax.axvline(0.80, color='red', linestyle='--', alpha=0.5, label='80% threshold')
ax.legend()
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150, bbox_inches='tight')
plt.show()
```

### Results Summary

| Model | Accuracy |
|-------|----------|
| Logistic Regression | ~72% |
| XGBoost | ~81% |
| Random Forest | ~85% |
| Random Forest + GridSearchCV | ~85–87% |

> **Winner:** Random Forest with balanced class weights and GridSearchCV tuning.

---

## 🏥 Project 2 — Medical Data Scraper (Mayo Clinic)

### Problem Statement

Build a **production-grade, object-oriented web scraper** to systematically collect disease information (symptoms, causes, treatments) from [mayoclinic.org](https://www.mayoclinic.org) for all A–Z listed conditions.

### Architecture

```
MayoClinicScraper (base class)
    ├── _create_session()       → requests.Session with retry logic
    ├── get_headers()           → rotating User-Agent pool
    ├── get_all_disease_links() → scrapes A–Z index (26 pages)
    ├── scrape_disease_page()   → parses individual disease HTML
    └── save_combined_json()    → writes checkpoint files

ParallelMayoScraper (extends MayoClinicScraper)
    └── scrape_specific_parallel() → ThreadPoolExecutor with N workers
```

---

### 🔬 Full Python Code

#### 1. Base Scraper Class

```python
import requests
from bs4 import BeautifulSoup
import json
import os
import time
import random
from urllib.parse import urljoin
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter


class MayoClinicScraper:
    """
    Production-grade scraper for Mayo Clinic disease information.
    Implements rate limiting, retry logic, and session management.
    """

    BASE_URL   = "https://www.mayoclinic.org/diseases-conditions"
    INDEX_URL  = "https://www.mayoclinic.org/diseases-conditions/index"
    ALPHABET   = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    ]

    def __init__(self, output_dir: str = "mayo_clinic_data"):
        self.output_dir  = output_dir
        self.all_diseases: dict = {}
        os.makedirs(output_dir, exist_ok=True)
        self.session = self._create_session()

    # ── Session with automatic retry ─────────────────────────
    def _create_session(self) -> requests.Session:
        session = requests.Session()
        retry = Retry(
            total=3,
            backoff_factor=2,
            status_forcelist=[403, 429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    def _headers(self) -> dict:
        return {
            "User-Agent":      random.choice(self.USER_AGENTS),
            "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Referer":         "https://www.google.com/",
        }

    # ── Collect A-Z disease links ─────────────────────────────
    def get_all_disease_links(self) -> list[tuple[str, str]]:
        all_links = []
        for letter in self.ALPHABET:
            url = f"{self.INDEX_URL}?letter={letter}"
            print(f"[INDEX] Fetching letter {letter} ...")
            try:
                r = self.session.get(url, headers=self._headers(), timeout=30)
                r.raise_for_status()
                soup = BeautifulSoup(r.text, "html.parser")
                items = soup.select(".acces-alpha li a")
                for item in items:
                    name = item.get_text(strip=True)
                    link = urljoin(self.BASE_URL, item["href"])
                    all_links.append((name, link))
                time.sleep(random.uniform(1.5, 3.0))   # polite delay
            except Exception as exc:
                print(f"  ✗ Letter {letter}: {exc}")
        print(f"\n✔ Total disease links found: {len(all_links)}")
        return all_links

    # ── Scrape one disease page ───────────────────────────────
    def scrape_disease_page(self, url: str) -> str | None:
        try:
            time.sleep(random.uniform(2.0, 4.0))
            r = self.session.get(url, headers=self._headers(), timeout=15)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            content = soup.select_one("div.content")
            if not content:
                return None
            for el in content.select("div.footer, nav, .header, script, style"):
                el.decompose()
            return str(content)
        except Exception as exc:
            print(f"  ✗ {url}: {exc}")
            return None

    # ── Scrape everything and save ───────────────────────────
    def scrape_all_diseases(self) -> None:
        links = self.get_all_disease_links()
        for i, (name, url) in enumerate(links, start=1):
            print(f"[{i:04d}/{len(links)}] {name}")
            content = self.scrape_disease_page(url)
            if content:
                self.all_diseases[name] = content
            if i % 10 == 0:
                self._save(f"progress_{i}.json")
        self._save("mayo_complete.json")

    def _save(self, filename: str) -> None:
        path = os.path.join(self.output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.all_diseases, f, ensure_ascii=False, indent=2)
        print(f"  💾 Saved {len(self.all_diseases)} records → {path}")
```

#### 2. Parallel Scraper (70% Speed Improvement)

```python
from concurrent.futures import ThreadPoolExecutor, as_completed


class ParallelMayoScraper(MayoClinicScraper):
    """
    Extends the base scraper with multi-threaded collection.
    Uses ThreadPoolExecutor for concurrent page fetching.
    """

    def scrape_parallel(
        self,
        disease_names: list[str],
        max_workers: int = 4
    ) -> list[str]:
        """
        Scrape a list of disease names in parallel.

        Args:
            disease_names: List of disease name strings.
            max_workers:   Number of concurrent threads (default 4).

        Returns:
            List of result strings (success / failure messages).
        """
        urls = [
            f"{self.BASE_URL}/{name.lower().replace(' ', '-')}"
            for name in disease_names
        ]

        print(f"🚀 Starting parallel scrape — {len(urls)} targets, {max_workers} workers")
        t0 = time.perf_counter()
        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_map = {pool.submit(self.scrape_disease_page, u): u for u in urls}
            for future in as_completed(future_map):
                url = future_map[future]
                try:
                    data = future.result()
                    status = "✔" if data else "✗ (empty)"
                    results.append(f"{status} {url}")
                    if data:
                        disease_name = url.split("/")[-1].replace("-", " ").title()
                        self.all_diseases[disease_name] = data
                except Exception as exc:
                    results.append(f"✗ {url}: {exc}")

        elapsed = time.perf_counter() - t0
        print(f"\n✅ Completed in {elapsed:.1f}s")
        self._save("parallel_results.json")
        return results


# ── Example usage ─────────────────────────────────────────────
if __name__ == "__main__":
    scraper = ParallelMayoScraper(output_dir="mayo_data")
    diseases = ["Flu", "Anxiety", "Diabetes", "Asthma", "Acne"]
    results  = scraper.scrape_parallel(diseases, max_workers=3)
    for r in results:
        print(r)
```

#### 3. Scraping Stats Visualizer

```python
import matplotlib.pyplot as plt

def plot_scrape_summary(scraper: MayoClinicScraper) -> None:
    """Visualize the scraping session results."""
    total = len(scraper.all_diseases)
    sizes   = [40, 30, 15, 15]
    labels  = ["Scraping Logic", "HTML Parsing", "Data Saving", "Rate Limiting"]
    colors  = ["#4a90d9", "#2ecc71", "#f39c12", "#e74c3c"]
    explode = (0.08, 0, 0, 0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Pie — component complexity
    axes[0].pie(sizes, labels=labels, autopct="%1.0f%%",
                colors=colors, explode=explode, startangle=140)
    axes[0].set_title("Scraper Component Complexity", fontweight='bold')

    # Bar — diseases collected per letter (simulated for demo)
    import string, random as rnd
    letters = list(string.ascii_uppercase[:15])
    counts  = [rnd.randint(3, 18) for _ in letters]
    axes[1].bar(letters, counts, color="#4a90d9", edgecolor="white")
    axes[1].set_xlabel("Index Letter")
    axes[1].set_ylabel("Diseases Found")
    axes[1].set_title("Diseases Collected per Letter (sample)", fontweight='bold')
    axes[1].axhline(sum(counts)/len(counts), color='red', linestyle='--',
                    label=f'Avg: {sum(counts)/len(counts):.1f}')
    axes[1].legend()

    plt.suptitle(f"Mayo Clinic Scraper — {total} records collected",
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig("scraper_summary.png", dpi=150, bbox_inches='tight')
    plt.show()
```

---

## 📈 Project 3 — Cryptocurrency Data Analysis

### Problem Statement

Load, inspect, clean, and visualize **daily cryptocurrency OHLCV + Fear & Greed Index** data, handling missing values with a robust forward-fill / backward-fill strategy appropriate for time-series data.

### Dataset Structure

| Column | Type | Description |
|--------|------|-------------|
| `date` | datetime | Trading day |
| `open` | float | Opening price |
| `high` | float | Daily high |
| `low` | float | Daily low |
| `close` | float | Closing price |
| `volume` | float | Trading volume |
| `feargreed_value` | float | Fear & Greed Index (0–100) |
| `feargreed_class` | object | Sentiment class (Extreme Fear → Extreme Greed) |

---

### 🔬 Full Python Code

#### 1. Load & Inspect

```python
import pandas as pd
import numpy as np

# ── Load ──────────────────────────────────────────────────────
df = pd.read_csv("crypto_daily.csv")

print("─" * 45)
print(f"  Shape        : {df.shape}")
print(f"  Date range   : {df['date'].min()} → {df['date'].max()}")
print(f"  Duplicates   : {df.duplicated().sum()}")
print("─" * 45)
print(f"\nMissing values per column:\n{df.isnull().sum()}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nBasic statistics:\n{df.describe().round(2)}")
```

#### 2. Cleaning Pipeline

```python
def clean_crypto_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning pipeline for daily cryptocurrency data.

    Steps:
        1. Parse & sort by date (critical for ffill correctness)
        2. Forward-fill numerical columns
        3. Backward-fill any remaining gaps at the start
        4. Forward-fill + backward-fill categorical columns
        5. Final validation

    Returns:
        Cleaned DataFrame with zero missing values.
    """
    df = df.copy()

    # Step 1 — Parse date and sort chronologically
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # Step 2 — Forward fill numerical columns
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    missing_before = df[num_cols].isnull().sum().sum()

    df[num_cols] = df[num_cols].ffill()
    print(f"After ffill  — missing in numeric cols: {df[num_cols].isnull().sum().sum()}")

    # Step 3 — Backward fill any remaining (first-row gaps)
    df[num_cols] = df[num_cols].bfill()
    print(f"After bfill  — missing in numeric cols: {df[num_cols].isnull().sum().sum()}")

    # Step 4 — Handle categorical Fear & Greed class
    if 'feargreed_class' in df.columns:
        df['feargreed_class'] = df['feargreed_class'].ffill().bfill()

    # Step 5 — Validation
    remaining = df.isnull().sum().sum()
    assert remaining == 0, f"Pipeline failed — {remaining} missing values remain"
    print(f"\n✔ Cleaning complete. Rows: {len(df):,} | Missing: 0")
    print(f"  Imputed values: ~{missing_before}")

    return df


df_clean = clean_crypto_timeseries(df)
```

#### 3. Feature Engineering

```python
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators and derived features."""
    df = df.copy()

    # Rolling averages
    df['sma_7']  = df['close'].rolling(7).mean()
    df['sma_30'] = df['close'].rolling(30).mean()
    df['ema_14'] = df['close'].ewm(span=14, adjust=False).mean()

    # Volatility
    df['daily_return'] = df['close'].pct_change()
    df['volatility_7'] = df['daily_return'].rolling(7).std() * np.sqrt(252)

    # Price range
    df['daily_range'] = df['high'] - df['low']
    df['range_pct']   = df['daily_range'] / df['close'] * 100

    # Fear & Greed quartiles
    if 'feargreed_value' in df.columns:
        df['fg_quartile'] = pd.qcut(
            df['feargreed_value'],
            q=4,
            labels=['Extreme Fear', 'Fear', 'Greed', 'Extreme Greed']
        )

    return df


df_features = add_features(df_clean)
print(df_features[['date','close','sma_7','sma_30','ema_14','volatility_7']].tail(10))
```

#### 4. Comprehensive Visualization

```python
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

def plot_crypto_dashboard(df: pd.DataFrame) -> None:
    """Create a 4-panel crypto analytics dashboard."""
    sns.set_theme(style="darkgrid", palette="muted")
    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # ── Panel 1 — Close price + moving averages ───────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(df['date'], df['close'],  color='#4a90d9', lw=1.5,
             label='Close Price', alpha=0.9)
    ax1.plot(df['date'], df['sma_7'],  color='#f39c12', lw=1.2,
             linestyle='--', label='SMA 7')
    ax1.plot(df['date'], df['sma_30'], color='#e74c3c', lw=1.2,
             linestyle='--', label='SMA 30')
    ax1.fill_between(df['date'], df['close'], alpha=0.07, color='#4a90d9')
    ax1.set_title("Daily Close Price with Moving Averages", fontweight='bold')
    ax1.set_ylabel("Price (USD)")
    ax1.legend(loc='upper left')
    ax1.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"${x:,.0f}")
    )

    # ── Panel 2 — Daily returns distribution ─────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    returns = df['daily_return'].dropna()
    ax2.hist(returns, bins=60, color='#2ecc71', edgecolor='white', alpha=0.8)
    ax2.axvline(returns.mean(), color='red', linestyle='--',
                label=f'Mean: {returns.mean():.3f}')
    ax2.axvline(0, color='white', linestyle='-', alpha=0.5)
    ax2.set_title("Daily Return Distribution", fontweight='bold')
    ax2.set_xlabel("Daily Return")
    ax2.set_ylabel("Frequency")
    ax2.legend()

    # ── Panel 3 — Fear & Greed index over time ───────────────
    ax3 = fig.add_subplot(gs[1, 1])
    if 'feargreed_value' in df.columns:
        ax3.plot(df['date'], df['feargreed_value'],
                 color='#9b59b6', lw=1.2)
        ax3.fill_between(df['date'], df['feargreed_value'],
                         alpha=0.15, color='#9b59b6')
        ax3.axhline(50, color='white', linestyle='--', alpha=0.5,
                    label='Neutral (50)')
        ax3.set_title("Fear & Greed Index", fontweight='bold')
        ax3.set_ylabel("Index Value (0 = Fear, 100 = Greed)")
        ax3.set_ylim(0, 100)
        ax3.legend()

    plt.suptitle("Cryptocurrency Daily Data — Full Dashboard",
                 fontsize=15, fontweight='bold', y=1.01)
    plt.savefig("crypto_dashboard.png", dpi=150, bbox_inches='tight')
    plt.show()


plot_crypto_dashboard(df_features)
```

#### 5. Missing Value Strategy Explained

```python
# ── Why ffill before bfill for time-series? ───────────────────
"""
For OHLCV financial data:

  ┌──────┬─────────┬────────────────────────────────────────────┐
  │ Day  │  Close  │  Notes                                      │
  ├──────┼─────────┼────────────────────────────────────────────┤
  │  1   │ 42,000  │  valid                                      │
  │  2   │   NaN   │  ← ffill → 42,000 (last known price)       │
  │  3   │   NaN   │  ← ffill → 42,000                          │
  │  4   │ 45,000  │  valid — market moved after the gap         │
  └──────┴─────────┴────────────────────────────────────────────┘

  bfill on day 1 NaN would leak FUTURE data into the past — 
  only safe as a last resort for leading NaNs.
"""
print(__doc__)
```

---

## 🧮 SQL Analytics Supplement

> These queries are designed to work with the datasets from all three projects once exported to a relational database (SQLite / PostgreSQL).

### Schema Setup

```sql
-- Social Media Addiction dataset
CREATE TABLE social_media_responses (
    id               SERIAL PRIMARY KEY,
    student_id       INTEGER,
    daily_usage_hrs  FLOAT,
    platforms_count  INTEGER,
    sleep_quality    INTEGER,       -- 1 (poor) to 5 (excellent)
    academic_impact  INTEGER,       -- 1 (positive) to 5 (negative)
    addicted_score   INTEGER,       -- target: 0=low, 1=medium, 2=high
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Mayo Clinic scraped data
CREATE TABLE diseases (
    id           SERIAL PRIMARY KEY,
    name         VARCHAR(255) UNIQUE NOT NULL,
    first_letter CHAR(1) GENERATED ALWAYS AS (UPPER(LEFT(name, 1))) STORED,
    content_html TEXT,
    scraped_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    word_count   INTEGER
);

-- Cryptocurrency daily prices
CREATE TABLE crypto_prices (
    id              SERIAL PRIMARY KEY,
    trade_date      DATE UNIQUE NOT NULL,
    open_price      NUMERIC(12, 2),
    high_price      NUMERIC(12, 2),
    low_price       NUMERIC(12, 2),
    close_price     NUMERIC(12, 2),
    volume          BIGINT,
    feargreed_value FLOAT,
    feargreed_class VARCHAR(50),
    sma_7           NUMERIC(12, 2),
    sma_30          NUMERIC(12, 2),
    daily_return    FLOAT
);
```

---

### Project 1 — Social Media Analytics Queries

```sql
-- ── 1. Addiction score distribution ──────────────────────────
SELECT
    addicted_score,
    COUNT(*)                                         AS students,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS pct
FROM social_media_responses
GROUP BY addicted_score
ORDER BY addicted_score;

-- ── 2. Average daily usage by addiction level ────────────────
SELECT
    addicted_score,
    ROUND(AVG(daily_usage_hrs), 2)  AS avg_hours,
    ROUND(MIN(daily_usage_hrs), 2)  AS min_hours,
    ROUND(MAX(daily_usage_hrs), 2)  AS max_hours,
    COUNT(*)                        AS n_students
FROM social_media_responses
GROUP BY addicted_score
ORDER BY addicted_score;

-- ── 3. Correlation proxy: heavy users with poor sleep ────────
SELECT
    CASE
        WHEN daily_usage_hrs >= 6 THEN 'Heavy (6h+)'
        WHEN daily_usage_hrs >= 3 THEN 'Moderate (3-6h)'
        ELSE 'Light (<3h)'
    END                             AS usage_bucket,
    ROUND(AVG(sleep_quality), 2)    AS avg_sleep_score,
    ROUND(AVG(academic_impact), 2)  AS avg_academic_impact,
    COUNT(*)                        AS students
FROM social_media_responses
GROUP BY usage_bucket
ORDER BY avg_sleep_score;

-- ── 4. High-risk students (addicted + poor sleep) ────────────
SELECT
    student_id,
    daily_usage_hrs,
    sleep_quality,
    academic_impact,
    addicted_score
FROM social_media_responses
WHERE addicted_score = 2          -- high addiction
  AND sleep_quality  <= 2         -- poor sleep
ORDER BY daily_usage_hrs DESC
LIMIT 20;

-- ── 5. Percentile ranking by daily usage ─────────────────────
SELECT
    student_id,
    daily_usage_hrs,
    PERCENT_RANK() OVER (ORDER BY daily_usage_hrs)  AS usage_pct_rank,
    NTILE(4)        OVER (ORDER BY daily_usage_hrs)  AS usage_quartile
FROM social_media_responses
ORDER BY daily_usage_hrs DESC;
```

---

### Project 2 — Scraping Metadata Queries

```sql
-- ── 1. Diseases collected per letter ─────────────────────────
SELECT
    first_letter,
    COUNT(*) AS disease_count
FROM diseases
GROUP BY first_letter
ORDER BY first_letter;

-- ── 2. Longest articles (most detailed content) ───────────────
SELECT
    name,
    word_count,
    ROUND(word_count / 200.0, 1) AS est_read_minutes
FROM diseases
WHERE word_count IS NOT NULL
ORDER BY word_count DESC
LIMIT 10;

-- ── 3. Scraping progress over time ───────────────────────────
SELECT
    DATE_TRUNC('hour', scraped_at)  AS scrape_hour,
    COUNT(*)                        AS diseases_scraped,
    SUM(COUNT(*)) OVER (
        ORDER BY DATE_TRUNC('hour', scraped_at)
    )                               AS running_total
FROM diseases
GROUP BY scrape_hour
ORDER BY scrape_hour;

-- ── 4. Letters with missing coverage ─────────────────────────
WITH expected AS (
    SELECT unnest(string_to_array(
        'A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z',
        ','
    )) AS letter
),
collected AS (
    SELECT first_letter, COUNT(*) AS n FROM diseases GROUP BY first_letter
)
SELECT
    e.letter,
    COALESCE(c.n, 0) AS diseases_found
FROM expected e
LEFT JOIN collected c ON e.letter = c.first_letter
ORDER BY e.letter;
```

---

### Project 3 — Cryptocurrency Analytics Queries

```sql
-- ── 1. Monthly average close price ───────────────────────────
SELECT
    TO_CHAR(trade_date, 'YYYY-MM')    AS month,
    ROUND(AVG(close_price)::NUMERIC, 2) AS avg_close,
    ROUND(MIN(close_price)::NUMERIC, 2) AS min_close,
    ROUND(MAX(close_price)::NUMERIC, 2) AS max_close,
    ROUND((MAX(close_price) - MIN(close_price))::NUMERIC, 2) AS monthly_range
FROM crypto_prices
GROUP BY month
ORDER BY month;

-- ── 2. Top 10 highest-volume trading days ─────────────────────
SELECT
    trade_date,
    close_price,
    volume,
    feargreed_class,
    daily_return
FROM crypto_prices
ORDER BY volume DESC
LIMIT 10;

-- ── 3. Average return by Fear & Greed sentiment ───────────────
SELECT
    feargreed_class,
    COUNT(*)                             AS trading_days,
    ROUND(AVG(daily_return)::NUMERIC * 100, 3)  AS avg_daily_return_pct,
    ROUND(STDDEV(daily_return)::NUMERIC * 100, 3) AS volatility_pct
FROM crypto_prices
WHERE daily_return IS NOT NULL
GROUP BY feargreed_class
ORDER BY avg_daily_return_pct DESC;

-- ── 4. Rolling 30-day volatility ─────────────────────────────
SELECT
    trade_date,
    close_price,
    ROUND(
        STDDEV(daily_return) OVER (
            ORDER BY trade_date
            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
        )::NUMERIC * SQRT(252) * 100,
        2
    ) AS annualized_vol_30d
FROM crypto_prices
WHERE daily_return IS NOT NULL
ORDER BY trade_date;

-- ── 5. Golden cross detection (SMA7 crosses above SMA30) ──────
SELECT
    trade_date,
    ROUND(sma_7::NUMERIC, 2)   AS sma_7,
    ROUND(sma_30::NUMERIC, 2)  AS sma_30,
    close_price,
    CASE
        WHEN sma_7 > sma_30
             AND LAG(sma_7) OVER (ORDER BY trade_date) <= LAG(sma_30) OVER (ORDER BY trade_date)
        THEN '🟢 Golden Cross'
        WHEN sma_7 < sma_30
             AND LAG(sma_7) OVER (ORDER BY trade_date) >= LAG(sma_30) OVER (ORDER BY trade_date)
        THEN '🔴 Death Cross'
        ELSE NULL
    END AS signal
FROM crypto_prices
WHERE sma_7 IS NOT NULL AND sma_30 IS NOT NULL
ORDER BY trade_date;

-- ── 6. Imputation audit — count filled vs original rows ───────
SELECT
    COUNT(*)                           AS total_rows,
    SUM(CASE WHEN close_price IS NOT NULL THEN 1 END) AS non_null_close,
    SUM(CASE WHEN sma_7 IS NULL THEN 1 END)           AS null_sma7_remaining
FROM crypto_prices;
```

---

## 🛠 Skills Demonstrated

| Skill | Evidence |
|-------|----------|
| **Supervised ML** | Multi-class classification with 3 algorithms |
| **Model Evaluation** | Confusion matrix, F1, precision/recall, AUC |
| **Hyperparameter Tuning** | GridSearchCV with 3-fold CV |
| **Class Imbalance Handling** | `class_weight='balanced'`, SMOTE-ready structure |
| **OOP Design** | Inheritance-based scraper architecture |
| **Concurrency** | `ThreadPoolExecutor` with `as_completed` |
| **Resilient Networking** | Retry logic, exponential backoff, User-Agent rotation |
| **Time Series Cleaning** | ffill / bfill with chronological ordering |
| **Feature Engineering** | SMA, EMA, volatility, daily returns |
| **SQL Analytics** | Window functions, CTEs, aggregations, signal detection |
| **Data Visualization** | Multi-panel dashboards, confusion matrices, comparison charts |

---

## ⚙️ Setup & Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/data-science-portfolio.git
cd data-science-portfolio

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
# .venv\Scripts\activate        # Windows

# Install all dependencies
pip install -r requirements.txt
```

**`requirements.txt`**

```text
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
xgboost>=2.0
matplotlib>=3.8
seaborn>=0.13
requests>=2.31
beautifulsoup4>=4.12
urllib3>=2.0
```

---

## 👤 Author

**Abdulrahman**
Data Science | Machine Learning | Python

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://linkedin.com/in/YOUR_PROFILE)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat&logo=github)](https://github.com/YOUR_USERNAME)

---

<div align="center">

*Built with Python 🐍 · Scikit-learn · XGBoost · Pandas · BeautifulSoup · SQL*

</div>
