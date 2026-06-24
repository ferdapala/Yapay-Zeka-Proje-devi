# -*- coding: utf-8 -*-
"""
ODEV-2 : Word2Vec ile Metin Benzerligi ve Karsilastirmali Degerlendirme
Proje  : Yelp Isletme (Business) Verisi Uzerinde Benzer Isletme Bulma
Veri   : Yelp Academic Dataset - business (name + categories), Ingilizce

Akis:
  Gorev-1  -> lemmatized + stemmed setleri icin 8'er = 16 Word2Vec modeli egitilir
  Gorev-1b -> secilen bir anahtar kelimenin ('coffee') her modeldeki en yakin 5 komsusu
  Gorev-2  -> veri setinden secilen ornek bir isletmeye en benzer 5 isletme (her model)
  Deg-1    -> Cosine (objektif)      -> outputs/cosine_eval.csv
  Deg-2    -> Anlamsal 1-5 (kategori ortusmesi tabanli) -> outputs/semantic_eval.csv
  Deg-3    -> 16x16 Jaccard matrisi + heatmap            -> outputs/jaccard_*.{csv,png}

Calistirma:  py -3.9 train_and_evaluate.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from gensim.models import Word2Vec

# ----------------------------------------------------------------------------
# Yapilandirma
# ----------------------------------------------------------------------------
SEED = 42
EPOCHS = 80           # Dokumanlar kisa (isletme adi + kategoriler) oldugu icin epoch yuksek
MIN_COUNT = 2         # Tek seferlik nadir tokenlari kelime hazinesine alma
WORKERS = 1           # seed ile birlikte tekrarlanabilirlik

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "data")
MODEL_DIR = os.path.join(BASE, "model")
OUT_DIR = os.path.join(BASE, "outputs")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# Yonergede verilen 8 parametre seti (her veri seti icin)
PARAM_GRID = [
    {"algo": "cbow",     "window": 2, "dim": 100},
    {"algo": "skipgram", "window": 2, "dim": 100},
    {"algo": "cbow",     "window": 4, "dim": 100},
    {"algo": "skipgram", "window": 4, "dim": 100},
    {"algo": "cbow",     "window": 2, "dim": 300},
    {"algo": "skipgram", "window": 2, "dim": 300},
    {"algo": "cbow",     "window": 4, "dim": 300},
    {"algo": "skipgram", "window": 4, "dim": 300},
]

DATASETS = {
    "lemmatized": "lemmatized_sentences.csv",
    "stemmed":    "stemmed_sentences.csv",
}

DEMO_KEYWORD = "coffee"   # Gorev-1b: en yakin 5 kelime demosu icin secilen anlamli kelime
QUERY_BID = "b1010"       # Gorev-2: giris isletmesi (veri setinden, bir kahve/cafe isletmesi)


# ----------------------------------------------------------------------------
# Yardimci fonksiyonlar
# ----------------------------------------------------------------------------
def tokens_of(text):
    return [w for w in str(text).split() if w.strip()]


def model_id(dataset, p):
    return "word2vec_{}_{}_win{}_dim{}".format(dataset, p["algo"], p["window"], p["dim"])


def label_of(row):
    """Sonuc tablolarinda isletmeyi okunur gosterecek etiket."""
    return "{} - {} ({})".format(row["name"], row["categories"], row["city"])


# ----------------------------------------------------------------------------
# Ana isleyici sinif
# ----------------------------------------------------------------------------
class SimilarityStudy:
    def __init__(self):
        self.frames = {}        # dataset -> DataFrame
        self.sentences = {}     # dataset -> list[list[str]]
        self.models = {}        # model_id -> Word2Vec
        self.order = []         # [(dataset, model_id), ...]  (rapor/jaccard ekseni)
        self.query_idx = {}     # dataset -> giris isletmesinin satir indeksi

    # ---- veri yukleme -------------------------------------------------------
    def load(self):
        for ds, fname in DATASETS.items():
            df = pd.read_csv(os.path.join(DATA_DIR, fname), sep=";")
            df["text"] = df["text"].fillna("").astype(str)
            self.frames[ds] = df.reset_index(drop=True)
            self.sentences[ds] = [tokens_of(t) for t in df["text"].tolist()]
            # giris isletmesinin indeksini business_id'den bul
            match = df.index[df["business_id"] == QUERY_BID].tolist()
            self.query_idx[ds] = match[0] if match else 0
        print("Veri yuklendi. Isletme sayisi:", len(self.frames["lemmatized"]))

    # ---- Gorev-1: 16 modeli egit ------------------------------------------
    def train(self):
        for ds in DATASETS:
            for p in PARAM_GRID:
                m = Word2Vec(
                    sentences=self.sentences[ds],
                    vector_size=p["dim"],
                    window=p["window"],
                    sg=1 if p["algo"] == "skipgram" else 0,
                    min_count=MIN_COUNT,
                    workers=WORKERS,
                    seed=SEED,
                    epochs=EPOCHS,
                )
                mid = model_id(ds, p)
                m.save(os.path.join(MODEL_DIR, mid + ".model"))
                self.models[mid] = m
                self.order.append((ds, mid))
                print("  egitildi:", mid, "| vocab =", len(m.wv.index_to_key))

    # ---- Gorev-1b: anahtar kelimeye en yakin 5 kelime ---------------------
    def keyword_neighbours(self):
        rows = []
        for ds, mid in self.order:
            wv = self.models[mid].wv
            if DEMO_KEYWORD in wv:
                top = wv.most_similar(DEMO_KEYWORD, topn=5)
                txt = "; ".join("{} ({:.3f})".format(w, s) for w, s in top)
            else:
                txt = "[kelime hazinesinde yok]"
            rows.append({"model": mid, "keyword": DEMO_KEYWORD, "top5_neighbours": txt})
        out = pd.DataFrame(rows)
        out.to_csv(os.path.join(OUT_DIR, "similar_words.csv"), index=False, encoding="utf-8-sig")
        return out

    # ---- dokuman vektoru + cosine -----------------------------------------
    @staticmethod
    def doc_vector(model, toks):
        """Modelde KARSILIGI OLAN tokenlarin vektor ortalamasi.
        Hicbir token modelde yoksa SIFIR VEKTOR doner (NaN/ZeroDivision korumasi)."""
        vecs = [model.wv[w] for w in toks if w in model.wv]
        if not vecs:
            return np.zeros(model.vector_size, dtype=np.float32)
        return np.mean(vecs, axis=0)

    @staticmethod
    def cosine(a, b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0.0 or nb == 0.0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def top5(self, mid, ds, topn=5):
        df = self.frames[ds]
        model = self.models[mid]
        qi = self.query_idx[ds]
        qvec = self.doc_vector(model, tokens_of(df.iloc[qi]["text"]))

        scores = []
        for i in range(len(df)):
            if i == qi:
                continue
            dvec = self.doc_vector(model, tokens_of(df.iloc[i]["text"]))
            scores.append((i, self.cosine(qvec, dvec)))
        scores.sort(key=lambda t: t[1], reverse=True)
        return scores[:topn]

    # ---- Anlamsal puan (1-5): kategori/tema ortusmesi ----------------------
    @staticmethod
    def semantic_score(query_toks, cand_toks):
        """Giris isletmesi ile aday isletmenin tema (kategori) tokenlarinin
        ortusme miktarina gore 1-5 arasi anlamsal yakinlik.
        Bu kural-tabanli eslestirme, elle puanlamanin tekrarlanabilir bir
        yansimasidir (insan mantigi: ne kadar cok ortak kategori, o kadar yakin)."""
        shared = len(set(query_toks) & set(cand_toks))
        if shared >= 4:
            return 5
        if shared == 3:
            return 4
        if shared == 2:
            return 3
        if shared == 1:
            return 2
        return 1

    # ---- Gorev-2 + Deg-1 + Deg-2 ------------------------------------------
    def run_similarity(self):
        # giris isletmesini dosyaya yaz (her iki set icin ayni isletme)
        lem = self.frames["lemmatized"]
        qi = self.query_idx["lemmatized"]
        qrow = lem.iloc[qi]
        with open(os.path.join(OUT_DIR, "query.txt"), "w", encoding="utf-8") as f:
            f.write("Giris isletmesi (business_id={}):\n".format(qrow["business_id"]))
            f.write("  Ad        : {}\n".format(qrow["name"]))
            f.write("  Kategoriler: {}\n".format(qrow["categories"]))
            f.write("  Sehir/Yildiz: {} / {}\n".format(qrow["city"], qrow["stars"]))
            f.write("  Lemmatized : {}\n".format(qrow["text"]))
            f.write("  Stemmed    : {}\n".format(
                self.frames["stemmed"].iloc[self.query_idx["stemmed"]]["text"]))
        print("Giris isletmesi:", label_of(qrow))

        top5_rows, cosine_rows, semantic_rows = [], [], []
        self.top5_sets = {}

        for ds, mid in self.order:
            df = self.frames[ds]
            qi = self.query_idx[ds]
            qtoks = tokens_of(df.iloc[qi]["text"])
            res = self.top5(mid, ds)
            self.top5_sets[mid] = set(i for i, _ in res)

            cos_list, sem_list = [], []
            for rank, (i, sc) in enumerate(res, start=1):
                row = df.iloc[i]
                sem = self.semantic_score(qtoks, tokens_of(row["text"]))
                cos_list.append(round(sc, 4))
                sem_list.append(sem)
                top5_rows.append({
                    "model": mid, "rank": rank, "business_id": row["business_id"],
                    "cosine": round(sc, 4), "semantic_1_5": sem,
                    "business": label_of(row),
                })
            cosine_rows.append({
                "model": mid,
                "s1": cos_list[0], "s2": cos_list[1], "s3": cos_list[2],
                "s4": cos_list[3], "s5": cos_list[4],
                "mean_cosine": round(float(np.mean(cos_list)), 4),
            })
            semantic_rows.append({
                "model": mid,
                "p1": sem_list[0], "p2": sem_list[1], "p3": sem_list[2],
                "p4": sem_list[3], "p5": sem_list[4],
                "mean_semantic": round(float(np.mean(sem_list)), 2),
            })

        pd.DataFrame(top5_rows).to_csv(
            os.path.join(OUT_DIR, "top5_per_model.csv"), index=False, encoding="utf-8-sig")
        self.cos_df = pd.DataFrame(cosine_rows)
        self.cos_df.to_csv(os.path.join(OUT_DIR, "cosine_eval.csv"), index=False, encoding="utf-8-sig")
        self.sem_df = pd.DataFrame(semantic_rows)
        self.sem_df.to_csv(os.path.join(OUT_DIR, "semantic_eval.csv"), index=False, encoding="utf-8-sig")

    # ---- Deg-3: Jaccard matrisi + heatmap ---------------------------------
    @staticmethod
    def jaccard(a, b):
        if not a and not b:
            return 1.0
        u = len(a | b)
        return len(a & b) / u if u else 0.0

    def ranking_agreement(self):
        ids = [mid for _, mid in self.order]
        short = [m.replace("word2vec_", "") for m in ids]
        n = len(ids)
        J = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                J[i, j] = self.jaccard(self.top5_sets[ids[i]], self.top5_sets[ids[j]])
        jdf = pd.DataFrame(J, index=short, columns=short)
        jdf.round(3).to_csv(os.path.join(OUT_DIR, "jaccard_matrix.csv"), encoding="utf-8-sig")

        plt.figure(figsize=(14, 11))
        sns.heatmap(jdf, annot=True, fmt=".2f", cmap="magma",
                    cbar_kws={"label": "Jaccard benzerligi"}, annot_kws={"size": 7})
        plt.title("16 Word2Vec Modeli - Top-5 Sonuc Ortusmesi (Jaccard)")
        plt.xticks(rotation=90, fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "jaccard_heatmap.png"), dpi=150)
        plt.close()
        return jdf

    # ---- Ozet --------------------------------------------------------------
    def summarize(self):
        s = self.cos_df[["model", "mean_cosine"]].merge(
            self.sem_df[["model", "mean_semantic"]], on="model")
        s.to_csv(os.path.join(OUT_DIR, "summary.csv"), index=False, encoding="utf-8-sig")
        print("\n=== OZET (model basina ortalamalar) ===")
        print(s.to_string(index=False))


def main():
    study = SimilarityStudy()
    print(">> Veri yukleniyor ...")
    study.load()
    print(">> Gorev-1: 16 Word2Vec modeli egitiliyor ...")
    study.train()
    print(">> Gorev-1b: '{}' kelimesine en yakin 5 kelime ...".format(DEMO_KEYWORD))
    study.keyword_neighbours()
    print(">> Gorev-2: her model icin en benzer 5 isletme ...")
    study.run_similarity()
    print(">> Deg-3: 16x16 Jaccard matrisi + heatmap ...")
    study.ranking_agreement()
    study.summarize()
    print("\nTum ciktilar 'outputs/' klasorunde, modeller 'model/' klasorunde.")


if __name__ == "__main__":
    main()
