# Ödev-2: Eğitilen Word2Vec Modelleri ile Metin Benzerliği Hesaplama ve Değerlendirme

**Proje Konusu:** Yelp İşletme (Business) Verisi Üzerinde Benzer İşletme Bulma
**Proje Sahipleri:** _(grup üyeleri isim ve numaralarını buraya yazar)_

Bu çalışma, Ödev-1'de Yelp işletme verisinden (her işletmenin `name + categories` metni) oluşturulan iki temiz veri seti (`lemmatized_sentences.csv`, `stemmed_sentences.csv`) üzerinde **16 Word2Vec modeli** eğitir, veri setinden seçilen örnek bir işletmeye en benzer 5 işletmeyi her model için bulur ve modelleri **Cosine**, **Anlamsal (1–5)** ve **Jaccard** olmak üzere üç yöntemle karşılaştırmalı değerlendirir.

## Klasör Yapısı
```
odev2_yelp/
├── 00_make_sample_data.py    # Yelp şemasına uygun temsili işletme JSON'u üretir
├── 01_preprocess.py          # JSON -> lemmatized_sentences.csv + stemmed_sentences.csv (NLTK)
├── train_and_evaluate.py     # Görev-1 + Görev-2 + 3 değerlendirme (ana script)
├── build_report.py           # outputs/ çıktılarından Word/PDF rapor üretir
├── requirements.txt
├── README.md
├── data/                     # JSON kaynağı + iki temiz CSV
├── model/                    # Eğitilen 16 Word2Vec modeli (.model)
├── outputs/                  # Sonuç tabloları (.csv) + jaccard_heatmap.png + query.txt
└── rapor/                    # Odev2_Raporu.docx ve Odev2_Raporu.pdf
```

## Kurulum
Python 3.9+ önerilir.
```bash
pip install -r requirements.txt
```
> İlk çalıştırmada NLTK veri paketleri (`punkt`, `stopwords`, `wordnet`) otomatik indirilir.

## Çalıştırma

**1) Veriyi hazırla** (gerçek `yelp_academic_dataset_business.json` elinizdeyse bu adımı atlayıp
dosyayı `data/` altına koyabilirsiniz; şema aynıdır):
```bash
python 00_make_sample_data.py     # data/yelp_academic_dataset_business.json üretir
python 01_preprocess.py           # iki temiz CSV üretir
```

**2) Modelleri eğit ve tüm değerlendirmeleri üret:**
```bash
python train_and_evaluate.py
```
Bu komut:
- `lemmatized` ve `stemmed` setleri için 8'er = **16 Word2Vec modeli** eğitir → `model/`
- Seçilen anahtar kelimeye (`coffee`) en yakın 5 kelimeyi çıkarır → `outputs/similar_words.csv`
- Örnek bir giriş işletmesi için her modelde en benzer 5 işletmeyi bulur → `outputs/top5_per_model.csv`
- **Cosine** değerlendirme → `outputs/cosine_eval.csv`
- **Anlamsal (1–5)** değerlendirme → `outputs/semantic_eval.csv`
- **16×16 Jaccard matrisi** → `outputs/jaccard_matrix.csv` ve **heatmap** → `outputs/jaccard_heatmap.png`

**3) PDF/Word raporu üret:**
```bash
python build_report.py
```
> PDF dönüşümü `docx2pdf` + MS Word ile (Windows) yapılır. Word yoksa oluşan `.docx`'i elle PDF'e çevirin.
> Rapor kapağındaki isim/numara alanlarını `build_report.py` içindeki `GROUP_MEMBERS` listesinden düzenleyin.

## Yöntem Özeti
- **Vektörleştirme:** Gensim `Word2Vec`, `min_count=2`, `epochs=80`, `workers=1`, `seed=42`. Değişen parametreler: algoritma (CBOW/SkipGram), `window` (2/4), `vector_size` (100/300).
- **Doküman vektörü:** Metindeki kelime vektörlerinin **aritmetik ortalaması**. OOV kelimeler atlanır; hiçbir kelime yoksa **sıfır vektörü** atanır (NaN/ZeroDivision koruması).
- **Benzerlik:** Giriş işletmesi vektörü ile her doküman vektörü arasında **cosine similarity**; en yüksek 5 sonuç seçilir.
- **Jaccard:** İki modelin ilk 5 sonuç kümesinin kesişim/birleşim oranı.

## Modeller
Boyut nedeniyle GitHub'a yüklenemezse modeller Google Drive'a yüklenir ve herkese açık link buraya eklenir.
Drive linki: _(gerekirse buraya eklenecek)_

Model isimleri örneği: `word2vec_lemmatized_cbow_win2_dim100.model`, `word2vec_stemmed_skipgram_win4_dim300.model` …
