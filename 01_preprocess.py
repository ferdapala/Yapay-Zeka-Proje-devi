# -*- coding: utf-8 -*-
"""
01_preprocess.py
----------------
Odev-1 (on isleme) adimini Odev-2 girdisi olacak iki temiz veri setine donusturur.

Kaynak: data/yelp_academic_dataset_business.json  (Yelp business kayitlari)
Her isletme icin metin = name + " " + categories alinir; ardindan:
  - kucuk harfe cevirme
  - kelime tokenizasyonu (NLTK word_tokenize, yoksa regex fallback)
  - sadece harf iceren tokenlar + Ingilizce stop-word temizligi
  - Lemmatization  (WordNetLemmatizer)  -> data/lemmatized_sentences.csv
  - Stemming       (PorterStemmer)      -> data/stemmed_sentences.csv

Cikti CSV'leri (ayirici ';'):
  business_id ; name ; categories ; city ; stars ; <lemmatized|stemmed>
"""
import os
import re
import json
import csv

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "data")
SOURCE_JSON = os.path.join(DATA_DIR, "yelp_academic_dataset_business.json")

# ----------------------------------------------------------------------------
# NLTK kurulu ise onu kullan; degilse hafif bir fallback ile devam et.
# ----------------------------------------------------------------------------
try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer, PorterStemmer

    for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass

    STOP = set(stopwords.words("english"))
    _LEMMATIZER = WordNetLemmatizer()
    _STEMMER = PorterStemmer()

    def tokenize(text):
        return word_tokenize(text)

    def lemmatize(tok):
        return _LEMMATIZER.lemmatize(tok)

    def stem(tok):
        return _STEMMER.stem(tok)

    HAVE_NLTK = True
except Exception:
    # Minimal fallback (NLTK yoksa) - sade ve baglantisiz
    STOP = {
        "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with",
        "at", "by", "from", "is", "are", "be", "this", "that", "it", "as", "&",
    }
    _STEM_SUFFIXES = ["ization", "iveness", "fulness", "ousness", "ing", "ied",
                      "ies", "ily", "ies", "ed", "es", "s", "ly", "er", "ment"]

    def tokenize(text):
        return re.findall(r"[A-Za-z]+", text)

    def lemmatize(tok):
        if tok.endswith("ies") and len(tok) > 4:
            return tok[:-3] + "y"
        if tok.endswith("s") and not tok.endswith("ss") and len(tok) > 3:
            return tok[:-1]
        return tok

    def stem(tok):
        for suf in _STEM_SUFFIXES:
            if tok.endswith(suf) and len(tok) - len(suf) >= 3:
                return tok[: -len(suf)]
        return tok

    HAVE_NLTK = False


def clean_tokens(text):
    tokens = tokenize(text.lower())
    return [t for t in tokens if t.isalpha() and t not in STOP]


def main():
    if not os.path.exists(SOURCE_JSON):
        raise SystemExit(
            "Kaynak bulunamadi: {}\nOnce '00_make_sample_data.py' calistirin "
            "veya gercek Yelp business JSON'unu bu yola koyun.".format(SOURCE_JSON)
        )

    businesses = []
    with open(SOURCE_JSON, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            businesses.append(json.loads(line))

    lemma_path = os.path.join(DATA_DIR, "lemmatized_sentences.csv")
    stem_path = os.path.join(DATA_DIR, "stemmed_sentences.csv")

    n_written = 0
    with open(lemma_path, "w", encoding="utf-8", newline="") as lf, \
         open(stem_path, "w", encoding="utf-8", newline="") as sf:
        lw = csv.writer(lf, delimiter=";")
        sw = csv.writer(sf, delimiter=";")
        header = ["business_id", "name", "categories", "city", "stars", "text"]
        lw.writerow(header)
        sw.writerow(header)

        for biz in businesses:
            name = str(biz.get("name", "") or "")
            cats = str(biz.get("categories", "") or "")
            raw = (name + " " + cats).strip()
            toks = clean_tokens(raw)
            if not toks:
                continue  # tamamen bos kalan kayitlari atla

            lemma_text = " ".join(lemmatize(t) for t in toks)
            stem_text = " ".join(stem(t) for t in toks)

            common = [biz.get("business_id", ""), name, cats,
                      biz.get("city", ""), biz.get("stars", "")]
            lw.writerow(common + [lemma_text])
            sw.writerow(common + [stem_text])
            n_written += 1

    print("NLTK kullanildi:" if HAVE_NLTK else "Fallback kullanildi (NLTK yok):", HAVE_NLTK)
    print("Yazildi: {} ({} satir)".format(lemma_path, n_written))
    print("Yazildi: {} ({} satir)".format(stem_path, n_written))


if __name__ == "__main__":
    main()
