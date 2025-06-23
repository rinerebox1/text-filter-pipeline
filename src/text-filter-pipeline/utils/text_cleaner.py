import os
import logging
import re
import emoji
import hojichar
from hojichar import Compose, Document
from hojichar.core.filter_interface import Filter
from hojichar.filters.deduplication import GenerateDedupLSH, LSHDeduplicator
from hojichar import document_filters
from hojichar.core.parallel import Parallel
from bs4 import BeautifulSoup

# It's good practice for utility modules to have their own logger
# or allow a logger to be passed in. For simplicity, using a local logger.
logger = logging.getLogger(__name__)

# ======================================
# Custom Filter Definitions
# ======================================
class RemoveHTMLTags(Filter):
    """HTMLタグとエンティティを除去し、純テキストに変換するフィルタ"""
    def apply(self, doc: Document):
        text = BeautifulSoup(doc.text, "lxml").get_text(" ", strip=True)
        doc.text = text
        return doc

class RemoveTemplatePhrases(Filter):
    """記事冒頭・末尾の定型フレーズを削除するフィルタ"""
    _templates = [
        "この記事では",
        "結論から言うと",
        "最後までご覧いただきありがとうございます",
        "いかがでしたか",
        "まとめ",
        "以上です",
        "ご覧いただき",
        "ありがとうございました",
    ]
    def apply(self, doc: Document):
        t = doc.text
        for ph in self._templates:
            t = t.replace(ph, "")
        doc.text = t
        return doc

class StripEmojis(Filter):
    """絵文字や機種依存文字を完全に除去するフィルタ"""
    def apply(self, doc: Document):
        doc.text = emoji.replace_emoji(doc.text, replace="")
        return doc

def clean_texts(texts: list[str], num_cores: int | None = None) -> list[str]:
    """
    Cleans a list of text strings using HojiChar.

    Args:
        texts: A list of raw text strings.
        num_cores: The number of CPU cores to use for parallel processing.
                   Defaults to os.cpu_count().

    Returns:
        A list of cleaned text strings.
    """
    if not texts:
        logger.info("No texts provided to clean_texts function.")
        return []

    logger.info(f"Starting text cleaning with HojiChar for {len(texts)} texts.")

    # Create comprehensive cleaning pipeline based on the reference code
    cleaner = Compose([
        # --- 言語フィルタ ---
        document_filters.AcceptJapanese(p=0.5),             # FastTextベースで日本語テキストのみを通過させる(閾値0.5)
        document_filters.ExampleHojiChar(),

        # --- 前処理（テキスト整形系）---
        RemoveHTMLTags(),                                   # HTMLタグ削除
        RemoveTemplatePhrases(),                            # 定型文削除
        StripEmojis(),                                      # 絵文字削除
        document_filters.DocumentNormalizer(),              # Unicode NFKC 正規化（全角→半角など）

        # --- 形態＆長さフィルタ ---
        document_filters.DocumentLengthFilter(              # 極端に短い/長い文書を破棄
            min_doc_len=50, max_doc_len=20000),
        document_filters.DiscardTooShortLines(              # 短すぎる行が多すぎれば破棄
            threshold=0.5, min_length=5),
        document_filters.DiscardTooManyNouns(               # 名詞ばかりの羅列を破棄（要 fugashi）
            max_noun_ratio=0.8),
        document_filters.CharRepetitionRatioFilter(),       # 同一文字の連続が多すぎるテキストを除去
        document_filters.WordRepetitionRatioFilter(),       # 同じ単語・語句の繰り返しが多いテキストを除去
        document_filters.SingleCharacterRepetitionFilter(), # 単一文字の連続（例：「ああああ」）が多すぎるテキストを除去
        document_filters.DiscardTooManyEndingEllipsis(),    # 「……」など、末尾の省略記号が多いテキストを除去

        # --- 不適切表現・スパム検知 ---
        document_filters.DiscardAds(),                      # 広告スパム判定
        document_filters.DiscardAdultContentJa(),           # アダルト表現
        document_filters.DiscardViolenceContentJa(),        # 暴力表現
        document_filters.DiscardDiscriminationContentJa(),  # 差別表現
        document_filters.DiscardTooManySpecialToken(        # 記号や絵文字ばかりの文書を破棄
            threshold=0.4),

        # --- 重複除去 ---
        GenerateDedupLSH(),                                  # LSH (局所感度ハッシュ) を使って重複検出情報をDocumentに付与
        LSHDeduplicator(),                                   # 既出ハッシュとの重複判定・破棄
    ])

    input_documents = [hojichar.Document(text) for text in texts]

    if num_cores is None:
        num_cores = os.cpu_count() or 1  # Default to 1 core if os.cpu_count() is None

    logger.info(f"Using {num_cores} CPU cores for HojiChar processing.")

    cleaned_texts_content = []
    rejected_count = 0
    
    # Initialize Parallel with the cleaner and number of jobs
    with Parallel(cleaner, num_jobs=num_cores) as pfilter:
        # pfilter.imap_apply returns an iterator of processed Document objects
        for i, doc in enumerate(pfilter.imap_apply(input_documents)):
            if not doc.is_rejected:
                cleaned_texts_content.append(doc.text)
            else:
                rejected_count += 1
            
            if (i + 1) % 100 == 0: # Log progress every 100 texts
                logger.info(f"HojiChar processed {i+1}/{len(texts)} texts in clean_texts... "
                           f"(Accepted: {len(cleaned_texts_content)}, Rejected: {rejected_count})")

    logger.info(f"HojiChar cleaning complete in clean_texts. "
               f"Processed {len(texts)} texts -> "
               f"Accepted: {len(cleaned_texts_content)}, Rejected: {rejected_count}")
    return cleaned_texts_content


if __name__ == '__main__':
    # Example usage:
    # This part is for testing the module directly.
    # It requires HojiChar to be installed.
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    sample_texts = [
        "これは最初のサンプルテキストです。日本語のテキストで、十分な長さがあります。テストのために作成されたサンプルです。",
        "これは　２番目の　テキストです。　スペースが多いです。全角数字や全角スペースが含まれています。正規化のテストに使用します。",
        "This is the third text with English words. It should be filtered out by AcceptJapanese filter.",
        "<p>HTMLタグが含まれたテキストです。</p><div>これらのタグは除去されるはずです。</div>BeautifulSoupによって処理されます。",
        "この記事では、テンプレートフレーズのテストを行います。定型文が除去されることを確認します。最後までご覧いただきありがとうございます。",
        "絵文字テスト😀🎉✨ これらの絵文字は除去されるはずです 🚀💯",
        "髙島屋と髙島屋での買い物体験について。正規化により髙が高に変換されるかテストします。",
        "短い", # This should be filtered out by DocumentLengthFilter
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", # This should be filtered out by SingleCharacterRepetitionFilter
        "重複重複重複重複重複重複重複重複重複重複", # This should be filtered out by WordRepetitionRatioFilter
    ]

    logger.info(f"Original texts count: {len(sample_texts)}")
    for i, text in enumerate(sample_texts):
        logger.info(f"Text {i+1}: {text[:50]}...")

    # Using default number of cores
    cleaned = clean_texts(sample_texts)

    logger.info(f"Cleaned texts count: {len(cleaned)}")
    for i, text in enumerate(cleaned):
        logger.info(f"Cleaned {i+1}: {text[:100]}...")

    # Note: Due to filtering, len(cleaned) may be less than len(sample_texts)
    logger.info(f"Filtering ratio: {len(cleaned)}/{len(sample_texts)} = {len(cleaned)/len(sample_texts)*100:.1f}% passed")

    logger.info("Test run of text_cleaner.py complete.")

