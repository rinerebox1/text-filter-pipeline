import re, emoji, json, os
from bs4 import BeautifulSoup
from faker import Faker
import hojichar
from hojichar import Compose, Document
from hojichar.core.filter_interface import Filter
from hojichar.filters.deduplication import GenerateDedupLSH, LSHDeduplicator
from hojichar import document_filters
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn

# 日本語ロケールの Faker を初期化
fake = Faker('ja_JP')
console = Console()

# ======================================
# 1) ダミーデータジェネレーター
# ======================================
def generate_dummy(n):
    """
    - 日本語テキストのダミーデータを n 件生成する。
    - 各項目は JSON 形式 (key="text") の文字列にして返す。
    """
    for _ in range(n):
        # 50文字前後の段落ランダム生成
        text1 = fake.text(max_nb_chars=50)
        # 10語前後の短文生成
        text2 = fake.sentence(nb_words=10)
        # テンプレートとランダム生成を混在させる
        txt = f"{text1} {text2}"
        yield json.dumps({"text": txt}, ensure_ascii=False)

# ======================================
# 2) カスタムフィルタ定義
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

# ======================================
# 3) Compose パイプライン定義
#    上から順番に適用される
# ======================================
cleaner = Compose([
    document_filters.JSONLoader(key="text"),            # JSON形式の入力から "text" 値を抽出して Document にロード

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

    # --- 出力 ---
    document_filters.JSONDumper()                        # フィルタ後のテキストをJSON形式で出力

])

# ======================================
# 4) メイン処理：生成からフィルタリング、統計出力
# ======================================
def main(num=1000, jobs=4, raw_path="dummy_raw.jsonl", filtered_path="filtered_dummy.jsonl"):
    # 概要テーブルの準備
    summary = {"初期": 0, "通過": 0, "破棄": 0}

    # 1) ダミーデータ生成と保存
    console.rule("データ生成フェーズ")
    with Progress(TextColumn("{task.fields[phase]}", justify="right"), BarColumn(), TimeElapsedColumn()) as prog:
        gen_task = prog.add_task("", total=num, phase="生成中...")
        with open(raw_path, "w", encoding="utf-8") as raw_f:
            for line in generate_dummy(num):
                raw_f.write(line + "\n")
                prog.update(gen_task, advance=1)
                summary["初期"] += 1
    console.print(f"[bold]初期生成データ数:[/] {summary['初期']} 件")

    # 2) フィルタリングを実行
    console.rule("フィルタリングフェーズ")
    input_iter = (Document(line) for line in open(raw_path, encoding="utf-8"))
    with Progress(TextColumn("{task.fields[phase]}", justify="right"), BarColumn(), TimeElapsedColumn()) as prog:
        filter_task = prog.add_task("", total=summary["初期"], phase="フィルタ中...")
        with hojichar.Parallel(cleaner, num_jobs=jobs) as p, open(filtered_path, "w", encoding="utf-8") as fout:
            for doc in p.imap_apply(input_iter):
                prog.update(filter_task, advance=1)
                if not doc.is_rejected:
                    fout.write(doc.text + "\n")
                    summary["通過"] += 1
    summary["破棄"] = summary["初期"] - summary["通過"]

    # 3) 結果概要をテーブル表示
    console.rule("フィルタ結果サマリー")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("種別", justify="center")
    table.add_column("件数", justify="right")
    for key in ["初期", "通過", "破棄"]:
        table.add_row(key, str(summary[key]))
    console.print(table)

if __name__ == "__main__":
    # 利用可能なCPUコア数の取得
    num_cores = os.cpu_count()
    main(num=5000, jobs=num_cores)