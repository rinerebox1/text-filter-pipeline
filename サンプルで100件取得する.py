# pip install --upgrade datasets

import os
import json
import itertools
import random
from datasets import load_dataset

# ----------------------------------------------------
# 1. 準備：キャッシュディレクトリと保存するファイル名を設定
# ----------------------------------------------------
# Hugging Faceデータセットのキャッシュを保存するディレクトリ
cache_directory = "./hf_cache_dir" 
os.makedirs(cache_directory, exist_ok=True)

# 保存するJSONファイルの名前
output_filename = "fineweb_100_samples.json"

try:
    # ----------------------------------------------------
    # 2. データセットの読み込み
    # ----------------------------------------------------
    print("データセットを読み込んでいます...（ストリーミング）")
    # cache_dir を指定してストリーミング読み込み
    # trust_remote_code=True は、データセットが必要とするカスタムコードの実行を許可します
    dataset = load_dataset(
        "hotchpotch/fineweb-2-edu-japanese",
        name="small_tokens_cleaned",
        split="train",
        streaming=True,
        cache_dir=cache_directory,
        trust_remote_code=True 
    )

    # ----------------------------------------------------
    # 3. データの取得とシャッフル
    # ----------------------------------------------------
    print("データセットから先頭100件を取得し、シャッフルします...")
    # 先頭から100件取得
    first_100 = list(itertools.islice(dataset, 100))
    # 取得した100件をシャッフル
    random.shuffle(first_100)
    
    # 取得した件数を確認
    num_records = len(first_100)
    print(f"{num_records}件のデータを取得しました。")

    # ----------------------------------------------------
    # 4. JSONファイルとして保存  <-- ここが変更点
    # ----------------------------------------------------
    print(f"データを'{output_filename}'に保存しています...")
    # 'w' (書き込みモード) と 'utf-8' (文字コード) を指定してファイルを開く
    # with文を使うと、処理が終わった後に自動でファイルが閉じられるので安全です
    with open(output_filename, 'w', encoding='utf-8') as f:
        # json.dump() を使ってリストをファイルに書き込む
        #   - first_100: 書き込むデータ（Pythonのリスト）
        #   - f: 書き込み先のファイルオブジェクト
        #   - ensure_ascii=False: 日本語をそのままの文字で保存（\uXXXXのようなエスケープをしない）
        #   - indent=4: 人間が読みやすいように4スペースのインデントを付けて整形する
        json.dump(first_100, f, ensure_ascii=False, indent=4)
        
    print(f"完了: '{output_filename}' に {num_records} 件のデータを保存しました。")


except Exception as e:
    print(f"エラーが発生しました: {e}")
