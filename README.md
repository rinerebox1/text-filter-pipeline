# Python Template for OpenHands

[OpenHands](https://github.com/All-Hands-AI/OpenHands)との協働に最適化された、プロダクション対応のPythonプロジェクトテンプレートです。自律型AI開発エージェントであるOpenHandsがタスクを遂行しやすいよう、厳格な型チェック、自動パフォーマンス測定、包括的なドキュメント、そしてタスク指示のためのガイドを備えています。

## 🔗 重要なドキュメント

- **[OPENHANDS.md](./OPENHANDS.md)**: OpenHands向けの技術仕様・タスク指示ガイド
- **[template/](./template/)**: ベストプラクティスのモデルコード集

## 📋 OpenHands CLI チートシート

OpenHandsは、自然言語でタスクを指示し、AIエージェントが自律的に計画を立てて実行するツールです。そのため、コマンドは主にエージェントの制御と設定に焦点を当てています。

#### 基本的な使い方
```bash
# 対話モードでOpenHandsを起動
openhands

# 起動後、プロンプトにタスクを自然言語で入力
> src/utils/math.py に、2つの数値を加算する関数を追加してください。
> また、その関数の単体テストも tests/unit/test_math.py に作成してください。
```

#### 対話モード内コマンド
OpenHandsの対話モード中に使用できるコマンド：

```bash
# セッション・対話管理
/help                          # ヘルプ・利用可能コマンド一覧表示
/new                           # 現在の対話を破棄し、新しいセッションを開始
/history                       # 現在のセッションの会話履歴を表示
/exit                          # OpenHandsを終了
/quit                          # OpenHandsを終了（/exitの別名）

# 設定管理
/settings                      # 現在の設定（LLMモデル、APIキー等）を表示・変更
/settings list                 # 設定可能な項目を一覧表示
/settings set <key> <value>    # 指定したキーの値を設定 (例: /settings set LLM_MODEL gpt-4o)

# エージェント制御
/run                           # 現在のタスクプランを実行
/approve                       # エージェントが提案したアクションを承認
/reject                        # エージェントが提案したアクションを拒否
/think                         # エージェントに次のステップを考えさせる
/plan                          # 現在のタスクに対する計画を再表示

# デバッグ・診断
/debug                         # デバッグ情報を表示
/status                        # システム状態・接続状況表示
/version                       # OpenHandsのバージョン表示
/reset                         # 設定をデフォルトにリセット
```

#### 環境変数による設定
CLIを起動する前に環境変数を設定することで、APIキーやモデルを構成できます。

```bash
# 基本設定（必須）
export LLM_API_KEY="your_api_key"  # OpenAI, AnthropicなどのAPIキー

# モデル・エージェント設定
export LLM_MODEL="gpt-4o"          # 使用するLLMモデル (例: gpt-4o, claude-3-opus-20240229)
export AGENT="CodeActAgent"        # 使用するエージェントクラス

# 高度な設定
export MAX_ITERATIONS=20           # エージェントの最大反復回数
export SANDBOX_TYPE="exec"         # コード実行に使用するサンドボックスのタイプ
export WORKSPACE_DIR="./workspace" # エージェントが作業するディレクトリ
export LOG_LEVEL="INFO"            # ログレベル (DEBUG, INFO, WARN, ERROR)
```

#### タスク指示のベストプラクティス

明確で具体的な指示が、AIエージェントの成功率を高めます。

```bash
# 良い例：具体的で明確な指示
> 1. `src/api/endpoints.py`に新しいPOSTエンドポイント `/users` を作成してください。
> 2. このエンドポイントは、リクエストボディとしてユーザー名(username)とメールアドレス(email)を受け取ります。
> 3. 受け取ったデータは `src/db/database.py` の `add_user` 関数を使って保存してください。
> 4. 最後に、この新しいエンドポイントに対するテストを `tests/integration/test_api.py` に追加してください。

# 避けるべき例：曖昧な指示
> ユーザー機能を作って
```

#### 高度な使用例・統合活用

```bash
# 特定のモデルとエージェントを指定して起動
export LLM_MODEL="claude-3-opus-20240229"
export AGENT="PlannerAgent"
openhands

# デバッグモードで詳細なログを確認しながら実行
export LOG_LEVEL="DEBUG"
openhands

# GitHub Actionsでの自動化（例）
# .github/workflows/fix-issue.yml
# - name: Run OpenHands to fix issue
#   run: |
#     openhands --task "${{ github.event.issue.body }}" --non-interactive
#   env:
#     LLM_API_KEY: ${{ secrets.LLM_API_KEY }}

```

## 🚀 クイックスタート

### このテンプレートを使用する

1.  GitHubで「Use this template」ボタンをクリックして新しいリポジトリを作成
2.  新しいリポジトリをクローン
3.  セットアップスクリプトを実行

```bash
# 新しいリポジトリをクローン
git clone https://github.com/yourusername/project-name.git
cd project-name

# セットアップ
make setup
```

セットアップスクリプトは以下を実行します：
- すべての `project_name` を実際のプロジェクト名に更新（途中でプロジェクト名を入力するように求められます）
- uvを使用してPython環境を初期化
- OpenHandsをインストール（またはDocker環境を構築）
- GitHub CLI（`gh`）をインストール（途中でログインを求められます）
- すべての依存関係をインストール
- pre-commitフックを設定
- 初期テストを実行

### 手動セットアップ（代替方法）

手動セットアップを希望する場合：

```bash
# プロジェクト名を更新
python scripts/update_project_name.py your_project_name

# uvをインストール（まだインストールしていない場合）
curl -LsSf https://astral.sh/uv/install.sh | sh

# Pythonバージョンを設定
uv python pin 3.12

# 依存関係とOpenHandsをインストール
uv sync --all-extras
uv pip install openhands-ai

# pre-commitフックをインストール
uv run pre-commit install
uv run pre-commit install --hook-type commit-msg

# テストを実行
uv run pytest
```

## ✨ 主な特徴

### 🛠️ 開発ツールチェーン
- **[uv](https://github.com/astral-sh/uv)** - 高速なPythonパッケージマネージャー
- **[Ruff](https://github.com/astral-sh/ruff)** - 超高速Pythonリンター・フォーマッター
- **[mypy](https://mypy-lang.org/)** - strictモード＋PEP 695型構文対応
- **[pytest](https://pytest.org/)** - カバレッジ付きテストフレームワーク
- **[hypothesis](https://hypothesis.readthedocs.io/)** - プロパティベーステストフレームワーク
- **[pytest-benchmark](https://pytest-benchmark.readthedocs.io/)** - 自動パフォーマンステスト
- **[bandit](https://github.com/PyCQA/bandit)** - セキュリティスキャン
- **[pip-audit](https://github.com/pypa/pip-audit)** - 依存関係の脆弱性チェック
- **[pre-commit](https://pre-commit.com/)** - コード品質用Gitフック

### 🔍 コード品質・型安全性
- ✅ PEP 695新型構文（`type` statement）対応
- ✅ TypedDict・Literal・Protocol活用の堅牢な型システム
- ✅ JSON操作用の型安全なユーティリティ
- ✅ プロパティベーステストによるエッジケース検証
- ✅ 包括的なヘルパー関数テストスイート
- ✅ 自動セキュリティ・脆弱性チェック

### ⚡ パフォーマンス・プロファイリング
- ✅ `@profile`、`@timeit`デコレータによる性能測定
- ✅ 自動ベンチマークCI（PR時の性能比較レポート）
- ✅ コンテキストマネージャー型プロファイラー
- ✅ 性能回帰検出システム
- ✅ メモリ・実行時間の詳細監視

### 🔄 CI/CD・自動化
- ✅ 並列実行対応の高速CIパイプライン
- ✅ 自動パフォーマンスベンチマーク（PR時レポート生成）
- ✅ Dependabotによる自動依存関係更新
- ✅ GitHub CLIによるワンコマンドPR・Issue作成
- ✅ キャッシュ最適化された実行環境

### 📚 包括的ドキュメント
- ✅ **OPENHANDS.md** - AIエージェント向けタスク指示のベース
- ✅ **専門ガイド** - ML/バックエンドプロジェクト対応
- ✅ **協働戦略ガイド** - 人間とOpenHandsの効果的な連携方法
- ✅ **ドキュメント更新プロトコル** - ドキュメント品質管理フレームワーク

## 📁 プロジェクト構造

```
project-root/
├── .github/                     # GitHub Actionsの設定ファイル
│   ├── workflows/               # CI/CD + ベンチマークワークフロー
│   │   ├── ci.yml              # メインCI（テスト・リント・型チェック）
│   │   └── benchmark.yml       # パフォーマンスベンチマーク
│   ├── dependabot.yml           # Dependabotの設定
│   ├── ISSUE_TEMPLATE/          # Issueテンプレート
│   └── PULL_REQUEST_TEMPLATE.md # Pull Requestテンプレート
├── src/
│   └── project_name/            # メインパッケージ（uv syncでインストール可能）
│       ├── __init__.py
│       ├── py.typed             # PEP 561準拠の型情報マーカー
│       ├── types.py             # プロジェクト共通型定義
│       ├── core/                # コアロジック
│       └── utils/               # ユーティリティ
├── tests/                       # テストコード
│   ├── unit/                    # 単体テスト
│   ├── property/                # プロパティベーステスト
│   ├── integration/             # 統合テスト
│   └── conftest.py              # pytest設定
├── docs/                        # ドキュメント
├── scripts/                     # セットアップスクリプト
├── pyproject.toml               # 依存関係・ツール設定
├── .pre-commit-config.yaml      # pre-commit設定
├── README.md                    # プロジェクト説明
└── OPENHANDS.md                 # OpenHands用ガイド
```

## 📚 ドキュメント階層

### 🎯 メインドキュメント
- **[OPENHANDS.md](OPENHANDS.md)** - 包括的プロジェクトガイド
  - プロジェクト概要・コーディング規約
  - タスク指示の基本とベストプラクティス
  - 型ヒント・テスト戦略・セキュリティ

### 🤝 戦略ガイド

### 🎨 プロジェクトタイプ別ガイド
- **[ml-project-guide.md](docs/ml-project-guide.md)** - 機械学習プロジェクト
  - PyTorch・Hydra・wandb統合設定
  - 実験管理・データバージョニング
  - GPU最適化・モデル管理

- **[backend-project-guide.md](docs/backend-project-guide.md)** - FastAPIバックエンド
  - 非同期データベース操作・JWT認証
  - API設計・セキュリティ設定
  - Docker開発環境・プロダクション考慮事項

## ✅ 新規プロジェクト設定チェックリスト

### 🔧 基本プロジェクト設定
- [ ] **作者情報更新**: `pyproject.toml`の`authors`セクション
- [ ] **ライセンス選択**: LICENSEファイルを適切なライセンスに更新
- [ ] **README.md更新**: プロジェクト固有の説明・機能・使用方法
- [ ] **OPENHANDS.md カスタマイズ**: プロジェクト概要やタスク指示の基本方針を更新
- [ ] **専門ガイドの追加**: 適宜`docs/`内に詳細なガイドを追加

### ⚙️ 開発環境・品質設定
- [ ] **依存関係調整**: プロジェクトに必要な追加パッケージの導入
- [ ] **リントルール**: プロジェクトに合わせた`ruff`設定のカスタマイズ
- [ ] **テストカバレッジ**: `pytest`カバレッジ要件の調整
- [ ] **プロファイリング**: パフォーマンス要件に応じたベンチマーク設定

### 🔐 GitHubリポジトリ・セキュリティ設定
- [ ] **ブランチ保護**: `main`ブランチの保護ルール有効化
- [ ] **PR必須レビュー**: Pull Request作成時のレビュー要求設定
- [ ] **ステータスチェック**: CI・型チェック・テストの必須化
- [ ] **Dependabot**: 自動依存関係更新の有効化
- [ ] **Issues/Projects**: 必要に応じてプロジェクト管理機能の有効化

## 🔧 カスタマイズ

### 型チェックの厳格さ調整

mypyのstrictモードが最初から厳しすぎる場合：

```toml
# pyproject.toml - 基本設定から開始
[tool.mypy]
python_version = "3.12"
warn_return_any = true
warn_unused_configs = true

# 段階的により厳格な設定を有効化
[[tool.mypy.overrides]]
module = ["project_name.core.*"]
strict = true  # まずコアモジュールにstrictモードを適用
```

### リントルールの変更

```toml
# pyproject.toml
[tool.ruff.lint]
# 必要に応じてルールコードを追加・削除
select = ["E", "F", "I"]  # 基本から開始
ignore = ["E501"]  # 行の長さはフォーマッターが処理
```

### テストカバレッジ要件の変更

```toml
# pyproject.toml
[tool.pytest.ini_options]
addopts = [
    "--cov-fail-under=60",  # 初期要件を低めに設定
]
```

## 🔗 外部リソース・参考資料

### 🛠️ 開発ツール公式ドキュメント
- **[uv ドキュメント](https://docs.astral.sh/uv/)** - Pythonパッケージ管理
- **[Ruff ドキュメント](https://docs.astral.sh/ruff/)** - リント・フォーマッター
- **[mypy ドキュメント](https://mypy.readthedocs.io/)** - 型チェッカー
- **[pytest ドキュメント](https://docs.pytest.org/en/stable/)** - テストフレームワーク
- **[Hypothesis ドキュメント](https://hypothesis.readthedocs.io/)** - プロパティベーステスト

### 🤖 OpenHands関連
- **[OpenHands GitHubリポジトリ](https://github.com/All-Hands-AI/OpenHands)** - ソースコード・Issue
- **[OpenHands ドキュメント](https://all-hands-ai.github.io/OpenHands/)** - 公式ドキュメント・使用方法

### 🐍 Python・型ヒント
- **[PEP 695 - Type Parameter Syntax](https://peps.python.org/pep-0695/)** - 新型構文仕様
- **[TypedDict Guide](https://docs.python.org/3/library/typing.html#typing.TypedDict)** - 型安全な辞書
- **[Python 3.12 リリースノート](https://docs.python.org/3/whatsnew/3.12.html)** - 新機能一覧

---

## 📄 ライセンス

このテンプレートはMITライセンスの下でリリースされています。詳細は[LICENSE](LICENSE)をご覧ください。