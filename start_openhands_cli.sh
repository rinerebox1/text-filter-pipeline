#!/bin/bash

# OpenHands CLI モード起動スクリプト（OpenRouter プロバイダー使用）
# 使用方法: ./start_openhands_cli.sh

set -e

echo "🚀 OpenHands CLI モード起動スクリプト（OpenRouter プロバイダー）"
echo "================================================"

# .envファイルから環境変数をロード
if [ -f ".env" ]; then
    echo "📄 .envファイルから設定をロード中..."
    set -a
    source .env
    set +a
    echo "✅ .envファイルをロードしました"
else
    echo "ℹ️  .envファイルが見つかりません（手動設定モード）"
fi

# 必要なディレクトリの作成と権限設定
echo "📁 必要なディレクトリを準備中..."

# ホームディレクトリの確実な取得
if [ -z "$HOME" ]; then
    HOME=$(eval echo ~$(whoami))
    echo "⚠️  HOME環境変数が設定されていないため、自動取得しました: $HOME"
fi

# ホームディレクトリの存在確認
if [ ! -d "$HOME" ]; then
    echo "❌ ホームディレクトリが見つかりません: $HOME"
    echo "💡 手動でホームディレクトリを指定してください:"
    read -p "ホームディレクトリのパス: " manual_home
    if [ -d "$manual_home" ]; then
        HOME="$manual_home"
        echo "✅ ホームディレクトリを設定しました: $HOME"
    else
        echo "❌ 指定されたディレクトリが存在しません"
        exit 1
    fi
fi

OPENHANDS_DIR="$HOME/.openhands"
OPENHANDS_STATE_DIR="$OPENHANDS_DIR/.openhands-state"

echo "📍 OpenHandsディレクトリパス: $OPENHANDS_DIR"

# OpenHandsディレクトリの作成
if [ ! -d "$OPENHANDS_DIR" ]; then
    mkdir -p "$OPENHANDS_DIR"
    echo "✅ $OPENHANDS_DIR を作成しました"
fi

# OpenHands状態ディレクトリの作成
if [ ! -d "$OPENHANDS_STATE_DIR" ]; then
    mkdir -p "$OPENHANDS_STATE_DIR"
    echo "✅ $OPENHANDS_STATE_DIR を作成しました"
fi

# ディレクトリの権限を確実に設定（再帰的に適用）
chmod -R 755 "$OPENHANDS_DIR"
chown -R $(id -u):$(id -g) "$OPENHANDS_DIR"
echo "✅ ディレクトリの権限と所有者を設定しました"

# 既存のJWTシークレットファイルがあれば削除（権限問題を回避）
JWT_SECRET_FILE="$OPENHANDS_STATE_DIR/.jwt_secret"
if [ -f "$JWT_SECRET_FILE" ]; then
    rm -f "$JWT_SECRET_FILE"
    echo "✅ 既存のJWTシークレットファイルを削除しました"
fi

# 設定値の確認・入力
echo "📝 設定を確認してください："

# ワークスペースパスの設定
if [ -z "$SANDBOX_VOLUMES" ]; then
    read -p "ワークスペースのパス（現在のディレクトリを使用する場合は Enter）: " workspace_path
    if [ -z "$workspace_path" ]; then
        workspace_path=$(pwd)
    fi
    # OpenHands形式に変換: host_path:container_path:mode
    export SANDBOX_VOLUMES="$workspace_path:/workspace:rw"
else
    echo "ワークスペースパス: $SANDBOX_VOLUMES"
    # 既存の設定が正しい形式かチェック
    if [[ ! "$SANDBOX_VOLUMES" =~ : ]]; then
        echo "⚠️  SANDBOX_VOLUMESの形式を修正中..."
        export SANDBOX_VOLUMES="$SANDBOX_VOLUMES:/workspace:rw"
    fi
fi

# OpenRouter API キーの設定
if [ -z "$LLM_API_KEY" ]; then
    read -s -p "OpenRouter API キー: " api_key
    echo
    if [ -z "$api_key" ]; then
        echo "❌ API キーが必要です"
        exit 1
    fi
    export LLM_API_KEY="$api_key"
else
    echo "API キー: 設定済み"
fi

# LLM モデルの設定
if [ -z "$LLM_MODEL" ]; then
    echo "利用可能なモデル例："
    echo "  - openrouter/anthropic/claude-sonnet-4"
    echo "  - openrouter/anthropic/claude-3.5-sonnet"
    echo "  - openrouter/openai/gpt-4"
    echo "  - openrouter/google/gemini-pro"
    read -p "使用するモデル名（デフォルト: openrouter/anthropic/claude-sonnet-4）: " model_name
    if [ -z "$model_name" ]; then
        model_name="openrouter/anthropic/claude-sonnet-4"
    fi
    export LLM_MODEL="$model_name"
else
    echo "モデル: $LLM_MODEL"
fi

# プロバイダーの設定
export LLM_PROVIDER="openrouter"
export LLM_BASE_URL="https://openrouter.ai/api/v1"

# ユーザーIDとグループIDの取得
USER_ID=$(id -u)
GROUP_ID=$(id -g)

echo
echo "🔧 設定内容："
echo "  プロバイダー: $LLM_PROVIDER"
echo "  モデル: $LLM_MODEL"
echo "  llm_base_url: $LLM_BASE_URL"
echo "  ワークスペース: $SANDBOX_VOLUMES"
echo "  ユーザーID: $USER_ID"
echo "  グループID: $GROUP_ID"
echo

# 確認
read -p "この設定で起動しますか？ (y/N): " confirm
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo "❌ 起動をキャンセルしました"
    exit 1
fi

echo
echo "🐳 Docker コンテナを起動中..."

# コンテナ名の生成
container_name="openhands-app-$(date +%Y%m%d%H%M%S)"

# Docker コンテナの起動
docker run -it \
  --pull=always \
  -e SANDBOX_RUNTIME_CONTAINER_IMAGE=docker.all-hands.dev/all-hands-ai/runtime:0.45-nikolaik \
  -e SANDBOX_USER_ID="$USER_ID" \
  -e SANDBOX_GROUP_ID="$GROUP_ID" \
  -e SANDBOX_VOLUMES="$SANDBOX_VOLUMES" \
  -e LLM_PROVIDER="$LLM_PROVIDER" \
  -e LLM_MODEL="$LLM_MODEL" \
  -e LLM_API_KEY="$LLM_API_KEY" \
  -e LLM_BASE_URL="$LLM_BASE_URL" \
  -e PUID="$USER_ID" \
  -e PGID="$GROUP_ID" \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v "$OPENHANDS_DIR:/.openhands:rw" \
  -v "$OPENHANDS_STATE_DIR:/.openhands-state:rw" \
  --add-host host.docker.internal:host-gateway \
  --name "$container_name" \
  docker.all-hands.dev/all-hands-ai/openhands:0.45 \
  python -m openhands.cli.main --override-cli-mode true

echo
echo "✅ OpenHands CLI モードが終了しました"
echo "💡 ヒント："
echo "  - CLI内で '/settings' コマンドで設定を確認できます"
echo "  - CLI内で '/help' コマンドで利用可能なコマンドを確認できます"
echo "  - .envファイルに以下の形式で設定を保存できます："
echo "    LLM_API_KEY=your_openrouter_api_key"
echo "    LLM_MODEL=openrouter/anthropic/claude-sonnet-4"
echo "    SANDBOX_VOLUMES=/path/to/workspace"
echo "  - 権限問題が発生した場合は、以下のコマンドでディレクトリを再作成してください："
echo "    rm -rf ~/.openhands && mkdir -p ~/.openhands && chmod 755 ~/.openhands"
echo "  - それでも問題が解決しない場合は、以下を試してください："
echo "    sudo chown -R \$(id -u):\$(id -g) ~/.openhands"
echo "    chmod -R 755 ~/.openhands"
echo "  - Dockerコンテナの権限問題については以下のリンクを参照："
echo "    https://forums.docker.com/t/change-directory-permission-in-docker-container-and-preserving-them/89848" 