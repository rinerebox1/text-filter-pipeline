#!/bin/bash

# Docker クリーンアップスクリプト
# 特定のイメージとそれに関連するコンテナを強制削除

set -e

echo "=== Docker クリーンアップスクリプト ==="

# 削除対象のイメージパターンを指定（バージョン番号は動的に取得）
OPENHANDS_REPOSITORY_PATTERNS=(
    "docker.all-hands.dev/all-hands-ai/openhands"
    "docker.all-hands.dev/all-hands-ai/runtime"
)

# 関数: 特定のリポジトリパターンに一致するイメージを使用しているコンテナを停止・削除
cleanup_containers_for_repository() {
    local repo_pattern="$1"
    echo "--- $repo_pattern を使用しているコンテナをクリーンアップ中 ---"
    
    # リポジトリパターンに一致するイメージを使用しているコンテナIDを取得
    local container_ids=$(docker ps -a --format "{{.ID}} {{.Image}}" | grep "$repo_pattern" | awk '{print $1}' | tr '\n' ' ' | sed 's/[[:space:]]*$//')
    
    if [ -n "$container_ids" ]; then
        echo "発見されたコンテナ: $container_ids"
        
        # コンテナを停止
        echo "コンテナを停止中..."
        docker stop $container_ids 2>/dev/null || true
        
        # コンテナを削除
        echo "コンテナを削除中..."
        docker rm -f $container_ids 2>/dev/null || true
        
        echo "コンテナのクリーンアップが完了しました。"
    else
        echo "該当するコンテナが見つかりませんでした。"
    fi
}

# 関数: リポジトリパターンに一致するイメージを強制削除
force_remove_images_by_repository() {
    local repo_pattern="$1"
    echo "--- $repo_pattern に一致するイメージを強制削除中 ---"
    
    # リポジトリパターンに一致するイメージを取得
    local images=$(docker images --format "{{.Repository}}:{{.Tag}}" | grep "^$repo_pattern:" || true)
    
    if [ -n "$images" ]; then
        echo "削除対象のイメージ:"
        echo "$images"
        
        # 各イメージを削除
        echo "$images" | while read -r image; do
            if [ -n "$image" ]; then
                echo "削除中: $image"
                docker rmi -f "$image" 2>/dev/null || {
                    echo "警告: $image の削除に失敗しました。"
                }
            fi
        done
        echo "$repo_pattern のイメージ削除が完了しました。"
    else
        echo "$repo_pattern に一致するイメージは存在しません。"
    fi
}

# メイン処理
main() {
    echo "現在のDockerイメージとコンテナの状況:"
    echo "--- コンテナ一覧 ---"
    docker ps -a --format "table {{.ID}}\t{{.Image}}\t{{.Status}}\t{{.Names}}" | head -10
    
    echo -e "\n--- イメージ一覧 ---"
    docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.ID}}\t{{.Size}}" | grep -E "(openhands|runtime)" || echo "対象イメージが見つかりません"
    
    echo -e "\n=== クリーンアップ開始 ==="
    
    # 各リポジトリパターンに対してクリーンアップを実行
    for repo_pattern in "${OPENHANDS_REPOSITORY_PATTERNS[@]}"; do
        echo -e "\n処理中: $repo_pattern"
        cleanup_containers_for_repository "$repo_pattern"
        force_remove_images_by_repository "$repo_pattern"
    done
    
    echo -e "\n=== 未使用のDockerリソースもクリーンアップ ==="
    echo "未使用のコンテナを削除中..."
    docker container prune -f 2>/dev/null || true
    
    echo "未使用のイメージを削除中..."
    docker image prune -f 2>/dev/null || true
    
    echo -e "\n=== クリーンアップ完了 ==="
    echo "最終状態:"
    docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.ID}}\t{{.Size}}" | grep -E "(openhands|runtime)" || echo "対象イメージは全て削除されました"
}

# スクリプト実行確認
read -p "Dockerイメージとコンテナを強制削除しますか？ (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    main
else
    echo "処理をキャンセルしました。"
    exit 0
fi 