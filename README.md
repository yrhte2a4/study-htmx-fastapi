# MCP Host 学習プロジェクト

このプロジェクトは、Model Context Protocol (MCP) ホストの実装を学習目的で作成したものです。StreamlitとFastAPI + htmxの2つの異なるアプローチでMCPサーバーとの連携を実装しています。

## 概要

- **Streamlit版**: シンプルなWebUIでMCPエージェントとの対話
- **FastAPI + htmx版**: よりインタラクティブなWebアプリケーション

両方の実装でAzure OpenAIとMCPサーバー（デフォルトでAWS Documentation MCP Server）を使用してAIエージェントを構築しています。

## 必要な環境

- Python 3.12以上
- uv（Pythonパッケージマネージャー）
- Azure OpenAIのAPIキーとエンドポイント

## セットアップ

### 1. 依存関係のインストール

```bash
uv sync
```

### 2. 環境変数の設定

`.env.example`を参考に`.env`ファイルを作成し、以下の環境変数を設定してください：

```bash
# Azure OpenAI設定
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_API_KEY=your_api_key_here
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o-mini
AZURE_OPENAI_API_VERSION=2025-01-01-preview

# MCPサーバー設定（オプション）
MCP_SERVER_PACKAGE=awslabs.aws-documentation-mcp-server@latest
```

## 使用方法

### Streamlit版の起動

```bash
uv run streamlit run streamlit.py
```

ブラウザで `http://localhost:8501` にアクセスしてください。

### FastAPI版の起動

```bash
uv run fastapi dev main.py
```

ブラウザで `http://localhost:8000` にアクセスしてください。

## 機能

### 共通機能
- Azure OpenAIを使用したAIエージェント
- MCPサーバーとの連携（デフォルト：AWS Documentation）
- ツール実行履歴の表示
- トークン使用量の表示
- 設定状況の確認

### Streamlit版の特徴
- シンプルで直感的なUI
- 質問と回答の基本的な対話形式
- ツール実行の詳細表示

### FastAPI + htmx版の特徴
- よりリッチなWebUI
- htmxを使用したインタラクティブな操作
- Jinja2テンプレートによる柔軟なレンダリング
- エラーハンドリングの強化

## プロジェクト構造

```
.
├── main.py              # FastAPI + htmxアプリケーション
├── streamlit.py         # Streamlitアプリケーション
├── templates/           # FastAPI用HTMLテンプレート
│   ├── index.html
│   ├── chat_response.html
│   ├── error_response.html
│   ├── settings_modal.html
│   └── tools_list.html
├── pyproject.toml       # プロジェクト設定と依存関係
├── .env.example         # 環境変数の例
└── README.md           # このファイル
```

## 主な依存関係

- **FastAPI**: 高性能なWebフレームワーク
- **Streamlit**: データアプリケーション構築フレームワーク
- **openai-agents**: OpenAI Agent SDKライブラリ
- **Jinja2**: テンプレートエンジン
- **python-dotenv**: 環境変数管理

## 学習のポイント

このプロジェクトでは以下の技術要素を学習できます：

1. **MCP（Model Context Protocol）の基本概念**
   - MCPサーバーとの接続と通信
   - ツールの動的な取得と実行

2. **異なるWebフレームワークでの実装比較**
   - Streamlit: プロトタイピング向けの簡単な実装
   - FastAPI + htmx: 本格的なWebアプリケーション開発

3. **Azure OpenAIとの連携**
   - カスタムクライアントの作成
   - エージェントの設定とカスタマイズ

4. **非同期処理**
   - async/awaitを使用した非同期プログラミング
   - ライフサイクル管理

## トラブルシューティング

### よくある問題

1. **環境変数が設定されていない**
   - `.env`ファイルが正しく作成されているか確認
   - 必要な環境変数がすべて設定されているか確認

2. **MCPサーバーの起動に失敗**
   - `uvx`コマンドが利用可能か確認
   - ネットワーク接続を確認

3. **Azure OpenAIのレート制限**
   - 60秒待ってから再試行
   - Azure OpenAIポータルでクォータを確認

## ライセンス

このプロジェクトは学習目的で作成されています。