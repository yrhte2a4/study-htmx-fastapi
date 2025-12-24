import asyncio
import streamlit as st
from datetime import datetime
from agents import Agent, Runner, set_tracing_disabled, AsyncOpenAI
from agents.mcp import MCPServerStdio
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
import os
from dotenv import load_dotenv

# .envファイルから環境変数を読み込み
load_dotenv()

# トレース機能を無効にする（オフにしないと401エラーが出る）
set_tracing_disabled(disabled=True)

# 環境変数から設定を取得
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
MCP_SERVER_PACKAGE = os.getenv("MCP_SERVER_PACKAGE", "awslabs.aws-documentation-mcp-server@latest")

# メインエリア
st.title("OpenAI Agent SDK MCPエージェント")
st.text("Azure OpenAIとMCPサーバーを使用したエージェント")

# 設定状況を表示
with st.expander("設定状況"):
    st.write(f"**Azure OpenAI エンドポイント**: {AZURE_OPENAI_ENDPOINT or '未設定'}")
    st.write(f"**デプロイメント名**: {AZURE_OPENAI_DEPLOYMENT_NAME or '未設定'}")
    st.write(f"**APIバージョン**: {AZURE_OPENAI_API_VERSION}")
    st.write(f"**MCPサーバーパッケージ**: {MCP_SERVER_PACKAGE}")

question = st.text_input("質問を入力", "Bedrock AgentCoreではどんなことができる？AWSドキュメントを参照して教えて")

def create_azure_openai_client():
    """Azure OpenAIクライアントを作成"""
    return AsyncOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        base_url=f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT_NAME}",
        default_headers={"api-key": AZURE_OPENAI_API_KEY},
        default_query={"api-version": AZURE_OPENAI_API_VERSION},
    )

def create_mcp_server():
    """MCPサーバーを作成"""
    return MCPServerStdio(params={
        "command": "uvx",
        "args": [MCP_SERVER_PACKAGE]
    })

def create_agent(mcp_server, custom_client):
    """Azure OpenAIを使用するエージェントを作成"""
    return Agent(
        name="Assistant",
        instructions="""ユーザーの質問に対して適切な回答を提供します。

利用可能なツールがある場合は、積極的に活用して最新かつ正確な情報を提供してください。
特に以下の場合はツールの使用を検討してください：
- 最新の情報が必要な場合
- 詳細な技術仕様や公式ドキュメントの参照が必要な場合
- 具体的なデータや統計情報が求められる場合

ツールを使用する際は、ユーザーにとって価値のある情報を取得できるよう適切に活用してください。""",
        model=OpenAIChatCompletionsModel(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,
            openai_client=custom_client,
        ),
        mcp_servers=[mcp_server],
    )

async def stream_response(agent, question, container):
    """レスポンスをストリーミング表示"""
    text_holder = container.empty()
    
    try:
        # エージェント実行
        result = await Runner.run(agent, question)
        
        # 結果を表示
        if hasattr(result, 'final_output'):
            text_holder.markdown(result.final_output)
        else:
            text_holder.markdown(str(result))
        
        # Usage情報を表示
        usage_info = None
        if hasattr(result, 'raw_responses'):
            for raw_response in result.raw_responses:
                if hasattr(raw_response, 'usage'):
                    usage_info = raw_response.usage
                    break
        
        if usage_info:
            with st.expander("📊 トークン使用量"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("入力トークン", usage_info.input_tokens)
                with col2:
                    st.metric("出力トークン", usage_info.output_tokens)
                with col3:
                    st.metric("合計トークン", usage_info.total_tokens)
                
                if hasattr(usage_info, 'input_tokens_details') and usage_info.input_tokens_details:
                    if hasattr(usage_info.input_tokens_details, 'cached_tokens'):
                        st.info(f"キャッシュされたトークン: {usage_info.input_tokens_details.cached_tokens}")
        
        # ツール実行情報を抽出・表示（VSCode IDE風）
        tool_executions = []
        
        # raw_responsesからResponseFunctionToolCallを抽出
        if hasattr(result, 'raw_responses'):
            for raw_response in result.raw_responses:
                if hasattr(raw_response, 'output'):
                    for output_item in raw_response.output:
                        # ResponseFunctionToolCallを検出
                        if hasattr(output_item, 'name') and hasattr(output_item, 'arguments'):
                            try:
                                import json
                                args_dict = json.loads(output_item.arguments)
                                tool_executions.append({
                                    'name': output_item.name,
                                    'arguments': args_dict,
                                    'call_id': getattr(output_item, 'call_id', 'unknown')
                                })
                            except json.JSONDecodeError:
                                tool_executions.append({
                                    'name': output_item.name,
                                    'arguments': output_item.arguments,
                                    'call_id': getattr(output_item, 'call_id', 'unknown')
                                })
        
        # ツール実行情報を表示（VSCode IDE風）
        if tool_executions:
            with st.expander(f"🔧 ツール実行 ({len(tool_executions)}回)"):
                for i, tool in enumerate(tool_executions, 1):
                    # ツール名をヘッダーとして表示
                    st.markdown(f"### 🛠️ `{tool['name']}`")
                    
                    # パラメータを整形表示
                    if isinstance(tool['arguments'], dict):
                        st.markdown("**パラメータ:**")
                        for key, value in tool['arguments'].items():
                            # 値の型に応じて表示を調整
                            if isinstance(value, str) and len(value) > 100:
                                # 長い文字列は折りたたみ表示
                                with st.expander(f"📄 {key}"):
                                    st.text(value)
                            elif isinstance(value, (list, dict)):
                                # リストや辞書はJSON表示
                                st.code(f"{key}: {json.dumps(value, indent=2, ensure_ascii=False)}", language="json")
                            else:
                                # その他は通常表示
                                st.code(f"{key}: {value}")
                    else:
                        # 辞書でない場合はそのまま表示
                        st.code(tool['arguments'], language="json")
                    
                    # Call IDを小さく表示
                    st.caption(f"Call ID: {tool['call_id']}")
                    
                    # 区切り線（最後以外）
                    if i < len(tool_executions):
                        st.divider()
        else:
            st.info("🤖 この質問ではツールは使用されませんでした")
            
    except Exception as e:
        error_message = str(e)
        if "RateLimitReached" in error_message:
            text_holder.error("⚠️ レート制限に達しました。60秒後に再試行してください。")
            st.info("💡 **対処方法**:\n- 60秒待ってから再試行\n- Azure OpenAIポータルでクォータ増加を申請\n- より軽量なモデルに変更")
        else:
            text_holder.error(f"エラーが発生しました: {error_message}")

async def run_agent_async(question, container):
    """非同期でエージェントを実行"""
    mcp_server = create_mcp_server()
    custom_client = create_azure_openai_client()
    
    async with mcp_server:
        # ツール一覧を表示
        tools = await mcp_server.list_tools()
        if tools:
            with st.expander(f"利用可能なツール ({len(tools)}個)"):
                for tool in tools:
                    st.write(f"- **{tool.name}**: {tool.description}")
        
        agent = create_agent(mcp_server, custom_client)
        await stream_response(agent, question, container)

# 設定チェック
def check_configuration():
    """必要な設定がすべて揃っているかチェック"""
    missing = []
    if not AZURE_OPENAI_ENDPOINT:
        missing.append("AZURE_OPENAI_ENDPOINT")
    if not AZURE_OPENAI_API_KEY:
        missing.append("AZURE_OPENAI_API_KEY")
    if not AZURE_OPENAI_DEPLOYMENT_NAME:
        missing.append("AZURE_OPENAI_DEPLOYMENT_NAME")
    
    return missing

# ボタンを押したら生成開始
if st.button("質問する"):
    missing_config = check_configuration()
    
    if missing_config:
        st.error(f"以下の環境変数が設定されていません: {', '.join(missing_config)}")
        st.info("`.env`ファイルに必要な設定を追加してください。")
    else:
        with st.spinner("回答を生成中…"):
            container = st.container()
            
            # 非同期実行
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(
                    run_agent_async(question, container)
                )
            except Exception as e:
                st.error(f"実行エラー: {str(e)}")
            finally:
                loop.close()