import json
import os
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from contextlib import asynccontextmanager
from datetime import datetime
from agents import Agent, Runner, set_tracing_disabled, AsyncOpenAI
from agents.mcp import MCPServerStdio
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from dotenv import load_dotenv

# 初期設定
load_dotenv()
set_tracing_disabled(disabled=True)

# 環境変数から設定を取得
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
MCP_SERVER_PACKAGE = os.getenv("MCP_SERVER_PACKAGE", "awslabs.aws-documentation-mcp-server@latest")

# グローバル変数
mcp_server: Optional[MCPServerStdio] = None
agent: Optional[Agent] = None
available_tools: List[Any] = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    """アプリケーションのライフサイクル管理"""
    global mcp_server, agent, available_tools
    
    # 起動時の初期化
    try:
        mcp_server = create_mcp_server()
        await mcp_server.__aenter__()
        available_tools = await mcp_server.list_tools()
        agent = create_agent(mcp_server, create_azure_openai_client())
    except Exception:
        # エラーが発生してもアプリケーションは起動する
        available_tools = []
        agent = None
    
    yield
    
    # 終了時のクリーンアップ
    await safe_cleanup(mcp_server)

app = FastAPI(title="OpenAI Agent SDK MCPエージェント", lifespan=lifespan)
templates = Jinja2Templates(directory="templates")

# Jinja2にカスタム関数を追加
templates.env.globals.update({
    "moment": lambda: type('obj', (object,), {
        'format': lambda self, fmt: datetime.now().strftime(fmt.replace('HH:mm', '%H:%M'))
    })(),
    "format_time": lambda: datetime.now().strftime("%H:%M")
})

def create_azure_openai_client() -> AsyncOpenAI:
    """Azure OpenAIクライアントを作成"""
    return AsyncOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        base_url=f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT_NAME}",
        default_headers={"api-key": AZURE_OPENAI_API_KEY},
        default_query={"api-version": AZURE_OPENAI_API_VERSION},
    )

def create_mcp_server() -> MCPServerStdio:
    """MCPサーバーを作成"""
    return MCPServerStdio(params={
        "command": "uvx",
        "args": [MCP_SERVER_PACKAGE]
    })

def create_agent(mcp_server: MCPServerStdio, custom_client: AsyncOpenAI) -> Agent:
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

async def safe_cleanup(server: Optional[MCPServerStdio]) -> None:
    """安全にMCPサーバーをクリーンアップ"""
    if server:
        try:
            await server.__aexit__(None, None, None)
        except Exception:
            pass



def create_error_response(request: Request, error: str) -> HTMLResponse:
    """エラーレスポンスを作成"""
    return templates.TemplateResponse("error_response.html", {
        "request": request,
        "error": error
    })

def get_tools_list() -> List[Dict[str, str]]:
    """ツール一覧を取得"""
    return [{"name": tool.name, "description": tool.description} for tool in available_tools]

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request) -> HTMLResponse:
    """メインページ"""
    config = {
        "endpoint": bool(AZURE_OPENAI_ENDPOINT),
        "deployment": bool(AZURE_OPENAI_DEPLOYMENT_NAME),
        "apiVersion": AZURE_OPENAI_API_VERSION,
        "mcpServer": MCP_SERVER_PACKAGE
    }
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "config": config,
        "available_tools": get_tools_list()
    })

@app.post("/api/chat")
async def chat(request: Request) -> HTMLResponse:
    """チャット処理"""
    try:
        # フォームデータを取得
        form_data = await request.form()
        message = form_data.get("message", "").strip()
        
        if not message:
            return create_error_response(request, "メッセージを入力してください。")
        
        if not agent:
            return create_error_response(request, "エージェントが初期化されていません。アプリケーションを再起動してください。")
        
        # エージェント実行
        result = await Runner.run(agent, message)
        
        response_data = {
            "content": getattr(result, 'final_output', str(result)),
            "tool_executions": extract_tool_executions(result),
            "usage": extract_usage_info(result)
        }
        
        return templates.TemplateResponse("chat_response.html", {
            "request": request,
            "response": response_data
        })
        
    except Exception as e:
        error_message = str(e)
        if "RateLimitReached" in error_message:
            error_detail = "レート制限に達しました。60秒後に再試行してください。"
        else:
            error_detail = f"実行エラー: {error_message}"
        
        return create_error_response(request, error_detail)

@app.get("/api/tools")
async def get_tools(request: Request) -> HTMLResponse:
    """利用可能なツール一覧を取得"""
    return templates.TemplateResponse("tools_list.html", {
        "request": request,
        "tools": get_tools_list()
    })

@app.get("/api/settings")
async def get_settings(request: Request) -> HTMLResponse:
    """設定状況を取得"""
    config = {
        "endpoint": bool(AZURE_OPENAI_ENDPOINT),
        "deployment": bool(AZURE_OPENAI_DEPLOYMENT_NAME),
        "apiVersion": AZURE_OPENAI_API_VERSION,
        "mcpServer": MCP_SERVER_PACKAGE
    }
    
    return templates.TemplateResponse("settings_modal.html", {
        "request": request,
        "config": config
    })

def extract_tool_executions(result) -> List[Dict[str, Any]]:
    """結果からツール実行情報を抽出"""
    tool_executions = []
    
    if not hasattr(result, 'raw_responses'):
        return tool_executions
    
    for raw_response in result.raw_responses:
        if not hasattr(raw_response, 'output'):
            continue
            
        for output_item in raw_response.output:
            if hasattr(output_item, 'name') and hasattr(output_item, 'arguments'):
                try:
                    args_dict = json.loads(output_item.arguments)
                except json.JSONDecodeError:
                    args_dict = output_item.arguments
                
                tool_executions.append({
                    'name': output_item.name,
                    'arguments': args_dict,
                    'call_id': getattr(output_item, 'call_id', 'unknown')
                })
    
    return tool_executions

def extract_usage_info(result) -> Dict[str, Any]:
    """結果から使用量情報を抽出"""
    usage_info = {"input": 0, "output": 0, "total": 0, "cached": 0}
    
    if not hasattr(result, 'raw_responses'):
        return usage_info
    
    for raw_response in result.raw_responses:
        if hasattr(raw_response, 'usage'):
            usage = raw_response.usage
            usage_info.update({
                "input": getattr(usage, 'input_tokens', 0),
                "output": getattr(usage, 'output_tokens', 0),
                "total": getattr(usage, 'total_tokens', 0)
            })
            
            # キャッシュされたトークン数を取得
            if (hasattr(usage, 'input_tokens_details') and 
                usage.input_tokens_details and
                hasattr(usage.input_tokens_details, 'cached_tokens')):
                usage_info["cached"] = usage.input_tokens_details.cached_tokens
            break
    
    return usage_info

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)