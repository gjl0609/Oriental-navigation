"""
东方处境导航仪 - 后端 API 服务
基于 FastAPI 框架，结合传统八字命理逻辑与大语言模型，输出分层级的“处境导航”解读。
"""

import os
from datetime import datetime
from typing import Optional, Dict, Any

import httpx
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from lunar_python import Solar

# main.py 顶部新增导入
from rag.embedding import EmbeddingManager
from rag.vectorstore import GujiVectorStore
from rag.bm25_index import GujiBM25Index
from rag.retriever import HybridGujiRetriever
from rag.guji_loader import load_guji_from_directory
from rag.prompts import build_rag_navigation_prompt

# 加载环境变量
load_dotenv()

# 在应用启动时初始化 RAG 组件
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. 嵌入模型
    embedding_manager = EmbeddingManager()

    # 2. 向量库
    vector_store = GujiVectorStore(
        persist_directory="./chroma_guji_db",
        embedding_manager=embedding_manager,
    )

    # 3. BM25 索引
    bm25_index = GujiBM25Index()

    # 4. 从 guji/ 目录加载古籍（首次运行会构建索引）
    guji_dir = os.path.join(os.path.dirname(__file__), "guji")
    if os.path.isdir(guji_dir):
        texts, metadatas, ids = load_guji_from_directory(guji_dir)
        # 构建向量索引（若已存在则可跳过）
        if vector_store.collection.count() == 0:
            vector_store.build_index(texts, metadatas, ids)
        # 构建 BM25 索引
        bm25_index.build_index(texts, metadatas, ids)

    # 5. 混合检索器
    app.state.retriever = HybridGujiRetriever(
        vector_store=vector_store,
        bm25_index=bm25_index,
        alpha=0.6,
        top_k=8,
    )
    print("✅ 服务启动完成")
    
    yield  # 这里是服务运行期间
    
    print("🛑 服务正在关闭...")

app = FastAPI(
    title="东方处境导航仪",
    description="根据出生时间与八字命理，提供三层漏斗式处境导航解读",
    version="1.1.0",
    lifespan=lifespan
)
# ==========================================
# 1. 数据模型定义
# ==========================================

class BaziRequest(BaseModel):
    """八字排盘与导航请求模型"""
    year: int = Field(..., ge=1900, le=2100, description="出生年份（公历）")
    month: int = Field(..., ge=1, le=12, description="出生月份（公历）")
    day: int = Field(..., ge=1, le=31, description="出生日期（公历）")
    hour: int = Field(..., ge=0, le=23, description="出生小时（24小时制，公历）")
    user_query: Optional[str] = Field(None, description="用户当下的困惑（可选）")
    
class BaziResponse(BaseModel):
    """处境导航响应模型"""
    status: str = Field(..., description="请求状态：success 或 error")
    message: str = Field(..., description="状态信息")
    bazi_info: Dict[str, Any] = Field(..., description="排盘基础信息")
    analysis_result: str = Field(..., description="AI 生成的三层解读内容")


# ==========================================
# 2. 八字排盘模块
# ==========================================

class BaziCalculator:
    """
    八字排盘工具类 (v2.0 - 基于 lunar-python 库)
      * 节气数据基于香港天文台实测数据，精度到日/时；
      * 年柱以立春为界；
      * 月柱以节气划分（立春→惊蛰为寅月，惊蛰→清明为卯月，……）；
      * 时柱按传统时辰 + 五鼠遁元法推算。
    用法: BaziCalculator.calculate(year, month, day, hour)
    """
    
    @classmethod
    def calculate(cls, year: int, month: int, day: int, hour: int) -> Dict[str, Any]:
        # 1. 使用 Solar 对象初始化公历日期
        solar = Solar.fromYmdHms(year, month, day, hour, 0, 0)
        lunar = solar.getLunar()
        
        # 2. 通过 lunar 对象获取八字信息
        eight_char = lunar.getEightChar()
        
        # 3. 提取所需数据
        return {
            "year_pillar": eight_char.getYear(),        # 年柱
            "month_pillar": eight_char.getMonth(),      # 月柱
            "day_pillar": eight_char.getDay(),          # 日柱
            "hour_pillar": eight_char.getTime(),        # 时柱
            "day_master": eight_char.getDayGan(),       # 日主天干
            "current_month_zhi": eight_char.getMonthZhi() # 月支
        }
    

# ==========================================
# 3. AI 提示词构建模块
# ==========================================

class PromptBuilder:
    """构建大模型 System Prompt 的工具类"""
    
    @staticmethod
    def build_navigation_prompt(bazi_info: Dict[str, Any], user_query: Optional[str]) -> str:
        """
        动态组装发送给大模型的 System Prompt。
        强制 AI 扮演“东方处境导航师”，严格遵循四条铁律与三层漏斗格式。
        """
        
        bazi_str = f"年柱:{bazi_info['year_pillar']}, 月柱:{bazi_info['month_pillar']}, 日柱:{bazi_info['day_pillar']}({bazi_info['day_master']}日主), 时柱:{bazi_info['hour_pillar']}"
        current_env = f"当前处于{bazi_info['current_month_zhi']}月令能量场中"
        
        query_instruction = ""
        if user_query:
            query_instruction = f"""
【用户当下困惑】：{user_query}
**特别注意**：用户提出了具体的困惑，你必须在每一层的解读中，将八字格局分析与该困惑深度结合，解释这种困惑在当下时间坐标中的结构性原因，绝不能给泛泛的运势描述！
"""
        
        prompt = f"""你是一位极具洞察力的“东方处境导航师”。你精通传统八字命理，但你的表达方式是现代的、去标签化的、充满人文关怀的。

【当前用户的八字排盘】：{bazi_str}
【当前时空坐标】：{current_env}
{query_instruction}

请严格基于上述信息，为用户生成处境导航解读。在解读中，你必须恪守以下【四条铁律】：
铁律一：永远只说“关系”，不说“实体”。禁止使用“你是X型人”“你天生就Y”等定义句式。必须使用“你当前与...之间，呈现出...的关系”“你正处在...的能量场中”等关系描述句式。
铁律二：永远带上“时令”，不说永恒。每一句解读都必须包含时间限定词（“最近”“这几个月”“这十年”），让用户感受到状态是阶段性的，而非永久标签。
铁律三：解释“生克制化”，不评判吉凶好坏。禁止输出“财运不好”“感情不顺”等吉凶判断。必须揭示内在的结构性运作机制（例如“你的财富与专业技能紧密相连，意味着收入更可能来自才华变现，而非固定工资”）。
铁律四：用“你是否发现...”结尾，把解释权还给用户。每层输出的结尾，必须以问句收尾，邀请用户自己验证，而非下定论。

请严格按照【三层漏斗】格式输出你的解读：

第一层（轻入口）：社交玩具与情绪嘴替。
要求：用一句话精准戳中用户当下的感受，风格类似“互联网嘴替”。必须完全去术语化，让人3秒内产生“这不就是我吗”的共鸣。最后用“你是否发现...？”结尾。

第二层（中度解惑）：处境导航与行动建议。
要求：用现代隐喻和能量解析，告诉用户当前处在什么样的关系格局里，并给出具体的行动窗口建议。禁止使用命理黑话，但需体现八字的动态思维。最后用“你是否发现...？”结尾。

第三层（重出口）：深度洞察与底层逻辑。
要求：适当保留传统术语（如正官、七杀、伤官、食神等），进行文化溯源和现实映射，揭示当前处境在人生大周期中的结构性位置，并给出风险预警。这部分是给想认真理解自己处境深层逻辑的用户看的。最后用“你是否发现...？”结尾。
"""
        return prompt


# ==========================================
# 4. 大模型调用模块
# ==========================================

class LLMClient:
    """通用大模型调用客户端，兼容 OpenAI 格式 API"""
    
    BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
    MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-3.5-turbo")
    TEMPERATURE = float(os.getenv("LLM_DEFAULT_TEMPERATURE", "0.7"))
    LLM_API_KEY = os.getenv("DEEPSEEK_API_KEY")

    @classmethod
    async def call_llm(cls, prompt: str) -> str:
        """异步调用 LLM API"""
        headers = {
            "Authorization": f"Bearer {cls.LLM_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": cls.MODEL_NAME,
            "messages": [
                {"role": "system", "content": prompt}
            ],
            "temperature": cls.TEMPERATURE
        }
        
        url = f"{cls.BASE_URL}/chat/completions"
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                result = response.json()
                return result["choices"][0]["message"]["content"]
        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="大模型 API 请求超时，请稍后重试")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                raise HTTPException(status_code=401, detail="API Key 无效或已过期")
            raise HTTPException(status_code=e.response.status_code, detail=f"大模型 API 报错: {e.response.text}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"调用大模型时发生未知错误: {str(e)}")


# ==========================================
# 5. API 接口层
# ==========================================

@app.post("/api/v1/bazi/navigate", response_model=BaziResponse)
async def navigate_bazi(request: BaziRequest):
    """
    东方处境导航核心接口
    接收请求 -> 排盘 -> RAG查询 -> 组装 Prompt -> 调用 LLM -> 返回结构化响应
    """
    try:
        # 1. 八字排盘（保持不变）
        bazi_info = BaziCalculator.calculate(
            year=request.year,
            month=request.month,
            day=request.day,
            hour=request.hour,
        )

        # 2. 构建 RAG 查询（简单示例：日主 + 月令 + 用户困惑）
        query_text = (
            f"日主 {bazi_info['day_master']} "
            f"月令 {bazi_info['current_month_zhi']} "
            f"{request.user_query or ''}"
        )

        # 3. 混合检索
        retriever: HybridGujiRetriever = app.state.retriever
        rag_result = retriever.retrieve(
            query_text=query_text,
            where=None,  # 初期不做元数据过滤
        )
        chunks = rag_result["chunks"]
        contradictions = rag_result["contradictions"]

        # 4. 构建 RAG 版 Prompt
        prompt = build_rag_navigation_prompt(
            bazi_info=bazi_info,
            user_query=request.user_query,
            chunks=chunks,
            contradictions=contradictions,
        )

        # 5. 调用 LLM
        analysis_result = await LLMClient.call_llm(
            prompt=prompt,
        )

        # 6. 返回结构化响应
        return BaziResponse(
            status="success",
            message="处境导航生成成功（RAG 增强）",
            bazi_info=bazi_info,
            analysis_result=analysis_result,
        )

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"服务内部错误: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    # 通过 uvicorn 直接运行 python main.py 即可启动服务
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)