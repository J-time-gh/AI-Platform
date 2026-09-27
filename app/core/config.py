from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用配置：从环境变量 / .env 文件读取，类型安全。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Enterprise AI Platform"
    app_version: str = "0.1.0"
    environment: str = "development"
    log_level: str = "INFO"

    database_url: str
    test_database_url: str
    redis_url: str
    # Embedding 向量缓存：7 天
    redis_embedding_ttl_seconds: int = 604800
    # SearchResult 缓存：60 秒
    redis_search_ttl_seconds: int = 60
    # Redis 连接与读写最长等待时间
    redis_socket_timeout_seconds: float = 1.0

    upload_dir: str = "data/uploads"
    embedding_model: str = "BAAI/bge-m3"
    llm_base_url: str = "http://127.0.0.1:1234/v1"
    llm_model: str = "qwen2.5-14b-instruct"
    llm_api_key: str = "lm-studio"

    llm_timeout: float = 120.0
    llm_temperature: float = 0.2

    embedding_device: str = "cuda"
    embedding_batch_size: int = 32
    hf_endpoint: str = "https://hf-mirror.com"
    hf_home: str = "~/.cache/huggingface"

    # 检索层门控：余弦相似度低于此值视为"没检索到相关资料"（仅 vector 模式）
    min_vector_score: float = 0.45

    # 精排层（阶段 4.3）：CrossEncoder 对候选块重新打分
    rerank_model: str = "BAAI/bge-reranker-base"
    rerank_max_length: int = 512
    # 精排分门控：0 = 关闭。先测排序质量，再用真实数据定阈值（别拍脑袋）
    min_rerank_score: float = 0.0

    # Agent
    agent_max_steps: int = 4
    # SQL 工具
    sql_max_rows: int = 100
    sql_timeout_ms: int = 3000
    # Python 计算工具
    # 限制用户传入的表达式长度 (在Sandbox中也需要调用这个)
    python_max_expression_length: int = 1000

    # Python Sandbox
    # AST 是表达式解析后的语法树。
    python_max_ast_nodes: int = 100
    # 限制乘方 限制表达式中数字的最大指数和绝对值，防止模型请求过大数值计算。
    python_max_exponent: int = 20
    # 限制最终结果的绝对值
    python_max_abs_value: float = 1e100

    # JWT 配置
    jwt_secret_key: SecretStr
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 30
    # 队列名称默认
    ingest_queue_name: str = "document-ingest"
    ingest_job_timeout_seconds: int = 900
    ingest_stuck_after_seconds: int = 1800
    ingest_error_max_length: int = 1000


settings = Settings()
