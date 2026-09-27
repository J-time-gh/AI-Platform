# Frontend MVP

这是当前 FastAPI 后端配套的无构建依赖前端，页面由 FastAPI 在 `/ui/` 路径提供，不需要 Node.js 或 npm。

## 启动

在项目根目录运行：

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

然后打开：<http://127.0.0.1:8000/ui/>

## 已实现

- JWT 登录、注册、退出
- 文档上传（`202 Accepted`）
- `pending / processing / ready / failed` 状态轮询
- 当前用户文档列表与删除
- 向量、BM25、混合、混合重排四种搜索模式
- 搜索结果的文件名、分数、内容和分块信息展示
- 服务端持久化聊天会话：会话列表、历史记录恢复、新建与删除对话

前端与 API 同源部署，因此不需要额外的 CORS 配置。访问令牌仅保存在浏览器的 localStorage 中，属于 MVP 方案；生产环境应进一步评估更安全的会话存储策略。
