from dataclasses import dataclass
from typing import Literal


class SkillNotFound(LookupError):
    """请求的 Skill 未注册。"""


@dataclass(frozen=True)
class Skill:
    id: str
    label: str
    description: str
    instructions: str
    allowed_tools: tuple[str, ...]
    tool_choice: Literal["auto", "required"] = "auto"


KNOWLEDGE_BASE_QA = Skill(
    id="knowledge_base_qa",
    label="RAG 检索",
    description="检索当前用户的知识库资料，并依据资料回答。",
    instructions=(
        "用户已选择知识库检索模式。回答前必须调用 rag_search。"
        "仅依据检索结果回答；引用相关资料来源。"
        "如果检索结果为空或不足，明确说明资料不足，不要编造。"
    ),
    allowed_tools=("rag_search",),
    tool_choice="required",
)


DEFAULT_SKILLS = (KNOWLEDGE_BASE_QA,)


class SkillRegistry:
    """管理可供 Agent 按需加载的 Skill。"""

    def __init__(
        self,
        skills: tuple[Skill, ...] = DEFAULT_SKILLS,
    ) -> None:
        self._skills: dict[str, Skill] = {}

        for skill in skills:
            if skill.id in self._skills:
                raise ValueError(f"重复的 Skill ID：{skill.id}")
            self._skills[skill.id] = skill

    def get(self, skill_id: str) -> Skill:
        try:
            return self._skills[skill_id]
        except KeyError as exc:
            raise SkillNotFound(f"未知 Skill：{skill_id}") from exc

    def list_public(self) -> list[dict[str, str]]:
        """提供给 UI 的目录信息，不返回内部 instructions。"""
        return [
            {
                "id": skill.id,
                "label": skill.label,
                "description": skill.description,
            }
            for skill in self._skills.values()
        ]


skill_registry = SkillRegistry()
