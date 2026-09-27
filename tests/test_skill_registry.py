import pytest

from app.services.skill_registry import (
    KNOWLEDGE_BASE_QA,
    Skill,
    SkillNotFound,
    SkillRegistry,
)


def test_registry_returns_knowledge_base_skill() -> None:
    registry = SkillRegistry()

    skill = registry.get("knowledge_base_qa")

    assert skill == KNOWLEDGE_BASE_QA
    assert skill.allowed_tools == ("rag_search",)
    assert skill.tool_choice == "required"


def test_public_skill_list_excludes_instructions() -> None:
    registry = SkillRegistry()

    public_skills = registry.list_public()

    assert public_skills == [
        {
            "id": "knowledge_base_qa",
            "label": "RAG 检索",
            "description": "检索当前用户的知识库资料，并依据资料回答。",
        },
    ]
    assert "instructions" not in public_skills[0]


def test_registry_rejects_unknown_skill() -> None:
    registry = SkillRegistry()

    with pytest.raises(SkillNotFound, match="未知 Skill"):
        registry.get("does_not_exist")


def test_registry_rejects_duplicate_skill_ids() -> None:
    duplicate_skill = Skill(
        id="knowledge_base_qa",
        label="重复项",
        description="测试重复 ID",
        instructions="测试",
        allowed_tools=("rag_search",),
    )

    with pytest.raises(ValueError, match="重复的 Skill ID"):
        SkillRegistry(skills=(KNOWLEDGE_BASE_QA, duplicate_skill))
