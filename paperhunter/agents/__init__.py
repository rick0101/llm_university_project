"""
paperhunter.agents — Multi-agent survey writing pipeline (LangGraph).

Exports:
    State & models:
        SurveyState, SectionOutline, SectionDraft, SectionStatus,
        CitationRecord, VerificationDetail, VerificationResult,
        PipelinePhase

    State helpers:
        create_initial_state, state_to_dict, state_from_dict,
        get_outline_from_state, get_sections_from_state,
        get_verifications_from_state, get_l2_summaries_from_state,
        get_version_from_state, get_provenance_from_state,
        get_bibliography_keys, collect_all_citation_keys,
        get_pipeline_summary

    Planner Agent:
        plan, generate_outline, plan_from_l2_summaries,
        load_outline_from_json, get_outline_stats

    Writer Agent:
        write, write_section, write_all_sections,
        write_sections_from_outline, get_section_stats

    Critic Agent:
        verify, verify_section, verify_all_sections,
        get_verification_stats

    Assembler Agent:
        assemble, assemble_paper, assemble_document,
        generate_abstract, generate_introduction, generate_conclusion,
        generate_bibliography, create_version, save_version,
        get_assembly_stats

    Pipeline orchestration:
        build_pipeline, run_pipeline, run_pipeline_from_checkpoint,
        save_checkpoint, load_checkpoint, get_latest_checkpoint,
        clean_checkpoints, list_checkpoints, get_pipeline_graph_description
"""

from paperhunter.agents.state import (
    # Enums
    PipelinePhase,
    SectionStatus,
    # Models
    CitationRecord,
    SectionDraft,
    SectionOutline,
    SurveyState,
    VerificationDetail,
    VerificationResult,
    # State helpers
    collect_all_citation_keys,
    create_initial_state,
    get_bibliography_keys,
    get_l2_summaries_from_state,
    get_outline_from_state,
    get_pipeline_summary,
    get_provenance_from_state,
    get_sections_from_state,
    get_verifications_from_state,
    get_version_from_state,
    state_from_dict,
    state_to_dict,
)
from paperhunter.agents.planner_agent import (
    generate_outline,
    get_outline_stats,
    load_outline_from_json,
    plan,
    plan_from_l2_summaries,
)
from paperhunter.agents.writer_agent import (
    get_section_stats,
    write,
    write_all_sections,
    write_section,
    write_sections_from_outline,
)
from paperhunter.agents.critic_agent import (
    get_verification_stats,
    verify,
    verify_all_sections,
    verify_section,
)
from paperhunter.agents.assembler_agent import (
    assemble,
    assemble_paper,
    assemble_document,
    generate_abstract,
    generate_bibliography,
    generate_conclusion,
    generate_introduction,
    create_version,
    save_version,
    get_assembly_stats,
)
from paperhunter.agents.pipeline import (
    build_pipeline,
    clean_checkpoints,
    get_latest_checkpoint,
    get_pipeline_graph_description,
    list_checkpoints,
    load_checkpoint,
    run_pipeline,
    run_pipeline_from_checkpoint,
    save_checkpoint,
)

__all__ = [
    # Enums
    "PipelinePhase",
    "SectionStatus",
    # Models
    "CitationRecord",
    "SectionDraft",
    "SectionOutline",
    "SurveyState",
    "VerificationDetail",
    "VerificationResult",
    # State helpers
    "collect_all_citation_keys",
    "create_initial_state",
    "get_bibliography_keys",
    "get_l2_summaries_from_state",
    "get_outline_from_state",
    "get_pipeline_summary",
    "get_provenance_from_state",
    "get_sections_from_state",
    "get_verifications_from_state",
    "get_version_from_state",
    "state_from_dict",
    "state_to_dict",
    # Planner Agent
    "generate_outline",
    "get_outline_stats",
    "load_outline_from_json",
    "plan",
    "plan_from_l2_summaries",
    # Writer Agent
    "get_section_stats",
    "write",
    "write_all_sections",
    "write_section",
    "write_sections_from_outline",
    # Critic Agent
    "get_verification_stats",
    "verify",
    "verify_all_sections",
    "verify_section",
    # Assembler Agent
    "assemble",
    "assemble_paper",
    "assemble_document",
    "generate_abstract",
    "generate_bibliography",
    "generate_conclusion",
    "generate_introduction",
    "create_version",
    "save_version",
    "get_assembly_stats",
    # Pipeline orchestration
    "build_pipeline",
    "run_pipeline",
    "run_pipeline_from_checkpoint",
    "save_checkpoint",
    "load_checkpoint",
    "get_latest_checkpoint",
    "clean_checkpoints",
    "list_checkpoints",
    "get_pipeline_graph_description",
]
