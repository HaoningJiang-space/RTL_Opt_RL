"""
RTL-specific generation module for ReMA multi-agent system
Integrates RTL system prompts with ReMA framework
"""

from typing import Dict
from prompt.rtl.multi_turn_rtl import RTL_MTA_SYSTEM_PROMPT, RTL_RA_SYSTEM_PROMPT

def get_rtl_system_prompts(data_source: str) -> Dict[str, str]:
    """
    Get RTL-specific system prompts based on data source

    Args:
        data_source: Data source identifier (e.g., 'rtl_optimization', 'rtl_generation')

    Returns:
        Dictionary containing system prompts for meta_thinking and reasoning roles
    """

    if data_source.startswith('rtl_') or data_source in ['verilog_optimization']:
        return {
            "meta_thinking": RTL_MTA_SYSTEM_PROMPT,
            "reasoning": RTL_RA_SYSTEM_PROMPT
        }
    else:
        # Fallback to default math prompts for non-RTL tasks
        from prompt.math.multi_turn_mamrp import MTA_SYSTEM_PRMOPT, RA_SYSTEM_PRMOPT
        return {
            "meta_thinking": MTA_SYSTEM_PRMOPT,
            "reasoning": RA_SYSTEM_PRMOPT
        }

def get_system_prompt_for_role(role: str, data_source: str) -> str:
    """
    Get system prompt for specific role and data source

    Args:
        role: Agent role ('meta_thinking' or 'reasoning')
        data_source: Data source identifier

    Returns:
        Appropriate system prompt string
    """

    prompts = get_rtl_system_prompts(data_source)
    return prompts.get(role, prompts["meta_thinking"])  # Default to meta_thinking if role not found

def is_rtl_task(data_source: str) -> bool:
    """
    Check if the task is RTL-related

    Args:
        data_source: Data source identifier

    Returns:
        True if RTL-related, False otherwise
    """

    rtl_sources = [
        'rtl_optimization',
        'rtl_generation',
        'rtl_math',
        'verilog_optimization'
    ]

    return data_source in rtl_sources or data_source.startswith('rtl_')