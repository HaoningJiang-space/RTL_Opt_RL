"""
Helper module for dynamic prompt selection based on data source
Supports both RTL and math tasks
"""

def get_system_prompts(data_source: str, max_num_turns: int):
    """
    Get appropriate system prompts based on data source and number of turns

    Args:
        data_source: Data source identifier
        max_num_turns: Number of turns in the conversation

    Returns:
        Dictionary containing system prompts for meta_thinking and reasoning roles
    """

    # Check if this is an RTL task
    rtl_sources = ['rtl_optimization', 'rtl_generation', 'rtl_math', 'verilog_optimization']
    is_rtl_task = data_source in rtl_sources or data_source.startswith('rtl_')

    if is_rtl_task:
        # Use RTL-specific prompts
        from prompt.rtl.multi_turn_rtl import RTL_MTA_SYSTEM_PROMPT, RTL_RA_SYSTEM_PROMPT
        return {
            'meta_thinking': RTL_MTA_SYSTEM_PROMPT,
            'reasoning': RTL_RA_SYSTEM_PROMPT
        }
    else:
        # Use math prompts based on number of turns
        if max_num_turns > 1:
            from prompt.math.multi_turn_mamrp import MTA_SYSTEM_PRMOPT, RA_SYSTEM_PRMOPT
        else:
            from prompt.math.single_turn_mamrp import MTA_SYSTEM_PRMOPT, RA_SYSTEM_PRMOPT

        return {
            'meta_thinking': MTA_SYSTEM_PRMOPT,
            'reasoning': RA_SYSTEM_PRMOPT
        }

def get_rollout_meta_info(data_source: str, max_num_turns: int, finish_flag: str):
    """
    Get rollout meta info with appropriate system prompts

    Args:
        data_source: Data source identifier
        max_num_turns: Number of turns in the conversation
        finish_flag: Finish flag for conversations

    Returns:
        Dictionary containing rollout meta information
    """

    system_prompts = get_system_prompts(data_source, max_num_turns)

    return {
        'agent_roles': ['meta_thinking', 'reasoning'],
        'finish_flag': finish_flag if max_num_turns > 1 else None,
        'system_prompts': system_prompts,
        'max_num_turns': max_num_turns
    }