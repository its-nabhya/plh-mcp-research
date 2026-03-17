LOG_SCHEMA = {
    "interaction_id": str,
    "timestamp": str,
    "world": str,
    "model": str,
    
    "prompt": {
        "text": str,
        "inducement_strategy": str
    },
    
    "agent_reasoning": {
        "text": str,
        "confidence_score": float
    },
    
    "tool_call": {
        "name": str,
        "parameters": dict
    },
    
    "validation": {
        "tool_exists": bool,
        "schema_valid": bool,
        "permission_granted": bool,
        "violations": list
    },
    
    "world_response": {
        "status": str,
        "output": dict
    },
    
    "label": {
        "primary": str,
        "secondary": list
    }
}