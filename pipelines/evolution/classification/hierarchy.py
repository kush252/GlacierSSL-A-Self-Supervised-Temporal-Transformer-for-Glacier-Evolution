from utils.structures import GlacierState
from rule_engine import rules
from confidence.scorer import score_confidence

def classify_glacier(state: GlacierState):
    """
    Applies the decision hierarchy (Rule 8-A) and assigns a class to the state.
    Precedence: Disappearance > Fragmentation > Retreat > Advance > Stable
    """
    # 1. Base minimum epoch check
    if not rules.rule_2_3_A(state):
        state.assigned_class = "insufficient_evidence"
        state.confidence = "None"
        state.triggered_rules.append("2.3-A_fail")
        return

    # 2. Check Disappearance (Highest precedence)
    if rules.rule_7_4_A(state):
        state.assigned_class = "Disappearance"
        state.triggered_rules.append("7.4-A")
    
    # 3. Check Fragmentation
    elif rules.rule_5_5_B_fragmented(state):
        state.assigned_class = "Fragmentation"
        state.triggered_rules.append("5.5-B")

    # 4. Check Retreat
    elif rules.rule_4_4_A(state):
        state.assigned_class = "Retreat"
        state.triggered_rules.append("4.4-A")

    # 5. Check Advance
    elif rules.rule_6_4_A(state):
        state.assigned_class = "Advance"
        state.triggered_rules.append("6.4-A")
        
        # Additional check for surge
        if rules.rule_6_5_A(state):
            state.flags.append("possible_surge")

    # 6. Check Stable (Null class)
    elif rules.rule_3_4_A(state):
        state.assigned_class = "Stable"
        state.triggered_rules.append("3.4-A")
    
    else:
        # Fallback if no specific trend rules match but evidence is sufficient
        state.assigned_class = "Unknown"
        state.flags.append("ambiguous_trend")

    # Assign confidence score
    state.confidence = score_confidence(state)
    
    # Check MMU rule for fragments/disappearance flags
    if rules.rule_2_4_B(state):
        state.flags.append("insufficient_size_mmu")
