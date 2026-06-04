def resolve_nextham_uureal_mask(
    nextham_uureal_mask: bool = False,
    full_soc_prediction: bool = False,
) -> bool:
    """Resolve compact uu_real masking against explicit full-SOC opt-in."""
    if bool(full_soc_prediction):
        return False
    return bool(nextham_uureal_mask)
