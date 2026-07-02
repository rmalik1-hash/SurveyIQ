def get_contradiction_pairs(n_questions: int) -> list[tuple[int, int]]:
    """Designate reverse-coded question index pairs for contradiction scoring.

    Roughly 1 pair per 10 questions, minimum 1 pair. Pairs are the first
    2 * n_pairs question indices, taken in order: (0,1), (2,3), ...
    """
    if n_questions < 4:
        raise ValueError("n_questions must be at least 4 to generate contradiction pairs")

    n_pairs = max(1, n_questions // 10)
    if n_pairs * 2 > n_questions:
        n_pairs = n_questions // 2

    return [(i * 2, i * 2 + 1) for i in range(n_pairs)]
