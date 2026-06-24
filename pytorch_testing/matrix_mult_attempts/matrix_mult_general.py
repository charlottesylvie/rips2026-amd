def min_plus_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Computes min-plus tropical matrix multiplication:

        C[i, j] = min_k A[i, k] + B[k, j]
    """

    # A has shape (m, n). Add a size-1 column dimension:
    # (m, n) -> (m, n, 1)
    # This lets A broadcast across output columns j.
    A_expanded = A.unsqueeze(-1)

    # B has shape (n, p). Add a size-1 row dimension:
    # (n, p) -> (1, n, p)
    # This lets B broadcast across output rows i.
    B_expanded = B.unsqueeze(-3)

    # Broadcasted addition gives shape (m, n, p).
    # partial[i, k, j] = A[i, k] + B[k, j]
    partial = A_expanded + B_expanded

    # Take min over k, the middle dimension.
    # This gives C[i, j] = min_k A[i, k] + B[k, j].
    # C = torch.min(partial, dim=-2).values
    C = torch.amin(partial, dim=-2)


