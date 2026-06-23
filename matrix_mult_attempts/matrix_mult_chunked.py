def min_plus_matmul(A, B, block=256):
    m, n = A.shape
    n2, p = B.shape
    assert n == n2

    C = torch.full(
        (m, p),
        float('inf'),
        device=A.device,
        dtype=A.dtype,
    )

    for k0 in range(0, n, block):
        k1 = min(k0 + block, n)

        partial = (
            A[:, k0:k1].unsqueeze(-1)      # (m, block, 1)
            + B[k0:k1, :].unsqueeze(0)     # (1, block, p)
        )

        C = torch.minimum(C, partial.amin(dim=1))

    return C
