def minplus_mm(A, B, block=256):
    m, n = A.shape
    n2, p = B.shape
    assert n == n2

    C = torch.full((m, p), float('inf'),
                   device=A.device,
                   dtype=A.dtype)

    for k0 in range(0, n, block):
        k1 = min(k0 + block, n)

        partial = (
            A[:, k0:k1, None] +
            B[k0:k1, None, :]
        )               # shape (m, block, p)

        C = torch.minimum(C, partial.amin(dim=1))

    return C
