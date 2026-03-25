class ArrowSpaceModelAdapter(TTimeProbabilisticModel):
    """
    Thin adapter around your ArrowSpaceProbabilisticModel to conform
    to the generic epiplexity interface.
    """

    def __init__(self, arrowspace_model, C0: int, k: int, b: int = 32):
        self.mdl = arrowspace_model
        self.C0 = C0
        self.k = k
        self.b = b

    def description_length_bits(self) -> float:
        # Use your existing PAS computation as S_T(X)
        return float(self.mdl.descriptionlengthbits(self.C0, self.k, self.b))

    def log_prob(self, x: np.ndarray) -> float:
        # evaluatelogprob already returns log P(x) in natural logs
        return float(self.mdl.evaluatelogprob(x))

    def sample(self, n: int = 1) -> np.ndarray:
        return self.mdl.sample(n)

    def raw_bits(self, X: np.ndarray) -> float:
        # Your notebook uses N * F * 32 bits for float baselines
        N, F = X.shape
        return float(N * F * self.b)