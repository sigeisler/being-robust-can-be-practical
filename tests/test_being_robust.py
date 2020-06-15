import numpy as np
import pytest

from being_robust import __version__, BeingRobust


def test_version():
    assert __version__ == '0.1.0'


class TestBeingRobust():

    def test_empty(self):
        X = np.ndarray((0, 100))

        with pytest.raises(ValueError):
            BeingRobust(random_state=42).fit(X)

    def test_output_shape(self):
        n_features = 100
        n_samples = 100000
        np.random.seed(15062020)
        X = np.random.randn(n_samples, n_features)
        X = np.append(X, 12 * np.ones((10, n_features)), axis=0)

        br = BeingRobust(random_state=42, keep_filtered=True).fit(X)
        assert br.location_.shape == (n_features,)
        assert br.filtered_.shape[0] <= n_samples
        assert br.filtered_.shape[1] == n_features

    def test_debug(self):
        n_features = 100
        X = np.random.randn(1000, n_features)

        BeingRobust(random_state=42, debug=True).fit(X)

    def test_naive_gaussian(self):
        n_features = 100
        np.random.seed(15062020)
        X = np.random.randn(100000, n_features)

        # Random (truncated) SVD
        br = BeingRobust(random_state=42).fit(X)
        np.allclose(br.location_, np.zeros(n_features), rtol=1e-3, atol=1e-5)

        # Full SVD
        br = BeingRobust(random_state=42, use_randomized_svd=False).fit(X)
        np.allclose(br.location_, np.zeros(n_features), rtol=1e-3, atol=1e-5)
