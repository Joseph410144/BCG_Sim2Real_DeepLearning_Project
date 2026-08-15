import unittest

import torch

from Model.AlgorithmAwareFilter import AlgorithmAwareResidualFilter
from Model.AlgorithmAwareLoss import (
    HarmonicHeartRateLoss, HeartV6SurrogateLoss, SpectralReconstructionLoss,
    real_algorithm_aware_loss,
)


class AlgorithmAwareTests(unittest.TestCase):
    def test_filter_starts_as_identity(self):
        model = AlgorithmAwareResidualFilter(channels=8, blocks=2)
        signal = torch.randn(2, 1, 256)
        output, residual = model(signal, return_residual=True)
        torch.testing.assert_close(output, signal)
        torch.testing.assert_close(residual, torch.zeros_like(residual))

    def test_harmonic_loss_has_gradients(self):
        signal = torch.randn(2, 1, 1000, requires_grad=True)
        loss = HarmonicHeartRateLoss()(signal, torch.tensor([60.0, 80.0]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(signal.grad)

    def test_real_objective_backpropagates_to_model(self):
        model = AlgorithmAwareResidualFilter(channels=8, blocks=2)
        inputs = torch.randn(2, 1, 256)
        output = model(inputs)
        loss, _ = real_algorithm_aware_loss(output, inputs, torch.tensor([60.0, 75.0]),
                                             HarmonicHeartRateLoss())
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_v6_surrogate_prefers_target_periodicity(self):
        time = torch.arange(1000, dtype=torch.float32) / 100
        target = torch.sin(2 * torch.pi * 5 * time) * (1 + 0.8 * torch.sin(2 * torch.pi * time))
        wrong = torch.sin(2 * torch.pi * 5 * time) * (1 + 0.8 * torch.sin(2 * torch.pi * 1.6 * time))
        objective = HeartV6SurrogateLoss()
        target_loss = objective(target[None, None], torch.tensor([60.0]))
        wrong_loss = objective(wrong[None, None], torch.tensor([60.0]))
        self.assertLess(target_loss.item(), wrong_loss.item())

    def test_v6_surrogate_has_finite_gradients(self):
        signal = torch.randn(2, 1, 1000, requires_grad=True)
        loss = HeartV6SurrogateLoss()(signal, torch.tensor([60.0, 78.0]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(signal.grad).all())

    def test_spectral_loss_is_zero_for_equal_signals(self):
        signal = torch.randn(2, 1, 512)
        loss = SpectralReconstructionLoss((64, 128))(signal, signal)
        self.assertLess(loss.item(), 1e-6)


if __name__ == "__main__":
    unittest.main()
