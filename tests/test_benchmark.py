import unittest
from pathlib import Path

import numpy as np

import model
from diloco.distributed import TensorLayout, TrainingConfig, train_distributed
from diloco.optim import adamw_step, init_adamw_state
from diloco.workload import character_features, load_character_data


class ModelTests(unittest.TestCase):
    def test_backprop_matches_finite_differences(self):
        params = model.init_model_params(3, 4, 2, seed=4)
        rng = np.random.default_rng(7)
        x = rng.normal(size=(5, 3))
        y = np.array([0, 1, 1, 0, 0])
        _, cache = model.model_forward(params, x)
        gradients = model.model_backward(params, cache, y)

        delta = 1e-6
        for name, values in params.items():
            numerical = np.zeros_like(values)
            for index in np.ndindex(values.shape):
                original = values[index]
                values[index] = original + delta
                upper = model.evaluate_loss(params, x, y)
                values[index] = original - delta
                lower = model.evaluate_loss(params, x, y)
                values[index] = original
                numerical[index] = (upper - lower) / (2 * delta)
            np.testing.assert_allclose(gradients[name], numerical, atol=1e-9, rtol=1e-6)

    def test_cross_entropy_preserves_large_wrong_class_penalty(self):
        loss = model.cross_entropy_loss(np.array([[1_000.0, -1_000.0]]), np.array([1]))
        self.assertEqual(loss, 2_000.0)

    def test_adamw_decays_the_preupdate_weights(self):
        params = {"weight": np.array([2.0])}
        state = init_adamw_state(params)
        changed = adamw_step(params, state, {"weight": np.array([1.0])}, 0.1, 0, 0, 0, 0.2)
        np.testing.assert_allclose(changed["weight"], [1.86])
        np.testing.assert_array_equal(params["weight"], [2.0])
        self.assertEqual(state["t"], 1)

    def test_worker_validation_is_held_out_and_shards_are_disjoint(self):
        fixture = Path(__file__).parent / "fixtures" / "letters.txt"
        workload = load_character_data(fixture, 2, 1_000, 1_000)
        self.assertEqual(len(workload.train[1]), workload.training_characters - 2)
        self.assertEqual(len(workload.holdout[1]), workload.held_out_characters - 2)
        indices = np.arange(12).reshape(-1, 1)
        y = np.arange(12) % 4
        for sharder, args in ((model.iid_shard_dataset, (2,)),
                              (model.noniid_shard_dataset, (2, 4))):
            shards = sharder(indices, y, *args, seed=0)
            self.assertEqual(sorted(np.concatenate([x[:, 0] for x, _ in shards])), list(range(12)))
            for x_shard, y_shard in shards:
                np.testing.assert_array_equal(y[x_shard[:, 0]], y_shard)
        skewed = model.noniid_shard_dataset(indices, y, 2, 4)
        self.assertEqual([set(labels) for _, labels in skewed], [{0, 2}, {1, 3}])


class ProtocolTests(unittest.TestCase):
    def test_tensor_payload_round_trip(self):
        initial = model.init_model_params(3, 4, 2)
        layout = TensorLayout.from_params(initial)
        encoded = layout.pack(initial)
        self.assertEqual(len(encoded), 8 * sum(a.size for a in initial.values()))
        for name, array in layout.unpack(encoded).items():
            np.testing.assert_array_equal(array, initial[name])
        with self.assertRaises(ValueError):
            layout.unpack(encoded[:-1])

    def test_process_runs_match_compute_and_measure_successful_transfers(self):
        x = np.array([[1, 2], [2, 1], [1, 2], [2, 1]])
        y = np.array([1, 2, 1, 2])
        workers = [(x[0:2], y[0:2]), (x[2:], y[2:])]
        initial = model.init_model_params(6, 8, 3)
        size = TensorLayout.from_params(initial).nbytes
        config = TrainingConfig(steps=8, batch_size=2, eval_every=4)
        _, sync = train_distributed(initial, workers, (x, y), 3, config, None, 0)
        trained, local = train_distributed(initial, workers, (x, y), 3, config, 4, 0)

        self.assertEqual([r["local_steps"] for r in local], [0, 4, 8])
        self.assertEqual([r["examples_seen"] for r in sync], [r["examples_seen"] for r in local])
        self.assertEqual(local[0], sync[0])
        self.assertEqual(sync[-1]["model_payload_bytes"], 2 * 2 * 8 * size)
        self.assertEqual(local[-1]["model_payload_bytes"], 2 * 2 * 2 * size)
        self.assertEqual(local[-1]["sent_model_bytes"], local[-1]["received_model_bytes"])
        self.assertLess(local[-1]["val_loss"], local[0]["val_loss"])
        self.assertAlmostEqual(
            model.evaluate_loss(trained, character_features(x, 3), y), local[-1]["val_loss"]
        )

    def test_reference_nesterov_accepts_flat_and_wrapped_momentum(self):
        params = {"weight": np.array([1.0])}
        state = model.init_outer_optimizer(params)
        grad = {"weight": np.array([0.2])}
        model.update_outer_momentum(state, grad, 0.9)
        wrapped = model.nesterov_param_update(params, state, grad, 0.7, 0.9)
        flat = model.nesterov_param_update(params, state["momentum"], grad, 0.7, 0.9)
        np.testing.assert_allclose(wrapped["weight"], [0.734])
        np.testing.assert_array_equal(wrapped["weight"], flat["weight"])


if __name__ == "__main__":
    unittest.main()
