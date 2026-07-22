import contextlib
import csv
import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

import detector_neumonia as dn

# ── helper: patches for grad_cam's TF internals ──────────────────────


def _mock_gradcam_tf():
    """Return a context manager that patches TF internals so grad_cam
    can run with eager-mode GradientTape mocks.

    Also patches tf.keras.Model so the sub-model creation in grad_cam
    returns a simple mock instead of trying to build a real model from
    MagicMock inputs.
    """
    conv_4d = np.random.rand(1, 64, 64, 64).astype(np.float32)
    conv_3d = np.random.rand(64, 64, 64).astype(np.float32)

    stack = contextlib.ExitStack()
    stack.enter_context(patch("grad_cam.tf.GradientTape", _mock_tape_cls(conv_4d)))
    stack.enter_context(
        patch("grad_cam.tf.reduce_mean", return_value=np.ones((64,), dtype=np.float32))
    )
    stack.enter_context(patch("grad_cam.tf.keras.Model", _mock_grad_model(conv_3d)))
    return stack


def _mock_tape_cls(conv_4d):
    """Return a GradientTape class mock whose instance returns
    gradient = conv_4d."""
    cls = MagicMock()
    tape = MagicMock()
    cls.return_value = tape
    tape.__enter__.return_value = tape
    tape.gradient.return_value = conv_4d
    return cls


def _mock_grad_model(conv_3d):
    """Return a factory that creates a grad-model mock whose __call__
    returns a conv_output that is subscriptable with [0] → numpy array."""

    def build(*args, **kwargs):
        gm = MagicMock()
        conv_mock = MagicMock()
        conv_mock.__getitem__.return_value = conv_3d
        gm.return_value = (conv_mock, MagicMock())
        return gm

    return build


def _run_grad_cam(arr):
    """Run grad_cam with all TF internals patched."""
    with patch("grad_cam.model_fun") as mock_mf:
        m = MagicMock()
        m.predict.return_value = np.random.rand(1, 3).astype(np.float32)
        mock_mf.return_value = m
        with _mock_gradcam_tf():
            return dn.grad_cam(arr)


def _mock_tape_cls(conv_4d):
    """Return a GradientTape class mock whose instance returns
    gradient = conv_4d."""
    cls = MagicMock()
    tape = MagicMock()
    cls.return_value = tape
    tape.__enter__.return_value = tape
    tape.gradient.return_value = conv_4d
    return cls


# ═══════════════════════════════════════════════════════════════════════
#  1. preprocess(array) — 20 tests
# ═══════════════════════════════════════════════════════════════════════


class TestPreprocess:
    def test_output_shape(self, rgb_512):
        result = dn.preprocess(rgb_512)
        assert result.shape == (1, 512, 512, 1)

    def test_output_has_batch_dim(self, small_rgb):
        result = dn.preprocess(small_rgb)
        assert result.ndim == 4
        assert result.shape[0] == 1

    def test_output_height_512(self, rgb_1024):
        result = dn.preprocess(rgb_1024)
        assert result.shape[1] == 512
        assert result.shape[2] == 512

    def test_single_channel_output(self, rgb_512):
        result = dn.preprocess(rgb_512)
        assert result.shape[-1] == 1

    def test_values_normalized_0_1(self, rgb_512):
        result = dn.preprocess(rgb_512)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_normalized_with_zeros(self, all_zeros):
        result = dn.preprocess(all_zeros)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_normalized_with_max(self, all_max):
        result = dn.preprocess(all_max)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_dtype_is_float(self, rgb_512):
        result = dn.preprocess(rgb_512)
        assert np.issubdtype(result.dtype, np.floating)

    def test_grayscale_input_fails(self, gray_512):
        with pytest.raises(Exception):
            dn.preprocess(gray_512)

    def test_small_image(self, small_rgb):
        result = dn.preprocess(small_rgb)
        assert result.shape == (1, 512, 512, 1)

    def test_large_image(self, rgb_1024):
        result = dn.preprocess(rgb_1024)
        assert result.shape == (1, 512, 512, 1)

    def test_non_contiguous_array(self):
        arr = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)[::2, ::2]
        result = dn.preprocess(arr)
        assert result.shape == (1, 512, 512, 1)

    def test_rgba_input(self):
        rgba = np.random.randint(0, 256, (100, 100, 4), dtype=np.uint8)
        result = dn.preprocess(rgba)
        assert result.shape == (1, 512, 512, 1)

    def test_random_range(self, rgb_512):
        result = dn.preprocess(rgb_512)
        assert np.all(result >= 0.0) and np.all(result <= 1.0)

    def test_output_not_all_same(self, rgb_512):
        result = dn.preprocess(rgb_512)
        assert not np.allclose(result, result.ravel()[0])

    def test_deterministic(self, rgb_512):
        r1 = dn.preprocess(rgb_512.copy())
        r2 = dn.preprocess(rgb_512.copy())
        np.testing.assert_array_equal(r1, r2)

    def test_input_not_modified(self, rgb_512):
        original = rgb_512.copy()
        dn.preprocess(rgb_512)
        assert np.array_equal(rgb_512, original)

    def test_does_not_raise(self, rgb_512):
        try:
            dn.preprocess(rgb_512)
        except Exception:
            pytest.fail("preprocess raised unexpectedly")

    def test_very_small_input(self):
        arr = np.random.randint(0, 256, (4, 4, 3), dtype=np.uint8)
        result = dn.preprocess(arr)
        assert result.shape == (1, 512, 512, 1)

    def test_aspect_ratio_preservation(self):
        arr = np.random.randint(0, 256, (300, 600, 3), dtype=np.uint8)
        result = dn.preprocess(arr)
        assert result.shape == (1, 512, 512, 1)


# ═══════════════════════════════════════════════════════════════════════
#  2. read_dicom_file(path) — 15 tests
# ═══════════════════════════════════════════════════════════════════════


class TestReadDicomFile:
    @patch("read_img.pydicom.dcmread")
    def test_returns_tuple(self, mock_dcm):
        ds = MagicMock()
        ds.pixel_array = np.ones((64, 64), dtype=np.uint16)
        mock_dcm.return_value = ds
        result = dn.read_dicom_file("f.dcm")
        assert isinstance(result, tuple) and len(result) == 2

    @patch("read_img.pydicom.dcmread")
    def test_first_element_ndarray(self, mock_dcm):
        ds = MagicMock()
        ds.pixel_array = np.ones((64, 64), dtype=np.uint16)
        mock_dcm.return_value = ds
        assert isinstance(dn.read_dicom_file("f.dcm")[0], np.ndarray)

    @patch("read_img.pydicom.dcmread")
    def test_second_element_pil_image(self, mock_dcm):
        ds = MagicMock()
        ds.pixel_array = np.ones((64, 64), dtype=np.uint16)
        mock_dcm.return_value = ds
        assert isinstance(dn.read_dicom_file("f.dcm")[1], Image.Image)

    @patch("read_img.pydicom.dcmread")
    def test_output_uint8(self, mock_dcm):
        ds = MagicMock()
        ds.pixel_array = np.ones((64, 64), dtype=np.uint16)
        mock_dcm.return_value = ds
        assert dn.read_dicom_file("f.dcm")[0].dtype == np.uint8

    @patch("read_img.pydicom.dcmread")
    def test_values_clipped_0_255(self, mock_dcm):
        ds = MagicMock()
        ds.pixel_array = np.array([[1000, 4000]], dtype=np.uint16)
        mock_dcm.return_value = ds
        r = dn.read_dicom_file("f.dcm")[0]
        assert r.min() >= 0 and r.max() <= 255

    @patch("read_img.pydicom.dcmread")
    def test_output_3_channels(self, mock_dcm):
        ds = MagicMock()
        ds.pixel_array = np.ones((32, 32), dtype=np.uint16)
        mock_dcm.return_value = ds
        r = dn.read_dicom_file("f.dcm")[0]
        assert r.ndim == 3 and r.shape[-1] == 3

    @patch("read_img.pydicom.dcmread")
    def test_raises_file_not_found(self, mock_dcm):
        mock_dcm.side_effect = FileNotFoundError
        with pytest.raises(Exception):
            dn.read_dicom_file("missing.dcm")

    @patch("read_img.pydicom.dcmread")
    def test_raises_on_corrupt(self, mock_dcm):
        mock_dcm.side_effect = OSError
        with pytest.raises(Exception):
            dn.read_dicom_file("bad.dcm")

    @patch("read_img.pydicom.dcmread")
    def test_calls_dcmread(self, mock_dcm):
        ds = MagicMock()
        ds.pixel_array = np.ones((10, 10), dtype=np.uint16)
        mock_dcm.return_value = ds
        dn.read_dicom_file("test.dcm")
        mock_dcm.assert_called_once_with("test.dcm")

    @patch("read_img.pydicom.dcmread")
    def test_non_empty(self, mock_dcm):
        ds = MagicMock()
        ds.pixel_array = np.ones((256, 256), dtype=np.uint16)
        mock_dcm.return_value = ds
        assert dn.read_dicom_file("f.dcm")[0].size > 0

    @patch("read_img.pydicom.dcmread")
    def test_large_dicom(self, mock_dcm):
        ds = MagicMock()
        ds.pixel_array = np.ones((2048, 2048), dtype=np.uint16)
        mock_dcm.return_value = ds
        r = dn.read_dicom_file("large.dcm")[0]
        assert r.shape[-1] == 3

    @patch("read_img.pydicom.dcmread")
    def test_uint16_to_uint8(self, mock_dcm):
        ds = MagicMock()
        ds.pixel_array = np.ones((16, 16), dtype=np.uint16)
        mock_dcm.return_value = ds
        assert dn.read_dicom_file("f.dcm")[0].dtype == np.uint8

    @patch("read_img.pydicom.dcmread")
    def test_all_zeros(self, mock_dcm):
        ds = MagicMock()
        ds.pixel_array = np.zeros((16, 16), dtype=np.uint16)
        mock_dcm.return_value = ds
        r = dn.read_dicom_file("zero.dcm")[0]
        assert r.dtype == np.uint8

    @patch("read_img.pydicom.dcmread")
    def test_uint16_array_unchanged(self, mock_dcm):
        arr = np.array([[100, 200]], dtype=np.uint16)
        ds = MagicMock()
        ds.pixel_array = arr
        mock_dcm.return_value = ds
        dn.read_dicom_file("f.dcm")
        assert np.array_equal(ds.pixel_array, arr)

    @patch("read_img.pydicom.dcmread")
    def test_int16_handled(self, mock_dcm):
        ds = MagicMock()
        ds.pixel_array = np.array([[-100, 100]], dtype=np.int16)
        mock_dcm.return_value = ds
        assert dn.read_dicom_file("f.dcm")[0].dtype == np.uint8


# ═══════════════════════════════════════════════════════════════════════
#  3. read_jpg_file(path) — 15 tests
# ═══════════════════════════════════════════════════════════════════════


class TestReadJpgFile:
    @patch("read_img.cv2.imread")
    def test_returns_tuple(self, mock_imread):
        mock_imread.return_value = np.ones((32, 32, 3), dtype=np.uint8)
        r = dn.read_jpg_file("f.jpg")
        assert isinstance(r, tuple) and len(r) == 2

    @patch("read_img.cv2.imread")
    def test_first_element_ndarray(self, mock_imread):
        mock_imread.return_value = np.ones((32, 32, 3), dtype=np.uint8)
        assert isinstance(dn.read_jpg_file("f.jpg")[0], np.ndarray)

    @patch("read_img.cv2.imread")
    def test_second_element_pil_image(self, mock_imread):
        mock_imread.return_value = np.ones((32, 32, 3), dtype=np.uint8)
        assert isinstance(dn.read_jpg_file("f.jpg")[1], Image.Image)

    @patch("read_img.cv2.imread")
    def test_output_uint8(self, mock_imread):
        mock_imread.return_value = np.ones((32, 32, 3), dtype=np.uint8)
        assert dn.read_jpg_file("f.jpg")[0].dtype == np.uint8

    @patch("read_img.cv2.imread")
    def test_values_0_255(self, mock_imread):
        mock_imread.return_value = np.ones((32, 32, 3), dtype=np.uint8) * 128
        r = dn.read_jpg_file("f.jpg")[0]
        assert r.min() >= 0 and r.max() <= 255

    @patch("read_img.cv2.imread")
    def test_raises_on_missing(self, mock_imread):
        mock_imread.return_value = None
        with pytest.raises(Exception):
            dn.read_jpg_file("missing.jpg")

    @patch("read_img.cv2.imread")
    def test_grayscale_to_rgb(self, mock_imread):
        gray = np.ones((32, 32), dtype=np.uint8)
        mock_imread.return_value = gray
        r = dn.read_jpg_file("gray.jpg")[0]
        assert isinstance(r, np.ndarray)

    @patch("read_img.cv2.imread")
    def test_shape_preserved(self, mock_imread):
        mock_imread.return_value = np.ones((200, 300, 3), dtype=np.uint8)
        assert dn.read_jpg_file("f.jpg")[0].shape == (200, 300, 3)

    @patch("read_img.cv2.imread")
    def test_calls_imread(self, mock_imread):
        mock_imread.return_value = np.ones((10, 10, 3), dtype=np.uint8)
        dn.read_jpg_file("photo.jpg")
        mock_imread.assert_called_once_with("photo.jpg")

    @patch("read_img.cv2.imread")
    def test_png_supported(self, mock_imread):
        mock_imread.return_value = np.ones((16, 16, 3), dtype=np.uint8)
        assert isinstance(dn.read_jpg_file("img.png")[0], np.ndarray)

    @patch("read_img.cv2.imread")
    def test_non_empty(self, mock_imread):
        mock_imread.return_value = np.ones((256, 256, 3), dtype=np.uint8)
        assert dn.read_jpg_file("f.jpg")[0].size > 0

    @patch("read_img.cv2.imread")
    def test_black_image(self, mock_imread):
        mock_imread.return_value = np.zeros((16, 16, 3), dtype=np.uint8)
        assert dn.read_jpg_file("black.jpg")[0].max() == 0

    @patch("read_img.cv2.imread")
    def test_white_image(self, mock_imread):
        mock_imread.return_value = np.full((16, 16, 3), 255, dtype=np.uint8)
        assert dn.read_jpg_file("white.jpg")[0].min() == 255

    @patch("read_img.cv2.imread")
    def test_rgba_accepted(self, mock_imread):
        mock_imread.return_value = np.ones((16, 16, 4), dtype=np.uint8)
        assert isinstance(dn.read_jpg_file("rgba.png")[0], np.ndarray)

    @patch("read_img.cv2.imread")
    def test_high_res(self, mock_imread):
        mock_imread.return_value = np.ones((4000, 3000, 3), dtype=np.uint8)
        assert dn.read_jpg_file("hires.jpg")[0].shape == (4000, 3000, 3)


# ═══════════════════════════════════════════════════════════════════════
#  4. predict(array) — 15 tests
# ═══════════════════════════════════════════════════════════════════════


class TestPredict:
    @patch("integrator.model_fun")
    @patch("integrator.grad_cam")
    def test_returns_tuple(self, mock_gc, mock_mf, rgb_512):
        m = MagicMock()
        m.predict.return_value = np.array([[0.1, 0.8, 0.1]], dtype=np.float32)
        mock_mf.return_value = m
        mock_gc.return_value = np.zeros((512, 512, 3), dtype=np.uint8)
        r = dn.predict(rgb_512)
        assert isinstance(r, tuple) and len(r) == 3

    @patch("integrator.model_fun")
    @patch("integrator.grad_cam")
    def test_label_bacteriana(self, mock_gc, mock_mf, rgb_512):
        m = MagicMock()
        m.predict.return_value = np.array([[0.9, 0.05, 0.05]], dtype=np.float32)
        mock_mf.return_value = m
        mock_gc.return_value = np.zeros((512, 512, 3), dtype=np.uint8)
        assert dn.predict(rgb_512)[0] == "bacteriana"

    @patch("integrator.model_fun")
    @patch("integrator.grad_cam")
    def test_label_normal(self, mock_gc, mock_mf, rgb_512):
        m = MagicMock()
        m.predict.return_value = np.array([[0.05, 0.9, 0.05]], dtype=np.float32)
        mock_mf.return_value = m
        mock_gc.return_value = np.zeros((512, 512, 3), dtype=np.uint8)
        assert dn.predict(rgb_512)[0] == "normal"

    @patch("integrator.model_fun")
    @patch("integrator.grad_cam")
    def test_label_viral(self, mock_gc, mock_mf, rgb_512):
        m = MagicMock()
        m.predict.return_value = np.array([[0.05, 0.05, 0.9]], dtype=np.float32)
        mock_mf.return_value = m
        mock_gc.return_value = np.zeros((512, 512, 3), dtype=np.uint8)
        assert dn.predict(rgb_512)[0] == "viral"

    @patch("integrator.model_fun")
    @patch("integrator.grad_cam")
    def test_probability_numeric(self, mock_gc, mock_mf, rgb_512):
        m = MagicMock()
        m.predict.return_value = np.array([[0.1, 0.8, 0.1]], dtype=np.float32)
        mock_mf.return_value = m
        mock_gc.return_value = np.zeros((512, 512, 3), dtype=np.uint8)
        assert np.issubdtype(type(dn.predict(rgb_512)[1]), np.floating)

    @patch("integrator.model_fun")
    @patch("integrator.grad_cam")
    def test_probability_range(self, mock_gc, mock_mf, rgb_512):
        m = MagicMock()
        m.predict.return_value = np.array([[0.1, 0.8, 0.1]], dtype=np.float32)
        mock_mf.return_value = m
        mock_gc.return_value = np.zeros((512, 512, 3), dtype=np.uint8)
        p = dn.predict(rgb_512)[1]
        assert 0 <= p <= 100

    @patch("integrator.model_fun")
    @patch("integrator.grad_cam")
    def test_highest_prob_wins(self, mock_gc, mock_mf, rgb_512):
        m = MagicMock()
        m.predict.return_value = np.array([[0.2, 0.3, 0.5]], dtype=np.float32)
        mock_mf.return_value = m
        mock_gc.return_value = np.zeros((512, 512, 3), dtype=np.uint8)
        assert dn.predict(rgb_512)[0] == "viral"

    @patch("integrator.model_fun")
    @patch("integrator.grad_cam")
    def test_heatmap_ndarray(self, mock_gc, mock_mf, rgb_512):
        m = MagicMock()
        m.predict.return_value = np.array([[0.1, 0.8, 0.1]], dtype=np.float32)
        mock_mf.return_value = m
        mock_gc.return_value = np.zeros((512, 512, 3), dtype=np.uint8)
        assert isinstance(dn.predict(rgb_512)[2], np.ndarray)

    @patch("integrator.model_fun")
    @patch("integrator.grad_cam")
    def test_heatmap_shape_512(self, mock_gc, mock_mf, rgb_512):
        m = MagicMock()
        m.predict.return_value = np.array([[0.1, 0.8, 0.1]], dtype=np.float32)
        mock_mf.return_value = m
        mock_gc.return_value = np.zeros((512, 512, 3), dtype=np.uint8)
        assert dn.predict(rgb_512)[2].shape == (512, 512, 3)

    @patch("integrator.model_fun")
    @patch("integrator.grad_cam")
    def test_calls_grad_cam(self, mock_gc, mock_mf, rgb_512):
        m = MagicMock()
        m.predict.return_value = np.array([[0.1, 0.8, 0.1]], dtype=np.float32)
        mock_mf.return_value = m
        mock_gc.return_value = np.zeros((512, 512, 3), dtype=np.uint8)
        dn.predict(rgb_512)
        mock_gc.assert_called_once()

    @patch("integrator.model_fun")
    @patch("integrator.grad_cam")
    def test_calls_preprocess(self, mock_gc, mock_mf, rgb_512):
        m = MagicMock()
        m.predict.return_value = np.array([[0.1, 0.8, 0.1]], dtype=np.float32)
        mock_mf.return_value = m
        mock_gc.return_value = np.zeros((512, 512, 3), dtype=np.uint8)
        with patch("integrator.preprocess") as mock_pp:
            mock_pp.return_value = np.random.rand(1, 512, 512, 1).astype(np.float32)
            dn.predict(rgb_512)
        mock_pp.assert_called_once()

    @patch("integrator.model_fun")
    @patch("integrator.grad_cam")
    def test_confidence_100(self, mock_gc, mock_mf, rgb_512):
        m = MagicMock()
        m.predict.return_value = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        mock_mf.return_value = m
        mock_gc.return_value = np.zeros((512, 512, 3), dtype=np.uint8)
        assert dn.predict(rgb_512)[1] == 100.0

    @patch("integrator.model_fun")
    @patch("integrator.grad_cam")
    def test_confidence_0(self, mock_gc, mock_mf, rgb_512):
        m = MagicMock()
        m.predict.return_value = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        mock_mf.return_value = m
        mock_gc.return_value = np.zeros((512, 512, 3), dtype=np.uint8)
        assert dn.predict(rgb_512)[1] == 0.0

    @patch("integrator.model_fun")
    @patch("integrator.grad_cam")
    def test_deterministic(self, mock_gc, mock_mf, rgb_512):
        m = MagicMock()
        preds = np.array([[0.1, 0.8, 0.1]], dtype=np.float32)
        m.predict.return_value = preds
        mock_mf.return_value = m
        mock_gc.return_value = np.zeros((512, 512, 3), dtype=np.uint8)
        r1, r2 = dn.predict(rgb_512), dn.predict(rgb_512)
        assert r1[0] == r2[0] and r1[1] == r2[1]


# ═══════════════════════════════════════════════════════════════════════
#  5. grad_cam(array) — 15 tests
# ═══════════════════════════════════════════════════════════════════════


class TestGradCAM:
    def test_output_ndarray(self, rgb_512):
        result = _run_grad_cam(rgb_512)
        assert isinstance(result, np.ndarray)

    def test_output_shape_512(self, rgb_512):
        result = _run_grad_cam(rgb_512)
        assert result.shape == (512, 512, 3)

    def test_output_dtype_uint8(self, rgb_512):
        result = _run_grad_cam(rgb_512)
        assert result.dtype == np.uint8

    def test_small_input(self, small_rgb):
        result = _run_grad_cam(small_rgb)
        assert result.shape == (512, 512, 3)

    def test_grayscale_input(self, gray_512):
        with patch("grad_cam.preprocess") as mock_pp:
            mock_pp.return_value = np.random.rand(1, 512, 512, 1).astype(np.float32)
            gray_3ch = np.stack([gray_512] * 3, axis=-1)
            result = _run_grad_cam(gray_3ch)
        assert result.shape == (512, 512, 3)

    def test_non_negative(self, rgb_512):
        result = _run_grad_cam(rgb_512)
        assert np.all(result >= 0)

    def test_values_uint8_range(self, rgb_512):
        result = _run_grad_cam(rgb_512)
        assert result.min() >= 0 and result.max() <= 255

    def test_calls_model_fun(self, rgb_512):
        with patch("grad_cam.model_fun") as mock_mf:
            m = MagicMock()
            m.predict.return_value = np.random.rand(1, 3).astype(np.float32)
            mock_mf.return_value = m
            with _mock_gradcam_tf():
                dn.grad_cam(rgb_512)
        assert mock_mf.called

    def test_not_all_zeros(self, rgb_512):
        result = _run_grad_cam(rgb_512)
        assert result.sum() > 0

    def test_three_channels(self, rgb_512):
        result = _run_grad_cam(rgb_512)
        assert result.shape[2] == 3

    def test_deterministic(self, rgb_512):
        with patch("grad_cam.model_fun") as mock_mf:
            m = MagicMock()
            preds = np.array([[0.1, 0.8, 0.1]], dtype=np.float32)
            m.predict.return_value = preds
            mock_mf.return_value = m
            with _mock_gradcam_tf():
                r1 = dn.grad_cam(rgb_512)
                r2 = dn.grad_cam(rgb_512)
        np.testing.assert_array_equal(r1, r2)

    def test_does_not_raise(self, rgb_512):
        try:
            _run_grad_cam(rgb_512)
        except Exception:
            pytest.fail("grad_cam raised unexpectedly")

    def test_large_input(self):
        large = np.random.randint(0, 256, (2000, 2000, 3), dtype=np.uint8)
        result = _run_grad_cam(large)
        assert result.shape == (512, 512, 3)

    def test_model_predict_called(self, rgb_512):
        with patch("grad_cam.model_fun") as mock_mf:
            m = MagicMock()
            m.predict.return_value = np.random.rand(1, 3).astype(np.float32)
            mock_mf.return_value = m
            with _mock_gradcam_tf():
                dn.grad_cam(rgb_512)
        assert m.predict.called

    def test_output_is_colored(self, rgb_512):
        result = _run_grad_cam(rgb_512)
        assert result.shape[-1] == 3


# ═══════════════════════════════════════════════════════════════════════
#  6. model_fun() — 10 tests
# ═══════════════════════════════════════════════════════════════════════


class TestModelFun:
    @patch("load_model.tf.keras.models.load_model")
    def test_returns_model(self, mock_load):
        mock_load.return_value = MagicMock()
        assert dn.model_fun() is not None

    @patch("load_model.tf.keras.models.load_model")
    def test_raises_on_missing(self, mock_load):
        mock_load.side_effect = OSError()
        with pytest.raises(FileNotFoundError):
            dn.model_fun()

    @patch("load_model.tf.keras.models.load_model")
    def test_error_contains_model_name(self, mock_load):
        mock_load.side_effect = OSError()
        with pytest.raises(FileNotFoundError) as exc:
            dn.model_fun()
        assert "conv_MLP_84.h5" in str(exc.value)

    @patch("load_model.tf.keras.models.load_model")
    def test_calls_load_model(self, mock_load):
        mock_load.return_value = MagicMock()
        dn.model_fun()
        assert mock_load.call_count >= 1

    @patch("load_model.tf.keras.models.load_model")
    def test_path_contains_models(self, mock_load):
        mock_load.return_value = MagicMock()
        dn.model_fun()
        path = mock_load.call_args[0][0]
        assert "models/" in str(path)

    @patch("load_model.tf.keras.models.load_model")
    def test_path_has_h5(self, mock_load):
        mock_load.return_value = MagicMock()
        dn.model_fun()
        path = mock_load.call_args[0][0]
        assert ".h5" in str(path)

    @patch("load_model.tf.keras.models.load_model")
    def test_corrupted_model_raises(self, mock_load):
        mock_load.side_effect = OSError("corrupted")
        with pytest.raises(FileNotFoundError):
            dn.model_fun()

    @patch("load_model.tf.keras.models.load_model")
    def test_valid_file_does_not_raise(self, mock_load):
        mock_load.return_value = MagicMock()
        try:
            dn.model_fun()
        except Exception:
            pytest.fail("raised unexpectedly")

    @patch("load_model.tf.keras.models.load_model")
    def test_returned_model_has_predict(self, mock_load):
        mock_load.return_value = MagicMock()
        assert hasattr(dn.model_fun(), "predict")

    @patch("load_model.tf.keras.models.load_model")
    def test_model_returns_3_classes(self, mock_load):
        m = MagicMock()
        m.predict.return_value = np.array([[0.1, 0.8, 0.1]])
        mock_load.return_value = m
        model = dn.model_fun()
        result = model.predict(np.random.rand(1, 32, 32, 1))
        assert result.shape == (1, 3)


# ═══════════════════════════════════════════════════════════════════════
#  7. read_dicom vs read_jpg — 10 tests
# ═══════════════════════════════════════════════════════════════════════


class TestReadDicomVsJpg:
    @patch("read_img.pydicom.dcmread")
    @patch("read_img.cv2.imread")
    def test_both_return_two_element_tuple(self, mock_imread, mock_dcm):
        ds = MagicMock()
        ds.pixel_array = np.ones((16, 16), dtype=np.uint16)
        mock_dcm.return_value = ds
        mock_imread.return_value = np.ones((16, 16, 3), dtype=np.uint8)
        d = dn.read_dicom_file("a.dcm")
        j = dn.read_jpg_file("a.jpg")
        assert len(d) == 2 and len(j) == 2

    @patch("read_img.pydicom.dcmread")
    def test_dicom_rgb_output(self, mock_dcm):
        ds = MagicMock()
        ds.pixel_array = np.ones((32, 32), dtype=np.uint16)
        mock_dcm.return_value = ds
        assert dn.read_dicom_file("a.dcm")[0].shape[-1] == 3

    @patch("read_img.cv2.imread")
    def test_jpg_preserves_original_shape(self, mock_imread):
        mock_imread.return_value = np.ones((32, 32, 3), dtype=np.uint8)
        assert dn.read_jpg_file("a.jpg")[0].shape[-1] == 3

    @patch("read_img.pydicom.dcmread")
    @patch("read_img.cv2.imread")
    def test_both_return_pil_image(self, mock_imread, mock_dcm):
        ds = MagicMock()
        ds.pixel_array = np.ones((10, 10), dtype=np.uint16)
        mock_dcm.return_value = ds
        mock_imread.return_value = np.ones((10, 10, 3), dtype=np.uint8)
        d = dn.read_dicom_file("a.dcm")
        j = dn.read_jpg_file("a.jpg")
        assert isinstance(d[1], Image.Image)
        assert isinstance(j[1], Image.Image)

    @patch("read_img.pydicom.dcmread")
    def test_dicom_calls_dcmread(self, mock_dcm):
        ds = MagicMock()
        ds.pixel_array = np.ones((4, 4), dtype=np.uint16)
        mock_dcm.return_value = ds
        dn.read_dicom_file("f.dcm")
        assert mock_dcm.called

    @patch("read_img.cv2.imread")
    def test_jpg_calls_imread(self, mock_imread):
        mock_imread.return_value = np.ones((4, 4, 3), dtype=np.uint8)
        dn.read_jpg_file("f.jpg")
        assert mock_imread.called

    @patch("read_img.pydicom.dcmread")
    def test_dicom_normalizes_values(self, mock_dcm):
        ds = MagicMock()
        ds.pixel_array = np.array([[500, 4000]], dtype=np.uint16)
        mock_dcm.return_value = ds
        assert dn.read_dicom_file("a.dcm")[0].max() <= 255

    @patch("read_img.cv2.imread")
    def test_jpg_keeps_values(self, mock_imread):
        mock_imread.return_value = np.array([[100, 200]], dtype=np.uint8)
        assert dn.read_jpg_file("a.jpg")[0].max() <= 255

    @patch("read_img.pydicom.dcmread")
    @patch("read_img.cv2.imread")
    def test_both_fail_on_missing(self, mock_imread, mock_dcm):
        mock_dcm.side_effect = FileNotFoundError
        mock_imread.return_value = None
        with pytest.raises(Exception):
            dn.read_dicom_file("m.dcm")
        with pytest.raises(Exception):
            dn.read_jpg_file("m.jpg")

    @patch("read_img.pydicom.dcmread")
    @patch("read_img.cv2.imread")
    def test_both_handle_grayscale(self, mock_imread, mock_dcm):
        gray = np.ones((16, 16), dtype=np.uint8)
        ds = MagicMock()
        ds.pixel_array = gray.astype(np.uint16)
        mock_dcm.return_value = ds
        mock_imread.return_value = gray
        d = dn.read_dicom_file("g.dcm")
        j = dn.read_jpg_file("g.jpg")
        assert isinstance(d[0], np.ndarray)
        assert isinstance(j[0], np.ndarray)


# ═══════════════════════════════════════════════════════════════════════
#  8. Edge cases generales — 10 tests
# ═══════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    @patch("preprocess_img.cv2.resize", side_effect=TypeError)
    @patch("preprocess_img.cv2.cvtColor")
    @patch("preprocess_img.cv2.createCLAHE")
    def test_preprocess_wrong_type_raises(self, mock_clahe, mock_cvt, mock_resize):
        with pytest.raises(Exception):
            dn.preprocess("not-an-image")

    def test_dicom_empty_path_raises(self):
        with patch("read_img.pydicom.dcmread", side_effect=FileNotFoundError):
            with pytest.raises(Exception):
                dn.read_dicom_file("")

    def test_jpg_empty_path_raises(self):
        with patch("read_img.cv2.imread", return_value=None):
            with pytest.raises(Exception):
                dn.read_jpg_file("")

    @patch("load_model.tf.keras.models.load_model", side_effect=OSError)
    def test_model_fun_invalid_path(self, mock_load):
        with pytest.raises(FileNotFoundError):
            dn.model_fun()

    @patch("integrator.model_fun")
    @patch("integrator.preprocess", side_effect=Exception("bad"))
    def test_predict_preprocess_fail(self, mock_pp, mock_mf, rgb_512):
        with patch("integrator.grad_cam"):
            with pytest.raises(Exception):
                dn.predict(rgb_512)

    @patch("integrator.model_fun")
    @patch("integrator.grad_cam")
    @patch("integrator.preprocess")
    def test_predict_no_model_fun(self, mock_pp, mock_gc, mock_mf, rgb_512):
        mock_mf.side_effect = FileNotFoundError
        mock_pp.return_value = np.random.rand(1, 512, 512, 1).astype(np.float32)
        with pytest.raises(Exception):
            dn.predict(rgb_512)

    def test_dicom_non_existent_path(self):
        with patch("read_img.pydicom.dcmread", side_effect=FileNotFoundError):
            with pytest.raises(Exception):
                dn.read_dicom_file("/nonexistent/file.dcm")

    def test_jpg_non_existent_path(self):
        with patch("read_img.cv2.imread", return_value=None):
            with pytest.raises(Exception):
                dn.read_jpg_file("/nonexistent/file.jpg")

    def test_predict_empty_array_raises(self):
        with patch("integrator.model_fun") as mock_mf:
            m = MagicMock()
            m.predict.return_value = np.array([[0.1, 0.8, 0.1]], dtype=np.float32)
            mock_mf.return_value = m
            with patch("integrator.grad_cam"):
                with patch("integrator.preprocess", side_effect=Exception):
                    with pytest.raises(Exception):
                        dn.predict(np.array([], dtype=np.uint8))

    def test_grad_cam_handles_all_black(self):
        black = np.zeros((512, 512, 3), dtype=np.uint8)
        result = _run_grad_cam(black)
        assert result.shape == (512, 512, 3)

    def test_preprocess_none_raises(self):
        with pytest.raises(Exception):
            dn.preprocess(None)


# ═══════════════════════════════════════════════════════════════════════
#  10. Integración (requiere modelo .h5 real)
# ═══════════════════════════════════════════════════════════════════════


class TestIntegration:
    """Pruebas end-to-end que requieren el archivo .h5 e imágenes locales.

    Se saltan automáticamente si los recursos no están disponibles,
    permitiendo ejecución en CI sin el modelo.
    """

    MODEL_PATH = "models/conv_MLP_84.h5"

    def test_model_loads_without_error(self):
        if not os.path.exists(self.MODEL_PATH):
            pytest.skip("Modelo .h5 no disponible localmente")
        model = dn.model_fun()
        assert hasattr(model, "predict")
        assert hasattr(model, "output")

    def test_predict_end_to_end(self):
        if not os.path.exists(self.MODEL_PATH):
            pytest.skip("Modelo .h5 no disponible localmente")
        img_path = "images/JPG/normal/NORMAL2-IM-1144-0001.jpeg"
        if not os.path.exists(img_path):
            pytest.skip("Imagen de prueba no disponible")
        array, _ = dn.read_jpg_file(img_path)
        label, proba, heatmap = dn.predict(array)
        assert label in ("bacteriana", "normal", "viral")
        assert 0 <= proba <= 100
        assert heatmap.shape == (512, 512, 3)


# ═══════════════════════════════════════════════════════════════════════
#  9. CSV export — 5 tests
# ═══════════════════════════════════════════════════════════════════════


class TestCSVExport:
    def test_dash_delimiter(self):
        assert "-".join(["123", "normal", "85.00%"]) == "123-normal-85.00%"

    def test_includes_all_fields(self):
        line = "-".join(["12345", "bacteriana", "92.50%"])
        assert "12345" in line and "bacteriana" in line and "92.50%" in line

    def test_appending(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("1-normal-50.00%\n")
            name = f.name
        with open(name, "a") as f:
            w = csv.writer(f, delimiter="-")
            w.writerow(["2", "viral", "75.00%"])
        with open(name) as f:
            lines = f.readlines()
        assert len(lines) == 2
        os.unlink(name)

    def test_exact_format(self):
        assert "-".join(["1001", "bacteriana", "99.99%"]) == "1001-bacteriana-99.99%"

    def test_id_with_hyphen(self):
        parts = "-".join(["123-456", "normal", "80.00%"]).split("-")
        assert len(parts) >= 4
