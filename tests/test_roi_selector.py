"""roi_selector 单元测试 — 合成热力图/检测框 (无需真实模型)"""
import numpy as np
import pytest

from core.roi_selector import select_rois, crop_rois


def _blob_map(h, w, centers, radius=12, base=0.1):
    """生成带若干高斯亮斑的背景热力图。"""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    m = np.full((h, w), base, dtype=np.float32)
    for cy, cx in centers:
        m += np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2)
                    / (2 * radius ** 2))
    return m


class TestSelectRois:
    def test_single_blob_localized(self):
        m = _blob_map(200, 300, [(60, 150)])
        rois = select_rois((200, 300), anomaly_map=m)
        assert len(rois) == 1
        r = rois[0]
        # ROI 中心应落在亮斑附近 (含 padding 余量)
        cx, cy = r["x"] + r["w"] / 2, r["y"] + r["h"] / 2
        assert abs(cx - 150) < 40 and abs(cy - 60) < 40
        assert r["source"] == "heatmap"
        assert 0.5 < r["score"] <= 1.0

    def test_two_blobs_two_rois(self):
        m = _blob_map(300, 400, [(50, 50), (250, 350)])
        rois = select_rois((300, 400), anomaly_map=m, max_rois=3)
        assert len(rois) == 2

    def test_max_rois_cap(self):
        m = _blob_map(400, 400, [(50, 50), (50, 350), (350, 50), (350, 350)])
        rois = select_rois((400, 400), anomaly_map=m, max_rois=2)
        assert len(rois) == 2

    def test_flat_map_no_roi(self):
        m = np.full((100, 100), 0.5, dtype=np.float32)
        assert select_rois((100, 100), anomaly_map=m) == []

    def test_zero_map_no_roi(self):
        m = np.zeros((100, 100), dtype=np.float32)
        assert select_rois((100, 100), anomaly_map=m) == []

    def test_noise_blob_filtered(self):
        m = _blob_map(200, 200, [(100, 100)], radius=15)
        # 加一个 3px 噪点 (面积远低于 min_area_frac)
        m[10:13, 10:13] = 5.0
        rois = select_rois((200, 200), anomaly_map=m,
                           min_area_frac=0.001)
        # 噪点区域被过滤, 大亮斑保留
        assert all(r["w"] * r["h"] >= 0.001 * 200 * 200 for r in rois)

    def test_boxes_only(self):
        rois = select_rois((200, 200),
                           boxes=[(10, 20, 60, 80), (100, 100, 150, 160)],
                           box_scores=[0.9, 0.5])
        assert len(rois) == 2
        assert rois[0]["source"] == "det"
        assert rois[0]["score"] == pytest.approx(0.9)
        # 按分数降序
        assert rois[0]["score"] >= rois[1]["score"]

    def test_overlapping_boxes_merged(self):
        # 第二个框被第一个完全覆盖 → 归并
        rois = select_rois((200, 200),
                           boxes=[(10, 10, 100, 100), (20, 20, 60, 60)],
                           box_scores=[0.9, 0.4])
        assert len(rois) == 1
        assert rois[0]["score"] == pytest.approx(0.9)

    def test_padding_clipped_to_bounds(self):
        m = _blob_map(100, 100, [(5, 5)], radius=8)
        rois = select_rois((100, 100), anomaly_map=m, pad_frac=0.5)
        for r in rois:
            assert r["x"] >= 0 and r["y"] >= 0
            assert r["x"] + r["w"] <= 100
            assert r["y"] + r["h"] <= 100

    def test_invalid_inputs(self):
        assert select_rois(None) == []
        assert select_rois((5, 5)) == []          # 过小图
        assert select_rois((100, 100)) == []      # 无证据
        assert select_rois((100, 100), boxes=[(5, 5, 5, 5)]) == []  # 退化框


class TestCropRois:
    def test_crop_region_exact(self):
        img = np.arange(100 * 100 * 3, dtype=np.uint8).reshape(100, 100, 3)
        rois = [{"x": 10, "y": 20, "w": 40, "h": 30,
                 "score": 1.0, "source": "det"}]
        out = crop_rois(img, rois, min_side=0, max_side=10000)
        assert len(out) == 1
        roi, crop = out[0]
        assert crop.shape[:2] == (30, 40)
        assert np.array_equal(crop, img[20:50, 10:50])

    def test_small_crop_upscaled(self):
        img = np.zeros((500, 500, 3), dtype=np.uint8)
        # 70px 短边 → 放大至 min_side (256), 未触发上限
        rois = [{"x": 100, "y": 100, "w": 70, "h": 70,
                 "score": 1.0, "source": "det"}]
        out = crop_rois(img, rois, min_side=256)
        _, crop = out[0]
        assert min(crop.shape[:2]) >= 256
        # 20px 极小裁切 → 4倍上限 (纯插值不无限放大)
        rois2 = [{"x": 100, "y": 100, "w": 20, "h": 20,
                  "score": 1.0, "source": "det"}]
        _, crop2 = crop_rois(img, rois2, min_side=256)[0]
        assert max(crop2.shape[:2]) <= 20 * 4 + 2

    def test_large_crop_downscaled(self):
        img = np.zeros((3000, 3000, 3), dtype=np.uint8)
        rois = [{"x": 0, "y": 0, "w": 3000, "h": 2000,
                 "score": 1.0, "source": "full"}]
        out = crop_rois(img, rois, max_side=1024)
        _, crop = out[0]
        assert max(crop.shape[:2]) <= 1024

    def test_out_of_bounds_clipped(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        rois = [{"x": 90, "y": 90, "w": 50, "h": 50,
                 "score": 1.0, "source": "det"}]
        out = crop_rois(img, rois, min_side=0, max_side=10000)
        _, crop = out[0]
        assert crop.shape[:2] == (10, 10)

    def test_empty_inputs(self):
        assert crop_rois(None, [{"x": 0, "y": 0, "w": 5, "h": 5}]) == []
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        assert crop_rois(img, []) == []
