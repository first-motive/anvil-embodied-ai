"""The encoder override LeRobotWriter forwards to LeRobot 0.6.1.

`LeRobotDataset.create()` and `.resume()` dropped their `vcodec` argument in
0.6.1 and take an `rgb_encoder` config instead. A codec is forwarded only when
the caller names one, so LeRobot's own tuned pairing survives by default.
"""

from lerobot.datasets.video_utils import RGBEncoderConfig

from mcap_converter.core.writer import LeRobotWriter


def _writer(tmp_path, **kwargs) -> LeRobotWriter:
    return LeRobotWriter(output_dir=str(tmp_path / "out"), repo_id="local/test", **kwargs)


def test_no_codec_forwards_nothing(tmp_path):
    assert _writer(tmp_path)._encoder_kwargs() == {}


def test_named_codec_becomes_an_rgb_encoder_config(tmp_path):
    kwargs = _writer(tmp_path, vcodec="h264")._encoder_kwargs()

    assert set(kwargs) == {"rgb_encoder"}
    assert isinstance(kwargs["rgb_encoder"], RGBEncoderConfig)
    assert kwargs["rgb_encoder"].vcodec == "h264"
