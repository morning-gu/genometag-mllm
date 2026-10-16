"""convert_movienet 一次性工具：迷你官方格式 fixture → 适配器假定格式。"""
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.convert_movienet import convert  # noqa: E402


def _make_fixture(root: Path):
    """两个假电影的官方分发结构。"""
    ids = ["tt0000001", "tt0000002"]
    (root / "list.v1.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
    meta = root / "meta_data" / "meta"
    meta.mkdir(parents=True)
    (meta / "tt0000001.json").write_text(json.dumps(
        {"imdb_id": "tt0000001", "title": "Alpha Film (2001)"}), encoding="utf-8")
    (meta / "tt0000002.json").write_text(json.dumps(
        {"imdb_id": "tt0000002", "title": "Beta Film (TV Movie 2010)"}), encoding="utf-8")
    with zipfile.ZipFile(root / "script1K.v1.zip", "w") as zf:
        zf.writestr("script/tt0000001.script", "INT. ALPHA - NIGHT\nscene one")
        zf.writestr("script/tt9999999.script", "not in list, must be skipped")
    with zipfile.ZipFile(root / "subtitle1K.v1.zip", "w") as zf:
        zf.writestr("subtitle/tt0000001.srt", "1\n00:00:01,000 --> 00:00:02,000\nhello\n")
        zf.writestr("subtitle/tt0000002.srt", "1\n00:00:01,000 --> 00:00:02,000\nworld\n")
    for tt in ids:
        d = root / "frames" / "frames" / tt
        d.mkdir(parents=True)
        (d / "shot_0000_img_0.jpg").write_bytes(b"\xff\xd8fake")
    return ids


def test_convert_movienet(tmp_path):
    import pandas as pd

    root = tmp_path / "official"
    root.mkdir()
    ids = _make_fixture(root)
    out = tmp_path / "out"
    stats = convert(root, out, copy_frames=True)   # Windows 本地用复制；服务器用符号链接

    assert stats == {"list": 2, "meta": 2, "meta_missing": 0,
                     "scripts": 1, "subtitles": 2, "frames": 2}
    movies = pd.read_csv(out / "movies.csv")
    assert list(movies.columns) == ["movie_id", "title", "year", "imdb_id"]
    assert movies.loc[1, "year"] == 2010            # "(TV Movie 2010)" 尾部年份解析
    assert (out / "scripts" / "tt0000001.txt").exists()      # .script 改名 .txt
    assert not (out / "scripts" / "tt9999999.txt").exists()  # 非 list 内成员被跳过
    assert (out / "subtitles" / "tt0000002.srt").exists()
    assert (out / "keyframes" / "tt0000001" / "shot_0000_img_0.jpg").exists()

    # 输出直接满足适配器假定格式（load_movienet 可读）
    from genometag.data.movienet import load_movienet
    mms = load_movienet(out)
    assert len(mms) == 2
    m1 = next(m for m in mms if m.movienet_id == "tt0000001")
    assert m1.year == 2001 and m1.imdb_id == "tt0000001"
    assert m1.script is not None and "ALPHA" in m1.script
    assert m1.subtitles is not None and "hello" in m1.subtitles
    assert len(m1.keyframes) == 1
    m2 = next(m for m in mms if m.movienet_id == "tt0000002")
    assert m2.script is None                       # 该电影无剧本（zip 中没有）
