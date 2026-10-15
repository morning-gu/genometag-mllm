# 数据下载指南

## MovieLens（含 genome 数据）

1. 访问 https://grouplens.org/datasets/movielens/ ，下载 **ml-latest.zip**（或 ml-25m.zip，genome 文件结构相同）
2. 解压到 `data/raw/movielens/ml-latest/`，确保以下文件在位：
   - `genome-scores.csv`（movieId, tagId, relevance）
   - `genome-tags.csv`（tagId, tag；共 1,128 个）
   - `movies.csv`、`links.csv`、`tags.csv`
3. 校验：`.venv/Scripts/python scripts/01_verify_data.py`

## MovieNet

1. 访问 https://movienet.site/ ，按其流程申请/下载（通常含元数据 CSV 与各模态数据的分块下载）
2. 解压并整理到 `data/raw/movienet/`，目标结构（**当前适配器假定的格式**，
   与实际发布有出入时改 `genometag/data/movienet.py` 顶部的常量与 `load_movienet`）：

   ```
   data/raw/movienet/
   ├── movies.csv            # movie_id,title,year,imdb_id
   ├── keyframes/<id>/*.jpg
   ├── scripts/<id>.txt      # 覆盖不足时字幕可作台词模态代理
   └── subtitles/<id>.srt
   ```

3. 校验：`01_verify_data.py` 会报告**年份分布**——spec §9 风险：MovieNet 可能不含
   2022 年后电影，届时需重新确定时序分割年份（改配置 `temporal_split_year` 一行）。
