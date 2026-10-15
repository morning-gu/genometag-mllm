# 数据产物 Schema

## data/interim/aligned.parquet
| 列 | 类型 | 说明 |
|----|------|------|
| movie_id | int | MovieLens movieId（全管道主键） |
| title | str | 去年份的标题 |
| year | int | 上映年 |
| genres | list[str] | |
| imdb_id | int | 规范化 IMDb 数字 ID |
| movienet_id | str | |
| match_method | str | imdb / title |

## data/interim/align_report.md
匹配率、歧义清单（同名同年）、未匹配清单。

## data/processed/dataset.parquet
上述列 + `keyframes: list[str]`（相对 movienet 根）、`script_path`、`subtitles_path`（相对路径或 null）。

## data/processed/genome_scores.parquet
`movie_id, tag_id, relevance`（仅对齐电影；ground truth）。

## data/processed/genome_tags.parquet
`tag_id, tag`。

## data/processed/tag_groups.json（入 git）
`[{tag_id, tag, group, tag_type: content|preference, definition}]`。

## results/<exp>/annotations.jsonl（append-only，续跑事实来源）
每行 `{"task_key": "movie|group|run", "status": "done|parse_failed|no_content",
"rows": [...], "usage": {...}}`。

## results/<exp>/annotations.parquet
`movie_id, tag_id, group, modality_set, run_id, score`。

## results/baselines/baselines.parquet
`movie_id, tag_id, score, method`。
