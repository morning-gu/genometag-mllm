"""MovieLens 适配器：格式解析与规范化。"""
import pandas as pd

from genometag.data.movielens import (
    load_genome_scores, load_genome_tags, load_links, load_movies,
    load_user_tags, parse_title_year,
)


def test_parse_title_year():
    assert parse_title_year("Toy Story (1995)") == ("Toy Story", 1995)
    assert parse_title_year("No Year Here") == ("No Year Here", None)
    assert parse_title_year("City of God (2002)") == ("City of God", 2002)


def _write(tmp_path, name, rows, columns):
    p = tmp_path / name
    pd.DataFrame(rows, columns=columns).to_csv(p, index=False)
    return p


def test_load_genome_scores(tmp_path):
    p = _write(tmp_path, "genome-scores.csv",
               [(1, 5, 0.5), (1, 6, 0.9)], ["movieId", "tagId", "relevance"])
    df = load_genome_scores(p)
    assert list(df.columns) == ["movie_id", "tag_id", "relevance"]
    assert df["movie_id"].dtype.kind == "i"


def test_load_genome_tags(tmp_path):
    p = _write(tmp_path, "genome-tags.csv", [(5, "dark")], ["tagId", "tag"])
    assert list(load_genome_tags(p).columns) == ["tag_id", "tag"]


def test_load_movies_genres_list(tmp_path):
    p = _write(tmp_path, "movies.csv",
               [(1, "Toy Story (1995)", "Animation|Children"),
                (2, "Unknown (2001)", "(no genres listed)")],
               ["movieId", "title", "genres"])
    df = load_movies(p)
    assert df.loc[0, "genres"] == ["Animation", "Children"]
    assert df.loc[1, "genres"] == []          # (no genres listed) → 空列表
    assert df.loc[0, "year"] == 1995
    assert df.loc[0, "title"] == "Toy Story"


def test_load_links_and_user_tags(tmp_path):
    links = _write(tmp_path, "links.csv", [(1, 114709, 862)], ["movieId", "imdbId", "tmdbId"])
    assert list(load_links(links).columns) == ["movie_id", "imdb_id"]
    tags = _write(tmp_path, "tags.csv",
                  [(9, 1, "classic", 1700000000)], ["userId", "movieId", "tag", "timestamp"])
    df = load_user_tags(tags)
    assert list(df.columns) == ["user_id", "movie_id", "tag", "timestamp"]
