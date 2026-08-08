"""The house style, the aggregation behind it, and the figures it produces.

What is worth testing about a figure is not what it looks like — an image
comparison would fail on every matplotlib release and tell nobody anything.
What is worth testing is the set of claims the figure makes: that the band is a
median over seeds and not one seed, that a mixed-hardware group is refused
rather than plotted, that a diverging field is centred on zero, and that the
palette is the validated one rather than whatever was cycled next.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")  # no display in CI; must precede pyplot
import matplotlib.pyplot as plt  # noqa: E402

from pinnslab.registry.run import (  # noqa: E402
    CHECKPOINT_DIR,
    CONFIG_JSON,
    PROVENANCE_JSON,
    RESULT_JSON,
    TRACE_JSONL,
)
from pinnslab.registry.schema import (  # noqa: E402
    Provenance,
    ResultRow,
    RunStatus,
    TracePoint,
    json_safe,
)
from pinnslab.viz import style  # noqa: E402
from pinnslab.viz.aggregate import (  # noqa: E402
    RunRecord,
    Summary,  # noqa: E402
    assert_comparable,
    band,
    group,
    load_records,
    summarise,
)
from pinnslab.viz.convergence import convergence_figure, label_for  # noqa: E402
from pinnslab.viz.tables import (  # noqa: E402
    column_format,
    format_number,
    summary_table,
)

pytestmark = pytest.mark.unit


# -- fixtures: synthetic runs on disk ------------------------------------------


def make_row(**over) -> ResultRow:
    base = dict(
        run_id="r",
        config_hash="abcd1234",
        status=RunStatus.COMPLETED,
        pinnslab_version="0.1.0",
        git_sha="deadbeef",
        git_dirty=False,
        seed=0,
        gpu_name="cpu",
        dtype="float64",
        device_profile="cpu",
        timestamp_utc="2026-08-08T00:00:00Z",
        steps_completed=100,
        final_metrics={"rel_l2": 1e-3},
    )
    base.update(over)
    return ResultRow(**base)


def make_record(*, seed=0, final=1e-3, curve=None, status=RunStatus.COMPLETED, **over):
    curve = curve if curve is not None else [1.0, 0.1, 0.01]
    trace = tuple(
        TracePoint(step=step, wall_time=float(step) / 10.0, metrics={"rel_l2": value})
        for step, value in zip([0, 10, 100], curve, strict=True)
    )
    row = make_row(
        run_id=f"run{seed}", seed=seed, status=status,
        final_metrics={"rel_l2": final}, **over,
    )
    return RunRecord(row=row, trace=trace)


#: The smallest config that validates. `_unfinished_row` re-reads it to recover
#: the condition of a run that never wrote a result, so a run directory without
#: it is not a run.
MINIMAL_CONFIG = {
    "stages": [{"name": "adam", "steps": 1, "optimizers": [{"name": "adam"}]}]
}


def write_run(root: Path, record: RunRecord) -> Path:
    """A run directory as `Run` would have left it, without training anything."""
    directory = root / record.row.run_id
    (directory / CHECKPOINT_DIR).mkdir(parents=True)
    (directory / RESULT_JSON).write_text(
        json.dumps(json_safe(record.row.model_dump(mode="json"))), encoding="utf-8"
    )
    (directory / CONFIG_JSON).write_text(
        json.dumps({**MINIMAL_CONFIG, "seed": record.row.seed}), encoding="utf-8"
    )
    prov = Provenance(
        pinnslab_version="0.1.0", git_sha="deadbeef", git_dirty=False,
        git_source="git", gpu_name=record.row.gpu_name, device_profile="cpu",
        dtype=record.row.dtype, seed=record.row.seed,
        timestamp_utc="2026-08-08T00:00:00Z", hostname="h",
        python_version="3.13", torch_version="2.12", platform="test",
    )
    (directory / PROVENANCE_JSON).write_text(
        json.dumps(prov.model_dump(mode="json")), encoding="utf-8"
    )
    with (directory / TRACE_JSONL).open("w", encoding="utf-8") as fh:
        for point in record.trace:
            fh.write(json.dumps(json_safe(point.model_dump(mode="json"))) + "\n")
    return directory


@pytest.fixture
def five_seeds():
    """One condition, five seeds, one of them a heavy-tailed outlier.

    The outlier is the point: it is drawn from the real 2026-08-08 measurement
    where one seed of a correct Burgers config landed at 0.435 against siblings
    at 0.030-0.044.
    """
    finals = [0.030, 0.036, 0.041, 0.044, 0.435]
    return [
        make_record(seed=i, final=f, curve=[1.0, f * 3, f])
        for i, f in enumerate(finals)
    ]


# -- the palette ---------------------------------------------------------------


def test_the_palette_is_the_validated_one():
    """Pinned because a "nicer" palette is a tempting one-line change, and the
    reason this one is here is measured, not aesthetic: SciencePlots' default
    cycle collapses to ΔE 2.8 under protanopia on its orange/green pair."""
    assert style.PALETTE == (
        "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9",
    )


def test_every_hue_has_a_linestyle_and_a_marker():
    """Color is never the only channel: IEEE wants B&W-readable figures and
    three of these six sit below 3:1 contrast on white."""
    assert len(style.LINESTYLES) == len(style.PALETTE)
    assert len(style.MARKERS) == len(style.PALETTE)


def test_the_palette_is_not_cycled():
    """A 7th series would repeat the 1st, and two conditions would share an
    identity. Better to refuse and make the caller fold or facet."""
    assert style.color(5) == "#56B4E9"
    with pytest.raises(IndexError, match="never cycled"):
        style.color(len(style.PALETTE))


def test_series_kwargs_carry_both_channels():
    kwargs = style.series_kwargs(1)
    assert kwargs["color"] == style.PALETTE[1]
    assert kwargs["linestyle"] == style.LINESTYLES[1]


def test_no_rainbow_colormap_anywhere():
    """jet distorts by up to ~8% through uneven lightness and is unreadable in
    grayscale (Crameri et al. 2020). It must not be reachable from the style."""
    assert style.SEQUENTIAL == "viridis"
    assert "jet" not in (style.SEQUENTIAL, style.DIVERGING)


# -- the style -----------------------------------------------------------------


def test_usetex_is_off_by_default():
    """A figure that needs a working TeX install is a figure that dies in CI."""
    assert style.rc_params()["text.usetex"] is False
    assert style.rc_params(usetex=True)["text.usetex"] is True


def test_pdf_fonts_are_embedded_not_type_3():
    """Type 3 bitmap fonts are rejected outright by several publishers."""
    params = style.rc_params()
    assert params["pdf.fonttype"] == 42
    assert params["ps.fonttype"] == 42


def test_the_style_does_not_leak_out_of_its_context():
    """Importing or using pinnslab must not mutate a notebook's plotting state,
    the same no-global-side-effects rule the package follows for torch."""
    before = matplotlib.rcParams["font.family"]
    with style.use_style():
        assert matplotlib.rcParams["lines.linewidth"] == 1.1
    assert matplotlib.rcParams["font.family"] == before


def test_figure_widths_are_journal_columns():
    assert style.figsize("single")[0] == pytest.approx(3.35)
    assert style.figsize("double")[0] == pytest.approx(7.48)
    assert style.figsize("single", aspect=1.0)[0] == style.figsize("single", 1.0)[1]
    with pytest.raises(ValueError, match="unknown width"):
        style.figsize("A4")


# -- symmetric_norm ------------------------------------------------------------


def test_a_diverging_norm_is_centred_on_zero():
    """The load-bearing one. Matplotlib's default norm puts the colormap's
    neutral midpoint wherever the data's midpoint falls, so an asymmetric field
    renders with its white band inside the positive data — a sign change the
    solution does not have."""
    norm = style.symmetric_norm(np.array([-0.2, 1.0]))
    assert norm.vmin == -1.0 and norm.vmax == 1.0
    assert norm(0.0) == pytest.approx(0.5)


def test_a_robust_norm_clips_a_single_spike():
    field = np.concatenate([np.linspace(-1, 1, 999), [500.0]])
    assert style.symmetric_norm(field).vmax == 500.0
    assert style.symmetric_norm(field, robust=True).vmax < 2.0


def test_an_all_zero_field_still_gets_a_usable_norm():
    norm = style.symmetric_norm(np.zeros(10))
    assert norm.vmin < norm.vmax


def test_a_norm_needs_at_least_one_finite_value():
    with pytest.raises(ValueError, match="non-finite"):
        style.symmetric_norm(np.array([np.nan, np.inf]))


def test_non_finite_cells_do_not_set_the_limits():
    """A diverged field has NaNs in it; they must not swallow the scale."""
    norm = style.symmetric_norm(np.array([-2.0, np.nan, 1.0, np.inf]))
    assert norm.vmax == 2.0


# -- aggregation ---------------------------------------------------------------


def test_the_summary_is_a_median_not_a_mean(five_seeds):
    """The whole reason DESIGN.md §8 specifies median+IQR: the mean of these
    five is dragged past every non-outlier value by one diverged seed."""
    s = summarise(five_seeds, "rel_l2")
    assert s.median == pytest.approx(0.041)
    assert s.median < np.mean([0.030, 0.036, 0.041, 0.044, 0.435])
    assert s.q25 == pytest.approx(0.036)
    assert s.q75 == pytest.approx(0.044)


def test_the_summary_reports_how_many_seeds_it_used(five_seeds):
    failed = five_seeds + [make_record(seed=9, status=RunStatus.DIVERGED)]
    s = summarise(failed, "rel_l2")
    assert (s.n_total, s.n_used) == (6, 5)
    assert s.failure_rate == pytest.approx(1 / 6)


def test_a_non_finite_metric_does_not_count_as_a_result():
    """A diverged run writes nan, and nan must not become a median."""
    records = [
        make_record(seed=0, final=1e-3),
        make_record(seed=1, final=float("nan")),
    ]
    assert summarise(records, "rel_l2").n_used == 1


def test_a_condition_where_everything_failed_has_no_median():
    s = summarise([make_record(status=RunStatus.DIVERGED)], "rel_l2")
    assert math.isnan(s.median)
    assert s.failure_rate == 1.0


def test_grouping_by_a_tag(five_seeds):
    tagged = [
        make_record(seed=i, tags={"method": "ours" if i < 3 else "baseline"})
        for i in range(5)
    ]
    groups = group(tagged, "method")
    assert sorted(groups) == ["baseline", "ours"]
    assert len(groups["ours"]) == 3


def test_grouping_by_a_missing_tag_says_what_is_available():
    with pytest.raises(KeyError, match="method"):
        group([make_record(tags={"sampler": "rar"})], "method")


def test_mixed_hardware_is_refused_not_plotted():
    """DESIGN.md §5 makes this the code's job. A figure whose seeds ran on a T4
    and a P100 is not a figure with a caveat."""
    records = [
        make_record(seed=0, gpu_name="Tesla T4"),
        make_record(seed=1, gpu_name="Tesla P100"),
    ]
    with pytest.raises(ValueError, match="gpu_name"):
        assert_comparable(records)


def test_mixed_precision_is_refused_too():
    """float32 bottoms out at ~1e-4-1e-5, exactly where accuracy claims live."""
    records = [make_record(seed=0), make_record(seed=1, dtype="float32")]
    with pytest.raises(ValueError, match="dtype"):
        assert_comparable(records)


def test_one_hardware_is_fine(five_seeds):
    assert_comparable(five_seeds)  # does not raise


# -- bands ---------------------------------------------------------------------


def test_the_band_is_a_median_over_seeds(five_seeds):
    b = band(five_seeds, "rel_l2")
    assert list(b.x) == [0.0, 10.0, 100.0]
    assert b.n_used == 5
    assert b.median[-1] == pytest.approx(0.041)
    assert np.all(b.q25 <= b.median) and np.all(b.median <= b.q75)


def test_the_band_only_spans_steps_every_seed_reached():
    """A band that silently loses seeds as it goes right gets tighter exactly
    where the runs are failing."""
    short = make_record(seed=1)
    short = RunRecord(row=short.row, trace=short.trace[:2])
    b = band([make_record(seed=0), short], "rel_l2")
    assert list(b.x) == [0.0, 10.0]


def test_a_band_can_be_drawn_against_wall_clock(five_seeds):
    """Equal-compute comparison is a reviewer defence, not an optional extra."""
    b = band(five_seeds, "rel_l2", x="wall_time")
    assert list(b.x) == [0.0, 1.0, 10.0]


def test_an_unknown_metric_lists_the_ones_that_exist(five_seeds):
    with pytest.raises(ValueError, match="rel_l2"):
        band(five_seeds, "no_such_metric")


def test_a_group_with_no_traces_is_an_error():
    with pytest.raises(ValueError, match="no run in this group has a trace"):
        band([RunRecord(row=make_row(), trace=())], "rel_l2")


def test_a_diverged_seed_is_not_inside_the_median_band(five_seeds):
    """Caught by rendering the figure, not by the tests that preceded it: a
    diverged run still writes a trace, so it was being median-ed in while the
    figure's own failure annotation counted it as failed. The legend said n=5
    and the note said 1 of 5 failed, from the same five runs."""
    diverged = make_record(seed=9, curve=[40.0, 40.0, 40.0], status=RunStatus.DIVERGED)
    b = band([*five_seeds, diverged], "rel_l2")

    assert (b.n_used, b.n_total) == (5, 6)
    assert b.median[-1] == pytest.approx(0.041), "the diverged trace moved the median"
    assert b.q75[-1] < 1.0


def test_the_band_and_the_summary_count_the_same_population(five_seeds):
    """The legend and the failure note are two views of one number."""
    records = [*five_seeds, make_record(seed=9, status=RunStatus.DIVERGED)]
    b = band(records, "rel_l2")
    s = summarise(records, "rel_l2")
    assert (b.n_used, b.n_total) == (s.n_used, s.n_total)


def test_divergence_can_be_plotted_when_it_is_the_subject(five_seeds):
    diverged = make_record(seed=9, curve=[40.0, 40.0, 40.0], status=RunStatus.DIVERGED)
    records = [*five_seeds, diverged]

    included = band(records, "rel_l2", include_diverged=True)
    excluded = band(records, "rel_l2")

    assert (included.n_used, excluded.n_used) == (6, 5)
    assert included.q75[-1] > excluded.q75[-1], "the diverged run changed nothing"


def test_a_condition_where_everything_diverged_says_so():
    with pytest.raises(ValueError, match="include_diverged=True"):
        band([make_record(status=RunStatus.DIVERGED)], "rel_l2")


# -- reading from disk ---------------------------------------------------------


def test_records_round_trip_through_a_results_directory(tmp_path, five_seeds):
    for record in five_seeds:
        write_run(tmp_path, record)

    loaded = load_records(tmp_path)

    assert len(loaded) == 5
    assert [r.row.seed for r in loaded] == [0, 1, 2, 3, 4]
    assert all(len(r.trace) == 3 for r in loaded)
    assert summarise(loaded, "rel_l2").median == pytest.approx(0.041)


def test_an_unfinished_run_is_loaded_by_default(tmp_path, five_seeds):
    """Otherwise the failure rate is computed over the survivors only, which is
    not the failure rate."""
    write_run(tmp_path, five_seeds[0])
    killed = tmp_path / five_seeds[1].row.run_id
    write_run(tmp_path, five_seeds[1])
    (killed / RESULT_JSON).unlink()

    assert len(load_records(tmp_path)) == 2
    assert len(load_records(tmp_path, include_unfinished=False)) == 1


# -- the figure ----------------------------------------------------------------


def test_the_convergence_figure_draws_a_line_and_a_band_per_condition(five_seeds):
    fig = convergence_figure(five_seeds, by="config_hash")
    ax = fig.axes[0]

    assert len(ax.lines) == 1
    assert len(ax.collections) == 1, "the IQR band is missing"
    assert ax.get_yscale() == "log"
    plt.close(fig)


def test_the_figure_states_the_seed_count(five_seeds):
    """A median over 2 seeds and a median over 5 are different claims, and the
    figure has to say which without the reader going to the caption."""
    fig = convergence_figure(five_seeds)
    assert "n=5" in fig.axes[0].lines[0].get_label()
    plt.close(fig)


def test_a_condition_missing_seeds_says_so_in_the_legend(five_seeds):
    """A run that never traced cannot contribute to the band; the legend must
    report the band's real population rather than the condition's."""
    no_trace = RunRecord(row=make_row(run_id="run9", seed=9), trace=())
    fig = convergence_figure([*five_seeds, no_trace])
    assert "5/6 seeds" in fig.axes[0].lines[0].get_label()
    plt.close(fig)


def test_the_figure_refuses_mixed_hardware():
    records = [make_record(seed=0), make_record(seed=1, gpu_name="Tesla P100")]
    with pytest.raises(ValueError, match="gpu_name"):
        convergence_figure(records)


def test_colors_follow_the_condition_not_its_rank(five_seeds):
    """A reader who learned "ours is blue" must not be misled by the next
    figure, so identity comes from the caller's order, not from row count."""
    tagged = [
        make_record(seed=i, tags={"method": m})
        for i, m in enumerate(["ours", "ours", "rar", "rar"])
    ]
    first = convergence_figure(tagged, by="method", order=["ours", "rar"])
    second = convergence_figure(tagged, by="method", order=["ours", "rar"])
    assert [line.get_color() for line in first.axes[0].lines] == [
        line.get_color() for line in second.axes[0].lines
    ]
    assert first.axes[0].lines[0].get_color() == style.PALETTE[0]
    plt.close(first)
    plt.close(second)


def test_an_order_naming_a_missing_condition_is_an_error(five_seeds):
    with pytest.raises(KeyError, match="no runs"):
        convergence_figure(five_seeds, by="config_hash", order=["nope"])


def test_a_log_x_axis_reports_the_step_0_point_it_cannot_show(five_seeds, caplog):
    """Also caught by rendering. Matplotlib drops x<=0 from a log axis without
    a word, which silently deletes the untrained baseline that
    MetricSchedule.record_first exists to capture."""
    fig = convergence_figure(five_seeds, xscale="log")
    assert "dropping 1 point" in caplog.text
    assert fig.axes[0].get_xscale() == "log"
    assert fig.axes[0].lines[0].get_xdata()[0] == 10.0
    plt.close(fig)


def test_a_linear_x_axis_keeps_step_0(five_seeds):
    fig = convergence_figure(five_seeds, xscale="linear")
    assert fig.axes[0].get_xscale() == "linear"
    assert fig.axes[0].lines[0].get_xdata()[0] == 0.0
    plt.close(fig)


def test_an_unknown_xscale_is_rejected(five_seeds):
    with pytest.raises(ValueError, match="xscale"):
        convergence_figure(five_seeds, xscale="symlog")


def test_the_failure_note_appears_only_when_something_failed(five_seeds):
    clean = convergence_figure(five_seeds)
    assert not [t for t in clean.axes[0].texts if "failed" in t.get_text()]

    diverged = make_record(seed=9, status=RunStatus.DIVERGED)
    dirty = convergence_figure([*five_seeds, diverged])
    assert [t for t in dirty.axes[0].texts if "failed" in t.get_text()]
    plt.close(clean)
    plt.close(dirty)


def test_saving_writes_a_vector_pdf(tmp_path, five_seeds):
    fig = convergence_figure(five_seeds)
    path = style.save(fig, tmp_path / "figures" / "convergence")

    assert path.suffix == ".pdf"
    assert path.read_bytes().startswith(b"%PDF")
    assert path.with_suffix(".png").exists()
    plt.close(fig)


def test_axis_labels_are_typeset_not_raw_keys():
    assert "L_2" in label_for("rel_l2")
    assert label_for("unknown_metric") == "unknown metric"


# -- tables --------------------------------------------------------------------


def test_numbers_are_formatted_consistently():
    assert format_number(0.041) == "$0.0410$"
    assert format_number(1.23e-5) == r"$1.23 \times 10^{-5}$"
    assert format_number(0) == "$0$"
    assert format_number(float("nan")) == "---"


def test_one_notation_for_a_whole_column():
    """The third thing rendering caught: values straddling 1e-3 came out as
    `9.34 \\times 10^{-4}` directly above `0.00111` — the same quantity in two
    notations, in the cell a reader compares."""
    assert column_format([9.34e-4, 1.11e-3, 2.37e-3]) is True
    assert column_format([0.41, 0.83]) is False

    straddling = [
        Summary("a", "rel_l2", 9.34e-4, 7.87e-4, 1.11e-3, 5, 5),
        Summary("b", "rel_l2", 2.37e-3, 2.26e-3, 2.63e-3, 5, 5),
    ]
    tex = summary_table(straddling)
    assert "0.00111" not in tex
    assert tex.count(r"\times 10^{-") == 6


def test_the_generated_label_is_a_legal_latex_identifier(five_seeds):
    """`\\label{tab:rel_l2}` breaks plain LaTeX and hyperref's anchor names."""
    summaries = [summarise(five_seeds, "rel_l2", label="a")]
    assert r"\label{tab:rel-l2}" in summary_table(summaries, label="tab:rel_l2")


def test_the_table_is_booktabs_and_carries_the_seed_count(five_seeds):
    tex = summary_table([summarise(five_seeds, "rel_l2", label="ours")])

    assert r"\toprule" in tex and r"\bottomrule" in tex
    assert r"\hline" not in tex, "booktabs tables never use \\hline"
    assert "|" not in tex, "booktabs tables have no vertical rules"
    assert "ours" in tex and "$n$" in tex


def test_the_failure_column_appears_only_when_something_failed(five_seeds):
    clean = summary_table([summarise(five_seeds, "rel_l2", label="ours")])
    assert "Failed" not in clean

    with_failure = summary_table(
        [summarise(five_seeds + [make_record(seed=9, status=RunStatus.DIVERGED)],
                   "rel_l2", label="ours")]
    )
    assert "Failed" in with_failure


def test_a_label_with_latex_specials_is_escaped():
    s = summarise([make_record()], "rel_l2", label="rar_d & co")
    assert r"rar\_d \& co" in summary_table([s])


# -- the whole loop ------------------------------------------------------------


def test_results_to_figures_and_tables_in_one_command(tmp_path, five_seeds):
    """DESIGN.md §9 step 4's actual deliverable: config -> figure with zero
    manual steps. Everything above tests a piece; this tests that the pieces
    are wired together and that the entry point a paper repo copies works."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from make_figures import main

    results = tmp_path / "results"
    results.mkdir()
    for record in five_seeds:
        write_run(results, record)

    exit_code = main(
        [str(results), "--out", str(tmp_path / "analysis"), "--metrics", "rel_l2"]
    )

    assert exit_code == 0
    analysis = tmp_path / "analysis"
    assert (analysis / "figures" / "convergence_rel_l2_vs_step.pdf").exists()
    assert (analysis / "figures" / "convergence_rel_l2_vs_wall_time.pdf").exists()
    assert r"\toprule" in (analysis / "tables" / "rel_l2.tex").read_text("utf-8")
    # results/ is append-only and derived output never lands in it.
    assert sorted(p.name for p in results.iterdir()) == [
        r.row.run_id for r in five_seeds
    ]


def test_the_loop_reports_an_empty_results_directory(tmp_path):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from make_figures import main

    (tmp_path / "results").mkdir()
    assert main([str(tmp_path / "results"), "--out", str(tmp_path / "out")]) == 1


def test_one_table_holds_one_metric(five_seeds):
    summaries = [
        summarise(five_seeds, "rel_l2", label="a"),
        summarise(five_seeds, "max_error", label="b"),
    ]
    with pytest.raises(ValueError, match="one metric"):
        summary_table(summaries)
