"""Phase 4: the TCO model — KAPI's $0-per-query cost vs a dense+vectorDB recurring bill."""

from __future__ import annotations

from kapi.tco import TCOInputs, compute_tco, format_report


def test_query_cost_is_always_zero():
    r = compute_tco(TCOInputs())
    assert r.breakdown["kapi_query_monthly"] == 0.0


def test_kapi_wins_at_default_scale():
    r = compute_tco(TCOInputs())
    assert r.savings > 0 and 0 < r.savings_pct <= 100
    assert r.breakeven_months < r.months            # dense overtakes within the horizon


def test_more_traffic_widens_savings():
    lo = compute_tco(TCOInputs(queries_per_month=100_000))
    hi = compute_tco(TCOInputs(queries_per_month=10_000_000))
    assert hi.savings > lo.savings                  # KAPI is flat; dense scales with traffic


def test_savings_grow_with_horizon():
    short = compute_tco(TCOInputs(months=1))
    long = compute_tco(TCOInputs(months=36))
    assert long.savings > short.savings


def test_higher_dims_cost_dense_more_ram():
    small = compute_tco(TCOInputs(embedding_dims=384))
    big = compute_tco(TCOInputs(embedding_dims=3072))
    assert big.breakdown["dense_ram_monthly"] > small.breakdown["dense_ram_monthly"]
    assert big.savings > small.savings              # KAPI unaffected by vector width


def test_no_recurring_cost_means_dense_never_overtakes():
    r = compute_tco(TCOInputs(queries_per_month=0, vector_ram_gb_per_1m_docs=0.0,
                              ram_cost_per_gb_month=0.0, vectordb_flat_monthly=0.0))
    assert r.breakeven_months == float("inf")


def test_report_renders_headline():
    inp = TCOInputs()
    text = format_report(inp, compute_tco(inp))
    assert "TCO" in text and "$0.00" in text and "savings" in text.lower()
