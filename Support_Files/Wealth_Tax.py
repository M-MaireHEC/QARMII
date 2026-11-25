from decimal import Decimal, ROUND_HALF_UP
from typing import Final

# Barème de l'impôt sur la fortune 2025, coefficient 100 %.
# The published table (50'000 to 2'000'000 CHF, steps of 1'000) is a piecewise linear scale:
# each extra 1'000 CHF of taxable wealth costs the marginal rate below (in per mille),
# and the base tax is rounded to 0.05 CHF. Above 2'000'000 every extra 1'000 is taxed at 3.39 per mille.
WEALTH_TAX_START: Final = (50_000, Decimal("22.95"))  # first row of the table: wealth, base tax
WEALTH_TAX_BRACKETS: Final = [  # (upper wealth of the bracket, marginal rate in per mille)
    (95_000, Decimal("0.97")),
    (177_000, Decimal("1.69")),
    (355_000, Decimal("2.42")),
    (710_000, Decimal("3.15")),
    (2_000_000, Decimal("3.39")),
]
WEALTH_TAX_TOP_RATE: Final = Decimal("3.39")  # per mille, above 2'000'000


def _round_05(x: Decimal) -> Decimal:
    return (x * 20).quantize(Decimal("1"), rounding=ROUND_HALF_UP) / 20


def _exact_tax(wealth: float) -> Decimal:
    # tax before the final rounding: the table rounds the tax to 0.05 but computes the rate from this value.
    # The tax at each bracket limit is rounded to 0.05 before the next bracket starts (matches the table).
    start, tax = WEALTH_TAX_START
    if wealth < start:
        raise ValueError(f"The table starts at {start} CHF, got {wealth}")
    wealth = Decimal(str(wealth))
    low = Decimal(start)
    for high, rate in WEALTH_TAX_BRACKETS:
        if wealth <= high:
            return tax + (wealth - low) * rate / 1000
        tax = _round_05(tax + (high - low) * rate / 1000)
        low = Decimal(high)
    return tax + (wealth - low) * WEALTH_TAX_TOP_RATE / 1000


def wealth_base_tax(wealth: float) -> float:
    """Base tax in CHF (coefficient 100 %) for a taxable wealth >= 50'000 CHF, rounded to 0.05 like the table."""
    return float(_round_05(_exact_tax(wealth)))


def wealth_tax_rate(wealth: float) -> float:
    """Average tax rate as a fraction of wealth (0.003064 = 3.064 per mille = 0.3064 %), rounded like the table."""
    rate_per_mille = _exact_tax(wealth) / Decimal(str(wealth)) * 1000
    return float(rate_per_mille.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)) / 1000


# {taxable wealth in CHF: average tax rate as a fraction}, the 1'951 rows of the published table
WEALTH_TAX_RATE: Final = {w: wealth_tax_rate(w) for w in range(50_000, 2_000_001, 1_000)}


if __name__ == "__main__":
    print(len(WEALTH_TAX_RATE), WEALTH_TAX_RATE[50_000], WEALTH_TAX_RATE[2_000_000])
