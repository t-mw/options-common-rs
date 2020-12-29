use chrono::{DateTime, Duration, NaiveDate, Utc};
use num_rational::Rational64;
use num_traits::{Signed, ToPrimitive, Zero};
use ordered_float::NotNan;

use std::convert::TryInto;
use std::error::Error;
use std::fmt;
use std::str::FromStr;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OptionPosition {
    /// The symbol of the option itself.
    pub symbol: String,

    /// The symbol of the instrument that the option is a derivative of.
    pub underlying_symbol: String,

    pub option_type: OptionType,
    pub strike_price: Rational64,
    pub expiration_date: ExpirationDate,
    pub is_long: bool,

    /// The cost of this position per contract. If the position is long, this should be negative.
    pub unit_cost: Option<Rational64>,

    /// The net liquidity of this position per contract. If the position is long, this should be negative.
    pub unit_net_liq: Option<Rational64>,

    /// The delta of this position per contract.
    pub unit_delta: Option<NotNan<f64>>,

    /// The number of contracts in this position.
    pub quantity: u64,

    /// The lot size per contract. Defaults to 100 if not defined.
    pub lot_size: Option<u64>,
}

impl OptionPosition {
    pub fn description(&self) -> String {
        format!(
            "{} {:.2} {:?}",
            self.expiration_date,
            self.strike_price.to_f64().unwrap(),
            self.option_type,
        )
    }

    pub fn signed_quantity(&self) -> i64 {
        let q: i64 = self.quantity.try_into().unwrap();
        if self.is_long {
            q
        } else {
            -q
        }
    }

    pub fn cost(&self) -> Option<Rational64> {
        let q: i64 = self.quantity.try_into().unwrap();
        self.unit_cost.map(|x| x * q)
    }

    pub fn net_liq(&self) -> Option<Rational64> {
        let q: i64 = self.quantity.try_into().unwrap();
        self.unit_net_liq.map(|x| x * q)
    }

    pub fn delta(&self) -> Option<NotNan<f64>> {
        self.unit_delta.map(|x| x * self.quantity as f64)
    }

    pub fn profit_at_expiry(&self, underlying_price: Rational64) -> Rational64 {
        let lot_size: i64 = self.lot_size.unwrap_or(100).try_into().unwrap();

        let unit_expiry_net_liq = match self.option_type {
            OptionType::Call => underlying_price - self.strike_price,
            OptionType::Put => self.strike_price - underlying_price,
        }
        .max(Rational64::zero())
            * if self.is_long { lot_size } else { -lot_size };

        let q: i64 = self.quantity.try_into().unwrap();
        (self.unit_cost.expect("Undefined cost") + unit_expiry_net_liq) * q
    }

    #[cfg(test)]
    pub fn mock(option_type: OptionType, strike_price: i64, cost: i64, quantity: u64) -> Self {
        let is_long = cost < 0;
        OptionPosition {
            symbol: "OPTION".to_string(),
            underlying_symbol: "ABC".to_string(),
            option_type,
            strike_price: Rational64::from_integer(strike_price),
            expiration_date: Default::default(),
            is_long,
            unit_cost: Some(Rational64::from_integer(cost)),
            unit_net_liq: None,
            unit_delta: None,
            quantity,
            lot_size: None,
        }
    }
}

const MAX_DECIMAL_POINTS: i64 = 4;
const DECIMAL_MULTIPLIER: i64 = 10_000; // 10^MAX_DECIMAL_POINTS

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Decimal(pub Rational64);

impl Decimal {
    pub fn abs(&self) -> Decimal {
        Decimal(self.0.abs())
    }
}

#[derive(Debug, Clone)]
struct PriceFromStrError(String);

impl Error for PriceFromStrError {}

impl fmt::Display for PriceFromStrError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "'{}' could not be parsed as price", self.0)
    }
}

impl FromStr for Decimal {
    type Err = Box<dyn Error>;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let is_negative = s.starts_with('-');

        let mut components = s.split('.');
        let integer = components
            .next()
            .map(|s| i64::from_str(&s.replace(',', "")))
            .transpose()?
            .ok_or_else(|| PriceFromStrError(s.to_string()))?
            .abs();

        let decimal = components
            .next()
            .map(|s| {
                let padded = format!("{:0<4}", s);
                if padded.len() != MAX_DECIMAL_POINTS as usize {
                    let boxed: Box<dyn Error> = PriceFromStrError(s.to_string()).into();
                    return Err(boxed);
                }

                i64::from_str(&padded).map_err(|e| {
                    let boxed: Box<dyn Error> = e.into();
                    boxed
                })
            })
            .transpose()?
            .unwrap_or(0);

        let mut numerator = integer * DECIMAL_MULTIPLIER + decimal;
        if is_negative {
            numerator *= -1;
        }

        let denominator = DECIMAL_MULTIPLIER;

        Ok(Decimal(Rational64::new(numerator, denominator)))
    }
}

impl fmt::Display for Decimal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let integer = self.0.to_integer().abs();

        let decimal = format!(
            "{:0>4}",
            (self.0.fract() * DECIMAL_MULTIPLIER).to_integer().abs()
        );
        let decimal_trimmed = decimal.trim_end_matches('0');

        write!(
            f,
            "{}{}.{}",
            if self.0.is_negative() { "-" } else { "" },
            integer,
            decimal_trimmed
        )
    }
}

impl Default for Decimal {
    fn default() -> Self {
        Decimal(Rational64::zero())
    }
}

#[derive(Copy, Clone, Debug, Eq, Hash, PartialEq)]
pub struct ExpirationDate(pub NaiveDate);

impl ExpirationDate {
    pub fn time_to_expiration(&self, now: Option<fn() -> DateTime<Utc>>) -> Duration {
        let date_now = now.unwrap_or(Utc::now)().naive_utc().date();
        self.0 - date_now
    }
}

impl FromStr for ExpirationDate {
    type Err = chrono::ParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(ExpirationDate(NaiveDate::from_str(s)?))
    }
}

impl Default for ExpirationDate {
    fn default() -> Self {
        ExpirationDate(NaiveDate::from_ymd(1, 1, 1))
    }
}

impl fmt::Display for ExpirationDate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(Copy, Clone, Debug, Eq, Hash, PartialEq)]
pub enum OptionType {
    Call,
    Put,
}

impl Default for OptionType {
    fn default() -> Self {
        OptionType::Call
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StrategyBreakevens {
    // breakevens sorted with ascending price
    pub breakevens: Vec<Breakeven>,
}

impl StrategyBreakevens {
    pub fn min(&self) -> Option<&Breakeven> {
        match (self.breakevens.first(), self.breakevens.len()) {
            (Some(breakeven), 1) => {
                if breakeven.is_ascending {
                    Some(breakeven)
                } else {
                    None
                }
            }
            (breakeven, _) => breakeven,
        }
    }

    pub fn max(&self) -> Option<&Breakeven> {
        match (self.breakevens.last(), self.breakevens.len()) {
            (Some(breakeven), 1) => {
                if !breakeven.is_ascending {
                    Some(breakeven)
                } else {
                    None
                }
            }
            (breakeven, _) => breakeven,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Breakeven {
    pub price: Rational64,
    // whether the profit is increasing with increasing price
    pub is_ascending: bool,
}

// options should be sorted by strike price
pub fn calculate_breakevens_for_strategy(options: &[OptionPosition]) -> StrategyBreakevens {
    if options.is_empty() {
        return StrategyBreakevens { breakevens: vec![] };
    }

    let max_strike_price = options
        .iter()
        .map(|option| option.strike_price)
        .max()
        .unwrap();

    let profit_at_price = |price| {
        let profit: Rational64 = options
            .iter()
            .map(|option| option.profit_at_expiry(price))
            .sum();
        profit
    };

    let price_range = (Rational64::zero(), max_strike_price * 100);

    let mut prev_price = price_range.0;
    let mut prev_profit = profit_at_price(prev_price);

    let mut breakevens = vec![];

    for strike_price in options
        .iter()
        .map(|option| option.strike_price)
        .chain(std::iter::once(price_range.1))
    {
        if strike_price == prev_price {
            continue;
        }

        assert!(
            strike_price > prev_price,
            "Options should be sorted by strike price"
        );

        let profit = profit_at_price(strike_price);
        if profit.is_negative() != prev_profit.is_negative() {
            let x = strike_price - prev_price;
            let y = profit - prev_profit;

            let dy = -prev_profit / y;
            breakevens.push(Breakeven {
                price: prev_price + x * dy,
                is_ascending: prev_profit.is_negative(),
            });
        }

        prev_price = strike_price;
        prev_profit = profit;
    }

    StrategyBreakevens { breakevens }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StrategyProfitBounds {
    pub max_loss: Option<ProfitBound>,
    pub max_profit: Option<ProfitBound>,
}

impl StrategyProfitBounds {
    pub fn to_percentage_of_max_profit(&self, profit: Rational64) -> Option<f64> {
        self.max_profit
            .as_ref()
            .and_then(|b| b.finite_value())
            .map(|value| {
                debug_assert!(value.is_positive());
                (profit / value.abs()).to_f64().unwrap()
            })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ProfitBound {
    Infinite,
    Finite {
        value: Rational64,
        price: Rational64,
    },
}

impl ProfitBound {
    pub fn finite_value(&self) -> Option<Rational64> {
        match self {
            ProfitBound::Finite { value, .. } => Some(*value),
            _ => None,
        }
    }
}

pub fn calculate_profit_bounds_for_strategy(options: &[OptionPosition]) -> StrategyProfitBounds {
    if options.is_empty() {
        return StrategyProfitBounds {
            max_loss: None,
            max_profit: None,
        };
    }

    let mut min_gradient: i64 = 0;
    let mut max_gradient: i64 = 0;
    for option in options {
        match (option.option_type, option.is_long) {
            (OptionType::Call, true) => max_gradient += option.quantity as i64,
            (OptionType::Call, false) => max_gradient -= option.quantity as i64,
            (OptionType::Put, true) => min_gradient += option.quantity as i64,
            (OptionType::Put, false) => min_gradient -= option.quantity as i64,
        }
    }

    let max_loss_at_strike = {
        let mut max_loss = Rational64::from_integer(i64::MAX);
        let mut max_loss_price = Rational64::zero();
        for option in options {
            let price = option.strike_price;
            let profit_at_price = options.iter().map(|o| o.profit_at_expiry(price)).sum();
            if profit_at_price < max_loss {
                max_loss = profit_at_price;
                max_loss_price = price;
            }
        }
        ProfitBound::Finite {
            value: max_loss,
            price: max_loss_price,
        }
    };

    let max_profit_at_strike = {
        let mut max_profit = Rational64::from_integer(i64::MIN);
        let mut max_profit_price = Rational64::zero();
        for option in options {
            let price = option.strike_price;
            let profit_at_price = options.iter().map(|o| o.profit_at_expiry(price)).sum();
            if profit_at_price > max_profit {
                max_profit = profit_at_price;
                max_profit_price = price;
            }
        }
        ProfitBound::Finite {
            value: max_profit,
            price: max_profit_price,
        }
    };

    let max_loss = if max_gradient < 0 {
        ProfitBound::Infinite
    } else if min_gradient < 0 {
        let price = Rational64::zero();
        let profit_at_zero = options.iter().map(|o| o.profit_at_expiry(price)).sum();

        // profit at zero may not necessarily be the extreme
        if max_loss_at_strike.finite_value().unwrap() <= profit_at_zero {
            max_loss_at_strike
        } else {
            ProfitBound::Finite {
                value: profit_at_zero,
                price,
            }
        }
    } else {
        max_loss_at_strike
    };

    let max_profit = if max_gradient > 0 {
        ProfitBound::Infinite
    } else if min_gradient > 0 {
        let price = Rational64::zero();
        let profit_at_zero = options.iter().map(|o| o.profit_at_expiry(price)).sum();

        // profit at zero may not necessarily be the extreme
        if max_profit_at_strike.finite_value().unwrap() >= profit_at_zero {
            max_profit_at_strike
        } else {
            ProfitBound::Finite {
                value: profit_at_zero,
                price,
            }
        }
    } else {
        max_profit_at_strike
    };

    StrategyProfitBounds {
        max_loss: Some(max_loss).filter(|b| {
            let finite = b.finite_value();
            finite.is_none() || finite.filter(|v| v.is_negative()).is_some()
        }),
        max_profit: Some(max_profit).filter(|b| {
            let finite = b.finite_value();
            finite.is_none() || finite.filter(|v| v.is_positive()).is_some()
        }),
    }
}

pub trait ExpirationImpliedVolatilityProvider {
    fn find_iv_for_expiration_date(&self, date: ExpirationDate) -> Option<f64>;
}

pub fn calculate_pop_for_breakevens(
    breakevens: &StrategyBreakevens,
    profit_bounds: &StrategyProfitBounds,
    underlying_price: Rational64,
    iv_provider: &impl ExpirationImpliedVolatilityProvider,
    expiration_date: ExpirationDate,
    now: Option<fn() -> DateTime<Utc>>,
) -> Option<i32> {
    if breakevens.min().is_none() && breakevens.max().is_none() {
        if profit_bounds.max_loss.is_none() {
            return Some(100);
        } else {
            debug_assert!(profit_bounds.max_profit.is_none());
            return Some(0);
        }
    }

    let mut pop = if breakevens.breakevens.first().unwrap().is_ascending {
        0.0
    } else {
        1.0
    };

    for Breakeven {
        price,
        is_ascending,
    } in &breakevens.breakevens
    {
        if *is_ascending {
            pop += calculate_probability_of_expiring_gt_price(
                *price,
                underlying_price,
                iv_provider,
                expiration_date,
                now,
            )?;
        } else {
            pop -= calculate_probability_of_expiring_gt_price(
                *price,
                underlying_price,
                iv_provider,
                expiration_date,
                now,
            )?;
        }
    }

    Some((pop * 100.0).round() as i32)
}

fn calculate_probability_of_expiring_gt_price(
    price: Rational64,
    underlying_price: Rational64,
    iv_provider: &impl ExpirationImpliedVolatilityProvider,
    expiration_date: ExpirationDate,
    now: Option<fn() -> DateTime<Utc>>,
) -> Option<f64> {
    let stock_price = underlying_price.to_f64()?;
    let expiration_implied_volatility = iv_provider.find_iv_for_expiration_date(expiration_date)?;

    let time = expiration_date.time_to_expiration(now).num_minutes() as f64
        / Duration::days(365).num_minutes() as f64;
    let vol = expiration_implied_volatility * time.sqrt();

    // https://www.ltnielsen.com/wp-content/uploads/Understanding.pdf
    let interest_rate = 0.05;
    let d2 =
        ((interest_rate - 0.5 * vol * vol) * time - (price.to_f64()? / stock_price).ln()) / vol;

    use statrs::distribution::{Normal, Univariate};
    let prob = Normal::new(0.0, 1.0).unwrap().cdf(d2);

    Some(prob)
}

#[cfg(test)]
mod tests {
    use super::*;

    use chrono::{NaiveDate, TimeZone};
    use num_traits::One;

    #[test]
    fn test_decimal_from_str() {
        assert_eq!(
            Decimal::from_str("0.3").unwrap(),
            Decimal(Rational64::new(3, 10))
        );
        assert_eq!(
            Decimal::from_str("-0.3").unwrap(),
            Decimal(Rational64::new(-3, 10))
        );
        assert_eq!(
            Decimal::from_str("9.12").unwrap(),
            Decimal(Rational64::new(912, 100))
        );
        assert_eq!(
            Decimal::from_str("-9.12").unwrap(),
            Decimal(Rational64::new(-912, 100))
        );
        assert_eq!(
            Decimal::from_str("23.012").unwrap(),
            Decimal(Rational64::new(23012, 1000))
        );
        assert_eq!(
            Decimal::from_str("1.0001").unwrap(),
            Decimal(Rational64::new(10001, 10000))
        );
        assert_eq!(
            Decimal::from_str("12,345.4321").unwrap(),
            Decimal(Rational64::new(123454321, 10000))
        );
    }

    #[test]
    fn test_decimal_to_str() {
        assert_eq!(Decimal(Rational64::new(3, 10)).to_string(), "0.3",);
        assert_eq!(Decimal(Rational64::new(-3, 10)).to_string(), "-0.3",);
        assert_eq!(Decimal(Rational64::new(912, 100)).to_string(), "9.12",);
        assert_eq!(Decimal(Rational64::new(-912, 100)).to_string(), "-9.12",);
        assert_eq!(Decimal(Rational64::new(23012, 1000)).to_string(), "23.012",);
        assert_eq!(Decimal(Rational64::new(10001, 10000)).to_string(), "1.0001",);
    }

    #[test]
    fn test_short_call_profit_at_expiry() {
        let option = OptionPosition::mock(OptionType::Call, 100, 300, 1);
        let underlying_price = Rational64::from_integer(101);
        let profit = option.profit_at_expiry(underlying_price);

        assert_eq!(profit, Rational64::from_integer(200));
    }

    #[test]
    fn test_short_put_profit_at_expiry() {
        let option = OptionPosition::mock(OptionType::Put, 100, 300, 1);
        let underlying_price = Rational64::from_integer(99);
        let profit = option.profit_at_expiry(underlying_price);

        assert_eq!(profit, Rational64::from_integer(200));
    }

    #[test]
    fn test_long_put_profit_at_expiry() {
        let option = OptionPosition::mock(OptionType::Put, 100, -300, 1);
        let underlying_price = Rational64::from_integer(99);
        let profit = option.profit_at_expiry(underlying_price);

        assert_eq!(profit, Rational64::from_integer(-200));
    }

    #[test]
    fn test_calculate_breakevens_for_short_strangle() {
        let options = [
            OptionPosition::mock(OptionType::Put, 20, 37, 1),
            OptionPosition::mock(OptionType::Call, 28, 74, 1),
        ];

        let breakevens = calculate_breakevens_for_strategy(&options);

        assert_eq!(
            breakevens,
            StrategyBreakevens {
                breakevens: vec![
                    Breakeven {
                        price: Rational64::new(1889, 100),
                        is_ascending: true
                    },
                    Breakeven {
                        price: Rational64::new(2911, 100),
                        is_ascending: false
                    }
                ]
            }
        );
    }

    #[test]
    fn test_calculate_breakevens_for_short_call_ratio_spread() {
        let options = [
            OptionPosition::mock(OptionType::Call, 15, -305, 1),
            OptionPosition::mock(OptionType::Call, 20, 217, 2),
        ];

        let breakevens = calculate_breakevens_for_strategy(&options);

        assert_eq!(
            breakevens,
            StrategyBreakevens {
                breakevens: vec![Breakeven {
                    price: Rational64::new(2629, 100),
                    is_ascending: false
                }]
            }
        );
    }

    #[test]
    fn test_calculate_profit_for_short_strangle() {
        let options = [
            OptionPosition::mock(OptionType::Put, 20, 37, 1),
            OptionPosition::mock(OptionType::Call, 28, 74, 1),
        ];

        let profit_bounds = calculate_profit_bounds_for_strategy(&options);

        assert_eq!(
            profit_bounds,
            StrategyProfitBounds {
                max_loss: Some(ProfitBound::Infinite),
                max_profit: Some(ProfitBound::Finite {
                    value: Rational64::from_integer(37 + 74),
                    price: Rational64::from_integer(20)
                })
            }
        );
    }

    #[test]
    fn test_calculate_profit_for_long_strangle() {
        let options = [
            OptionPosition::mock(OptionType::Put, 20, -37, 1),
            OptionPosition::mock(OptionType::Call, 28, -74, 1),
        ];

        let profit_bounds = calculate_profit_bounds_for_strategy(&options);

        assert_eq!(
            profit_bounds,
            StrategyProfitBounds {
                max_loss: Some(ProfitBound::Finite {
                    value: Rational64::from_integer(-37 - 74),
                    price: Rational64::from_integer(20)
                }),
                max_profit: Some(ProfitBound::Infinite),
            }
        );
    }

    #[test]
    fn test_calculate_profit_for_short_call_ratio_spread() {
        let options = [
            OptionPosition::mock(OptionType::Call, 15, -305, 1),
            OptionPosition::mock(OptionType::Call, 20, 217, 2),
        ];

        let profit_bounds = calculate_profit_bounds_for_strategy(&options);

        assert_eq!(
            profit_bounds,
            StrategyProfitBounds {
                max_loss: Some(ProfitBound::Infinite),
                max_profit: Some(ProfitBound::Finite {
                    value: Rational64::from_integer(-305 + 2 * 217 + 500),
                    price: Rational64::from_integer(20)
                })
            }
        );
    }

    #[test]
    fn test_calculate_profit_for_long_put_ratio_spread() {
        let options = [
            OptionPosition::mock(OptionType::Put, 20, 305, 1),
            OptionPosition::mock(OptionType::Put, 15, -217, 2),
        ];

        let profit_bounds = calculate_profit_bounds_for_strategy(&options);

        let max_loss = 305 - 2 * 217 - 500;
        let max_profit = max_loss + 1500;
        assert_eq!(
            profit_bounds,
            StrategyProfitBounds {
                max_loss: Some(ProfitBound::Finite {
                    value: Rational64::from_integer(max_loss),
                    price: Rational64::from_integer(15)
                }),
                max_profit: Some(ProfitBound::Finite {
                    value: Rational64::from_integer(max_profit),
                    price: Rational64::from_integer(0)
                }),
            }
        );
    }

    #[test]
    fn test_calculate_pop_no_loss() {
        struct IVProvider;
        impl ExpirationImpliedVolatilityProvider for IVProvider {
            fn find_iv_for_expiration_date(&self, _: ExpirationDate) -> Option<f64> {
                None
            }
        }

        let pop = calculate_pop_for_breakevens(
            &StrategyBreakevens { breakevens: vec![] },
            &StrategyProfitBounds {
                max_loss: None,
                max_profit: Some(ProfitBound::Infinite),
            },
            Rational64::one(),
            &IVProvider,
            ExpirationDate(NaiveDate::from_ymd(2020, 10, 16)),
            None,
        );
        assert_eq!(pop, Some(100));
    }

    #[test]
    fn test_calculate_pop_multiple_breakevens() {
        let option_positions = [
            OptionPosition::mock(OptionType::Put, 20, -55, 2),
            OptionPosition::mock(OptionType::Put, 30, 277, 1),
            OptionPosition::mock(OptionType::Call, 60, -823, 1),
            OptionPosition::mock(OptionType::Call, 85, 456, 2),
        ];

        let breakevens = calculate_breakevens_for_strategy(&option_positions);

        assert_eq!(
            breakevens,
            StrategyBreakevens {
                breakevens: vec![
                    Breakeven {
                        price: Rational64::new(314, 25),
                        is_ascending: false,
                    },
                    Breakeven {
                        price: Rational64::new(686, 25),
                        is_ascending: true,
                    },
                    Breakeven {
                        price: Rational64::new(2814, 25),
                        is_ascending: false,
                    },
                ]
            }
        );

        fn expiration_date() -> ExpirationDate {
            ExpirationDate(NaiveDate::from_ymd(2020, 10, 16))
        };

        struct IVProvider;
        impl ExpirationImpliedVolatilityProvider for IVProvider {
            fn find_iv_for_expiration_date(&self, date: ExpirationDate) -> Option<f64> {
                if date == expiration_date() {
                    Some(2.00)
                } else {
                    None
                }
            }
        }

        let profit_bounds = calculate_profit_bounds_for_strategy(&option_positions);

        assert_eq!(
            profit_bounds,
            StrategyProfitBounds {
                max_loss: Some(ProfitBound::Infinite),
                max_profit: Some(ProfitBound::Finite {
                    value: Rational64::from_integer(2756),
                    price: Rational64::from_integer(85)
                },),
            }
        );

        let pop = calculate_pop_for_breakevens(
            &breakevens,
            &profit_bounds,
            Rational64::new(475, 10),
            &IVProvider,
            expiration_date(),
            Some(|| Utc.ymd(2020, 9, 18).and_hms(1, 1, 1)),
        );
        assert_eq!(pop, Some(79));
    }
}
