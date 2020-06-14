//! `hugefloat` provides a numeric type that acts like [`f64`] with a much
//! higher maximum value.
//!
//! This library is NOT intended for usecases that actually require precise
//! arithmetic. Use [`gmp`] or [`rug`] if you need that. `hugefloat` is
//! designed to be lightweight and have no foreign dependencies so it can easily
//! be compiled to WASM. It's intended for use in incremental and idle games
//! which must handle extremely large numbers.
//!
//! # Features
//!
//! Enable the `serde` feature of this crate to support de/serialization via
//! serde.
//!
//! [`gmp`]: https://crates.io/crates/gmp
//! [`rug`]: https://crates.io/crates/rug

use std::{
  cmp::Ordering,
  f64::consts,
  fmt::{self, Display},
  ops::*,
  str::FromStr,
};
use thiserror::Error;

#[cfg(feature = "serde")] mod serde;
mod table;

/// Parsing a `Float` can fail either while parsing the base (which is a
/// float) or the exponent (which is an int).
#[derive(Error, Debug, Eq, PartialEq, Clone)]
pub enum ParseError {
  #[error("{0}")]
  Float(#[from] std::num::ParseFloatError),
  #[error("{0}")]
  Int(#[from] std::num::ParseIntError),
}

/// A number that behaves mostly like a float, but with an extremely large
/// maximum value.
///
/// To construct a `Float` outside the range of `f64`, use either
/// [`Float::sci`] or the [`FromStr`] implementation.
///
/// ```
/// # use hugefloat::Float;
/// assert_eq!("1".parse(), Ok(Float::sci(1.0, 0)));
/// assert_eq!("1e10000".parse(), Ok(Float::sci(1.0, 10000)));
/// ```
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Float {
  mantissa: f64,
  exponent: i64,
}

const MAX_DIGITS: i64 = 17;
const MAX_EXPONENT: i64 = 9_000_000_000_000_000;

impl Float {
  /// The largest value that `Float` can represent:
  /// `1e9_000_000_000_000_000`.
  pub const MAX: Float = Float {
    exponent: MAX_EXPONENT,
    mantissa: 1.0,
  };
  /// The smallest value that `Float` can represent. Equivalent to `-MAX`.
  pub const MIN: Float = Float {
    exponent: MAX_EXPONENT,
    mantissa: -1.0,
  };
  pub const NAN: Float = Float {
    exponent: 0,
    mantissa: std::f64::NAN,
  };

  pub fn mantissa(self) -> f64 {
    self.mantissa
  }

  pub fn exponent(self) -> i64 {
    self.exponent
  }

  #[doc(hidden)]
  pub fn normalize(mut self) -> Self {
    if !self.mantissa.is_finite() {
      return self;
    }
    while self.mantissa.abs() < 1.0 && self.mantissa != 0.0 {
      self.mantissa *= 10.0;
      self.exponent -= 1;
    }
    while self.mantissa.abs() >= 10.0 {
      self.mantissa /= 10.0;
      self.exponent += 1;
    }
    self
  }

  /// Construct a `Float` from a base and exponent.
  pub fn sci(base: f64, exponent: i64) -> Self {
    if !base.is_finite() {
      Self::NAN
    } else {
      Float {
        mantissa: base,
        exponent,
      }
      .normalize()
    }
  }

  /// Convert a regular float to a `Float`.
  pub fn float(value: f64) -> Self {
    if value.is_nan() {
      Self::NAN
    } else if value.is_infinite() {
      if value.is_sign_positive() {
        Self::MAX
      } else {
        Self::MIN
      }
    } else if value == 0.0 || value == -0.0 {
      Float {
        mantissa: 0.0,
        exponent: 0,
      }
    } else {
      let exponent = value.abs().log10().floor() as i64;
      let base = table::POWERS[(exponent + table::TABLE_CENTER as i64) as usize];
      let mantissa = value / base;
      Float { exponent, mantissa }.normalize()
    }
  }

  /// Convert an integer value to a `Float`.
  pub fn int(value: i64) -> Self {
    Self::float(value as f64)
  }

  /// Try to convert `self` to `f64`. `None` if `self` does not fit in f64, i.e.
  /// if it's higher than `f64::MAX` or lower than `f64::MIN`.
  pub fn try_float(self) -> Option<f64> {
    let f = self.to_float();
    if !f.is_finite() {
      None
    } else {
      Some(f)
    }
  }

  /// Like `try_float`, but overflows will be represented as positive or
  /// negative infinity.
  ///
  /// ```
  /// # use hugefloat::Float;
  /// assert_eq!(Float::sci(1.0, 10).to_float(), 1.0e10);
  /// assert!(Float::sci(1.0, 10000).to_float().is_infinite());
  /// ```
  pub fn to_float(self) -> f64 {
    dbg!(self.mantissa) * dbg!(10.0f64.powi(dbg!(self.exponent as i32)))
  }

  pub fn abs(mut self) -> Self {
    self.mantissa = self.mantissa.abs();
    self
  }

  pub fn signum(self) -> f64 {
    self.mantissa.signum()
  }

  pub fn recip(self) -> Self {
    Self::sci(1.0 / self.mantissa, -self.exponent)
  }

  pub fn round(self) -> Self {
    if self.exponent.abs() < 308 {
      Self::float(self.to_float().round())
    } else {
      self
    }
  }

  pub fn floor(self) -> Self {
    if self.exponent.abs() < 308 {
      Self::float(self.to_float().floor())
    } else {
      self
    }
  }

  pub fn ceil(self) -> Self {
    if self.exponent.abs() < 308 {
      Self::float(self.to_float().ceil())
    } else {
      self
    }
  }

  pub fn trunc(self) -> Self {
    if self.exponent.abs() < 308 {
      Self::float(self.to_float().trunc())
    } else {
      self
    }
  }

  pub fn log10(self) -> f64 {
    (self.exponent as f64) + self.mantissa.log10()
  }

  pub fn log2(self) -> f64 {
    self.log10() * consts::LOG2_10
  }

  pub fn ln(self) -> f64 {
    self.log10() * consts::LN_10
  }

  /// This method panics if `base < 2.0`.
  pub fn log(self, base: f64) -> f64 {
    assert!(
      base >= 2.0,
      "cannot call Float::log() with a base lower than 2"
    );
    (consts::LN_10 / base.ln()) * self.log10()
  }

  pub fn powf(self, n: f64) -> Self {
    Self::float(n * self.ln()).exp()
  }

  pub fn powi(self, n: i32) -> Self {
    Self::sci(self.mantissa.powi(n), self.exponent * (n as i64))
  }

  pub fn exp(self) -> Self {
    let f = self.to_float();
    if -706.0 < f && f < 709.0 {
      Self::float(f.exp())
    } else {
      let mut x = self;
      let mut exp = 0f64;
      let expx = self.exponent;
      let ln10 = Self::float(consts::LN_10);

      if expx >= 0 {
        exp = (x / ln10).to_float().trunc();
        let tmp = Self::float(exp) * ln10;
        x -= tmp;
        if x >= ln10 {
          exp += 1.0;
          x -= ln10;
        }
      }
      if x.signum() < 0.0 {
        exp -= 1.0;
        x += ln10;
      }

      let nextx = x.to_float().exp();

      if exp != 0.0 {
        x = Self::sci(nextx, exp.floor() as i64);
      }

      x
    }
  }

  pub fn sqrt(self) -> Self {
    self.powf(0.5)
  }

  pub fn cbrt(self) -> Self {
    self.powf(1.0 / 3.0)
  }
}

impl From<i64> for Float {
  fn from(i: i64) -> Self {
    Self::int(i)
  }
}

impl From<f64> for Float {
  fn from(f: f64) -> Self {
    Self::float(f)
  }
}

impl FromStr for Float {
  type Err = ParseError;

  fn from_str(input: &str) -> Result<Self, Self::Err> {
    let mut sci_parts = input.splitn(2, 'e');
    let first = sci_parts
      .next()
      .expect("split should never yield zero elements");
    if let Some(second) = sci_parts.next() {
      Ok(Self::sci(first.parse()?, second.parse()?))
    } else {
      Ok(Self::float(first.parse()?))
    }
  }
}

impl Display for Float {
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
    let Float { mantissa, exponent } = *self;
    if mantissa.is_nan() {
      write!(f, "NaN")
    } else if exponent >= MAX_EXPONENT {
      if mantissa.is_sign_positive() {
        write!(f, "Infinity")
      } else {
        write!(f, "-Infinity")
      }
    } else if exponent <= -MAX_EXPONENT || mantissa == 0.0 {
      write!(f, "0")
    } else if exponent >= MAX_DIGITS as i64 {
      write!(
        f,
        "{value:0<prec$}",
        value = mantissa.to_string().replace(".", ""),
        prec = (exponent as usize) + 1
      )
    } else {
      write!(
        f,
        "{value:.prec$}",
        value = self.to_float(),
        prec = f.precision().unwrap_or(0)
      )
    }
  }
}

impl fmt::LowerExp for Float {
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
    format_exp(*self, 'e', f)
  }
}

impl fmt::UpperExp for Float {
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
    format_exp(*self, 'E', f)
  }
}

fn format_exp(h: Float, chr: char, f: &mut fmt::Formatter) -> fmt::Result {
  let Float { mantissa, exponent } = h;
  if mantissa.is_nan() {
    write!(f, "NaN")
  } else if exponent >= MAX_EXPONENT {
    if mantissa.is_sign_positive() {
      write!(f, "Infinity")
    } else {
      write!(f, "-Infinity")
    }
  } else if exponent <= -MAX_EXPONENT || mantissa == 0.0 {
    write!(f, "0")
  } else {
    write!(
      f,
      "{base:.prec$}{e}{exp}",
      e = chr,
      base = mantissa,
      exp = exponent,
      prec = f.precision().unwrap_or(0)
    )
  }
}

impl Neg for Float {
  type Output = Self;

  fn neg(self) -> Self::Output {
    Self::sci(-self.mantissa, self.exponent)
  }
}

impl Add<Float> for Float {
  type Output = Self;

  fn add(self, other: Self) -> Self::Output {
    let (bigger, smaller) = if self.exponent >= other.exponent {
      (self, other)
    } else {
      (other, self)
    };
    if bigger.exponent - smaller.exponent > MAX_DIGITS as i64 {
      bigger
    } else {
      let factor =
        table::POWERS[((smaller.exponent - bigger.exponent) + table::TABLE_CENTER as i64) as usize];
      Self::sci(bigger.mantissa + smaller.mantissa * factor, bigger.exponent)
    }
  }
}

impl Add<i64> for Float {
  type Output = Self;

  fn add(self, other: i64) -> Self::Output {
    self + Self::int(other)
  }
}

impl Add<f64> for Float {
  type Output = Self;

  fn add(self, other: f64) -> Self::Output {
    self + Self::float(other)
  }
}

impl<T> AddAssign<T> for Float
where
  Float: Add<T, Output = Float>,
{
  fn add_assign(&mut self, other: T) {
    *self = self.add(other)
  }
}

impl Sub<Float> for Float {
  type Output = Self;

  fn sub(self, other: Float) -> Self::Output {
    self + -other
  }
}

impl Sub<i64> for Float {
  type Output = Self;

  fn sub(self, other: i64) -> Self::Output {
    self + -other
  }
}

impl Sub<f64> for Float {
  type Output = Self;

  fn sub(self, other: f64) -> Self::Output {
    self + -other
  }
}

impl<T> SubAssign<T> for Float
where
  Float: Sub<T, Output = Float>,
{
  fn sub_assign(&mut self, other: T) {
    *self = self.sub(other)
  }
}

impl Mul<Float> for Float {
  type Output = Self;

  #[allow(clippy::suspicious_arithmetic_impl)]
  fn mul(self, other: Self) -> Self::Output {
    Self::sci(
      self.mantissa * other.mantissa,
      self.exponent + other.exponent,
    )
  }
}

impl Mul<i64> for Float {
  type Output = Self;

  fn mul(self, other: i64) -> Self::Output {
    self * Self::int(other)
  }
}

impl Mul<f64> for Float {
  type Output = Self;

  fn mul(self, other: f64) -> Self::Output {
    self * Self::float(other)
  }
}

impl<T> MulAssign<T> for Float
where
  Float: Mul<T, Output = Float>,
{
  fn mul_assign(&mut self, other: T) {
    *self = self.mul(other)
  }
}

impl Div<Float> for Float {
  type Output = Self;

  #[allow(clippy::suspicious_arithmetic_impl)]
  fn div(self, other: Self) -> Self::Output {
    self * other.recip()
  }
}

impl Div<i64> for Float {
  type Output = Self;

  fn div(self, other: i64) -> Self::Output {
    self / Self::int(other)
  }
}

impl Div<f64> for Float {
  type Output = Self;

  fn div(self, other: f64) -> Self::Output {
    self / Self::float(other)
  }
}

impl<T> DivAssign<T> for Float
where
  Float: Div<T, Output = Float>,
{
  fn div_assign(&mut self, other: T) {
    *self = self.div(other)
  }
}

impl PartialOrd for Float {
  fn partial_cmp(&self, value: &Self) -> Option<Ordering> {
    if !self.mantissa.is_finite() || !value.mantissa.is_finite() {
      return None;
    }
    if self.mantissa == 0.0 || value.mantissa == 0.0 {
      return self.mantissa.partial_cmp(&value.mantissa);
    }
    if self.mantissa > 0.0 {
      if value.mantissa < 0.0 {
        return Some(Ordering::Greater);
      }
      if self.exponent > value.exponent {
        return Some(Ordering::Greater);
      }
      if self.exponent < value.exponent {
        return Some(Ordering::Less);
      }
      if self.mantissa > value.mantissa {
        return Some(Ordering::Greater);
      }
      if self.mantissa < value.mantissa {
        return Some(Ordering::Less);
      }
      return Some(Ordering::Equal);
    } else if self.mantissa < 0.0 {
      if value.mantissa > 0.0 {
        return Some(Ordering::Less);
      }
      if self.exponent > value.exponent {
        return Some(Ordering::Less);
      }
      if self.exponent < value.exponent {
        return Some(Ordering::Greater);
      }
      if self.mantissa > value.mantissa {
        return Some(Ordering::Less);
      }
      if self.mantissa < value.mantissa {
        return Some(Ordering::Greater);
      }
      return Some(Ordering::Equal);
    }
    None
  }
}

#[test]
fn test_format() {
  let h = Float::sci(1.0, 10);
  assert_eq!(format!("{}", h), "10000000000");
  assert_eq!(format!("{:.3}", h), "10000000000.000");

  let h = Float::sci(1.0, 10000);
  assert_eq!(format!("{:e}", h), "1e10000");
  assert_eq!(format!("{:.3e}", h), "1.000e10000");
  assert_eq!(format!("{:.3e}", h.powi(2)), "1.000e20000");
}

#[test]
fn test_math_ops() {
  let h = Float::float(1.234);
  assert_eq!(-h, Float::float(-1.234));

  // need rounding of course due to float imprecision
  assert_eq!(Float::float(9.0).sqrt().round(), Float::float(3.0));
  assert_eq!(Float::float(9.0).powf(2.0).round(), Float::float(81.0));

  // exp > 308
  let mut bignum = Float::sci(1.2345, 347);

  bignum = bignum.powf(2.0);
  assert_eq!(bignum, Float::sci(1.523990249999763, 694));
  bignum = bignum.powf(56.1);
  assert_eq!(bignum, Float::sci(4.627013609064963, 38943));
}
