use super::Float;
use serde::{
  de::{Deserialize, Deserializer, Error, Visitor},
  ser::{Serialize, Serializer},
};
use std::fmt;

impl Serialize for Float {
  fn serialize<S>(&self, s: S) -> std::result::Result<S::Ok, S::Error>
  where
    S: Serializer,
  {
    s.serialize_str(&self.to_string())
  }
}

struct V;

impl<'de> Visitor<'de> for V {
  type Value = Float;

  fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.write_str("a floating-point number")
  }

  fn visit_str<E>(self, v: &str) -> Result<Float, E>
  where
    E: Error,
  {
    match v {
      "Infinity" => Ok(Float::MAX),
      "-Infinity" => Ok(Float::MIN),
      "NaN" => Ok(Float::NAN),
      _ => v.parse().map_err(Error::custom),
    }
  }
}

impl<'de> Deserialize<'de> for Float {
  fn deserialize<D>(de: D) -> Result<Self, D::Error>
  where
    D: Deserializer<'de>,
  {
    de.deserialize_str(V)
  }
}
