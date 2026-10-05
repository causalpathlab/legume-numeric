//! Encoder / decoder trait abstractions shared across topic models.
//!
//! - [`model`]: dense `EncoderModuleT` / `DecoderModuleT` interfaces
//! - [`indexed`]: sparse/indexed variants (`IndexedEncoderT`) used by
//!   top-K embedding pipelines

pub mod indexed;
pub mod model;

pub use indexed::IndexedEncoderT;
pub use model::{
    joint_multinomial_llik, DecoderModuleT, EncoderModuleT, JointDecoderModuleT,
    JointEncoderModuleT, NewDecoder,
};
