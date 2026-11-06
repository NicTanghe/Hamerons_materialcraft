use super::ATOMIC;
use crate::{CookContext, CookResult, NodeDefinition, node};
use materialcraft_image::{Image, Kind};
use std::sync::Arc;

pub(super) fn definition() -> NodeDefinition {
    node("distort", "Distort", ATOMIC)
        .desc("Moves each input pixel from UV to UV + RG from Input 2. Zero is neutral; signed R/G set direction and distance in UV units. B/A are ignored. Overlaps are averaged; holes are zero. Import vectors as Linear data.")
        .atomic()
        .any_in("input", "Input")
        .color_in("vector", "Input 2 (RG vector)")
        .any_out()
        .cook(distort)
}

fn distort(context: &CookContext) -> CookResult {
    let input = context.any_or(0, Kind::Gray, [0.0; 4]);
    if context.input(1).is_none() {
        return Ok(vec![input]);
    }

    let vector = context.any_or(1, Kind::Color, [0.0; 4]);
    let mut output = Image::filled(input.w, input.h, input.kind, [0.0; 4]);
    let channels = output.channels();
    let mut weights = vec![0.0_f32; (output.w * output.h) as usize];
    let data = Arc::make_mut(&mut output.data);
    // Source order makes overlapping contributions deterministic on every platform.
    for y in 0..input.h {
        for x in 0..input.w {
            let [r, g, ..] = vector.get(x as i64, y as i64, false);
            let Some(dx) = destination(x, r, input.w, context.tile) else { continue };
            let Some(dy) = destination(y, g, input.h, context.tile) else { continue };
            let pixel = input.get(x as i64, y as i64, false);
            let (left, top) = (dx.floor() as i64, dy.floor() as i64);
            let (fx, fy) = ((dx - dx.floor()) as f32, (dy - dy.floor()) as f32);
            for (py, wy) in [(top, 1.0 - fy), (top + 1, fy)] {
                for (px, wx) in [(left, 1.0 - fx), (left + 1, fx)] {
                    let weight = wx * wy;
                    if weight <= 0.0 {
                        continue;
                    }
                    let (px, py) = if context.tile {
                        (px.rem_euclid(input.w as i64), py.rem_euclid(input.h as i64))
                    } else if px < 0 || py < 0 || px >= input.w as i64 || py >= input.h as i64 {
                        continue;
                    } else {
                        (px, py)
                    };
                    let index = py as usize * input.w as usize + px as usize;
                    let Some(total) = weights.get_mut(index) else { continue };
                    let Some(target) = data.get_mut(index * channels..index * channels + channels) else { continue };
                    *total += weight;
                    let ratio = weight as f64 / *total as f64;
                    for (value, source) in target.iter_mut().zip(pixel) {
                        // A running mean in f64 avoids overflowing a sum of finite HDR colors.
                        *value = (*value as f64 + (source as f64 - *value as f64) * ratio) as f32;
                    }
                }
            }
        }
    }
    Ok(vec![output])
}

/// Pixel-center destination after adding a signed displacement in UV units.
fn destination(position: u32, offset: f32, side: u32, tile: bool) -> Option<f64> {
    // Invalid data-map components act as neutral rather than poisoning destination coordinates.
    let offset = if offset.is_finite() { offset as f64 } else { 0.0 };
    // Reduce whole periods before adding the pixel position so huge vectors retain that position.
    let offset = if tile { offset.rem_euclid(1.0) } else { offset };
    let coordinate = position as f64 + offset * side as f64;
    (tile || (coordinate > -1.0 && coordinate < side as f64)).then_some(coordinate)
}
