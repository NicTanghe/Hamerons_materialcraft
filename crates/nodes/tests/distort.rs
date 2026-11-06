use materialcraft_image::{Image, Kind};
use materialcraft_nodes::{CookContext, find, resolve_params};
use std::collections::BTreeMap;

fn distort(source: &Image, vector: Option<Image>, tile: bool) -> Image {
    let definition = find("distort").expect("Distort must be in the catalog");
    assert!(definition.atomic);
    assert!(definition.params.is_empty(), "the RG vector alone controls displacement");
    let params = resolve_params(definition, &BTreeMap::new());
    let inputs = [Some(source.clone()), vector];
    let vars = BTreeMap::new();
    let resource = |_: &str| None;
    let context = CookContext { w: source.w, h: source.h, seed: 0, tile, inputs: &inputs, params: &params, resource: &resource, vars: &vars };
    (definition.cook)(&context).unwrap().remove(0)
}

fn source(w: u32, h: u32, kind: Kind) -> Image {
    Image::generate(w, h, kind, |x, y| [x as f32, y as f32, (x + y) as f32, (x + 2 * y + 1) as f32 / 32.0])
}

#[test]
fn zero_and_missing_vectors_are_identity() {
    for kind in [Kind::Gray, Kind::Color] {
        let source = source(7, 5, kind);
        for tile in [false, true] {
            assert_eq!(distort(&source, None, tile), source);
            assert_eq!(distort(&source, Some(Image::filled(3, 5, Kind::Color, [0.0, 0.0, 0.8, 0.2])), tile), source);
        }
    }
}

#[test]
fn signed_rg_moves_source_pixels_forward_and_ignores_blue_and_alpha() {
    for kind in [Kind::Gray, Kind::Color] {
        let source = source(8, 4, kind);
        for (vector, dx, dy) in [
            ([0.125, 0.0, 0.7, 0.2], 1, 0),
            ([-0.125, 0.0, 0.2, 0.7], -1, 0),
            ([0.0, 0.25, 0.7, 0.2], 0, 1),
            ([0.0, -0.25, 0.2, 0.7], 0, -1),
            ([0.25, -0.25, 0.8, 0.1], 2, -1),
            ([1.125, -1.25, 0.8, 0.1], 9, -5),
        ] {
            for tile in [false, true] {
                let output = distort(&source, Some(Image::filled(3, 5, Kind::Color, vector)), tile);
                assert_eq!(output.kind, kind);
                for y in 0..4 {
                    for x in 0..8 {
                        let expected = if tile || ((0..8).contains(&(x - dx)) && (0..4).contains(&(y - dy))) {
                            source.get(x - dx, y - dy, tile)
                        } else if kind == Kind::Gray {
                            [0.0, 0.0, 0.0, 1.0]
                        } else {
                            [0.0; 4]
                        };
                        assert_eq!(output.get(x, y, false), expected);
                    }
                }
            }
        }
    }
}

#[test]
fn vector_is_read_at_the_source_pixel_and_overlaps_are_averaged() {
    let source = source(4, 1, Kind::Color);
    // Pixels 0 and 1 land at 1; pixels 2 and 3 land at 2. No source lands at 0 or 3.
    let vector = Image::generate(4, 1, Kind::Color, |x, _| {
        [
            match x {
                0 => 0.25,
                3 => -0.25,
                _ => 0.0,
            },
            0.0,
            1.0,
            1.0,
        ]
    });
    let output = distort(&source, Some(vector), false);
    assert_eq!(output.get(0, 0, false), [0.0; 4]);
    assert_eq!(output.get(3, 0, false), [0.0; 4]);
    assert_eq!(output.get(1, 0, false), [0.5, 0.0, 0.5, 1.5 / 32.0]);
    assert_eq!(output.get(2, 0, false), [2.5, 0.0, 2.5, 3.5 / 32.0]);
}

#[test]
fn fractional_displacement_uses_bilinear_weights_and_uv_units() {
    for (size, shift) in [(8, 0.5), (16, 1.0)] {
        let source = source(size, size, Kind::Color);
        let output = distort(&source, Some(Image::filled(3, 5, Kind::Color, [0.0625, 0.0, 0.0, 0.0])), true);
        assert_eq!(output.get(3, 2, false), [3.0 - shift, 2.0, 5.0 - shift, (8.0 - shift) / 32.0]);
        if size == 8 {
            assert_eq!(output.get(0, 2, false), [3.5, 2.0, 5.5, 8.5 / 32.0]);
        }
    }
}

#[test]
fn hostile_vectors_are_bounded_and_non_finite_components_are_neutral() {
    let source = source(8, 4, Kind::Color);
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let output = distort(&source, Some(Image::filled(8, 4, Kind::Color, [value, value, 0.0, 1.0])), true);
        assert_eq!(output, source);
    }
    for value in [f32::MAX, f32::MIN] {
        for tile in [true, false] {
            let output = distort(&source, Some(Image::filled(8, 4, Kind::Color, [value, value, 0.0, 1.0])), tile);
            assert!(output.data.iter().all(|value| value.is_finite()));
            if tile {
                assert_eq!(output, source, "whole UV periods wrap without losing the source position");
            } else {
                assert!(output.data.iter().all(|value| *value == 0.0));
            }
        }
    }
}

#[test]
fn overlapping_finite_hdr_colors_do_not_overflow() {
    let source = Image::filled(8, 1, Kind::Color, [f32::MAX, -f32::MAX, 0.5, 1.0]);
    let vector = Image::generate(8, 1, Kind::Color, |x, _| [-(x as f32) / 8.0, 0.0, 0.0, 0.0]);
    let output = distort(&source, Some(vector), false);
    assert_eq!(output.get(0, 0, false), source.get(0, 0, false));
    assert!(output.data.iter().all(|value| value.is_finite()));
}

#[test]
#[ignore = "release cook timing; run with --release --ignored --nocapture"]
fn benchmark_distort() {
    for size in [1024, 2048] {
        let source = Image::generate_uv(size, size, Kind::Color, |u, v| [u, v, 0.5, 1.0]);
        let vector = Image::generate_uv(size, size, Kind::Color, |u, v| [0.05 * v, -0.05 * u, 0.0, 0.0]);
        let start = std::time::Instant::now();
        let output = distort(&source, Some(vector), true);
        eprintln!("Distort {size}x{size}: {:?}", start.elapsed());
        assert_eq!((output.w, output.h), (size, size));
        assert!(output.data.iter().all(|value| value.is_finite()));
    }
}
