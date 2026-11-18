//! SRT formatting — a straight port of `src/services/srt_formatter.py`.
//!
//! The exact rendering matters: callers diff this text against subtitle files
//! they already hold. `00:00:01,500` (comma decimal, always 3 digits) is SRT's
//! format, not a locale choice.

use crate::models::TranscriptSegment;

/// whisper.cpp reports timestamps in centiseconds. Everything user-visible in
/// this service is in seconds, matching mlx-whisper.
pub const TICKS_PER_SECOND: f64 = 100.0;

pub fn to_srt(segments: &[TranscriptSegment]) -> String {
    let mut out = String::new();
    for (i, seg) in segments.iter().enumerate() {
        if !out.is_empty() {
            out.push('\n');
        }
        out.push_str(&(i + 1).to_string());
        out.push('\n');
        out.push_str(&format!(
            "{} --> {}\n{}\n",
            fmt_timestamp(seg.start),
            fmt_timestamp(seg.end),
            seg.text.trim()
        ));
    }
    out
}

fn fmt_timestamp(seconds: f64) -> String {
    let total_ms = (seconds.max(0.0) * 1000.0).round() as u64;
    let ms = total_ms % 1000;
    let total_s = total_ms / 1000;
    let s = total_s % 60;
    let m = (total_s / 60) % 60;
    let h = total_s / 3600;
    format!("{h:02}:{m:02}:{s:02},{ms:03}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_the_srt_timestamp_shape() {
        assert_eq!(fmt_timestamp(0.0), "00:00:00,000");
        assert_eq!(fmt_timestamp(1.5), "00:00:01,500");
        assert_eq!(fmt_timestamp(3661.25), "01:01:01,250");
    }

    #[test]
    fn renders_segments_with_blank_line_between_cues() {
        let segs = vec![
            TranscriptSegment {
                id: 0,
                start: 0.0,
                end: 1.5,
                text: " hello ".into(),
                ..Default::default()
            },
            TranscriptSegment {
                id: 1,
                start: 1.5,
                end: 3.0,
                text: "world".into(),
                ..Default::default()
            },
        ];
        assert_eq!(
            to_srt(&segs),
            "1\n00:00:00,000 --> 00:00:01,500\nhello\n\n2\n00:00:01,500 --> 00:00:03,000\nworld\n"
        );
    }
}
