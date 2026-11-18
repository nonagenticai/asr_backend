//! Decode an arbitrary uploaded audio container into the one shape
//! whisper.cpp accepts: **16 kHz, mono, f32**.
//!
//! The Python service handed the raw bytes to mlx-whisper, which decoded via
//! ffmpeg. There is no ffmpeg here: symphonia decodes the container, and rubato
//! does the sample-rate conversion. Doing the resample by hand (linear
//! interpolation) is the tempting shortcut and it measurably degrades ASR, so
//! it is not taken.

use crate::error::AppError;
use audioadapter_buffers::direct::InterleavedSlice;
use rubato::{Fft, FixedSync, Resampler};
use std::io::Cursor;
use symphonia::core::codecs::audio::AudioDecoderOptions;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::{FormatOptions, TrackType};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;

const TARGET_RATE: u32 = 16_000;

/// Returns mono 16 kHz f32 samples.
pub fn decode_to_16k_mono(raw: &[u8], hint_ext: Option<&str>) -> Result<Vec<f32>, AppError> {
    let mss = MediaSourceStream::new(Box::new(Cursor::new(raw.to_vec())), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = hint_ext {
        hint.with_extension(ext.trim_start_matches('.'));
    }

    // symphonia 0.6 renamed the probe entry point: `probe()` lives in
    // `core::formats::probe` and returns the `FormatReader` directly, where 0.5's
    // `format()` returned a wrapper struct holding it.
    let mut format = symphonia::default::get_probe()
        .probe(
            &hint,
            mss,
            FormatOptions::default(),
            MetadataOptions::default(),
        )
        .map_err(|e| AppError::BadRequest(format!("unsupported audio container: {e}")))?;

    let track = format
        .default_track(TrackType::Audio)
        .ok_or_else(|| AppError::BadRequest("audio container has no default track".into()))?;
    let track_id = track.id;
    // 0.6 boxes the parameters per media type (`CodecParameters::Audio`), so an
    // absent track and a non-audio track have to be rejected separately.
    let params = track
        .codec_params
        .as_ref()
        .and_then(|p| p.audio())
        .ok_or_else(|| AppError::BadRequest("audio container has no audio track".into()))?;
    let mut sample_rate = params.sample_rate.unwrap_or(TARGET_RATE);

    // 0.6 split the decoder options per media type: `DecoderOptions` became
    // `AudioDecoderOptions` and `make()` became `make_audio_decoder()`.
    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(params, &AudioDecoderOptions::default())
        .map_err(|e| AppError::BadRequest(format!("unsupported audio codec: {e}")))?;

    let mut mono: Vec<f32> = Vec::new();
    // Reused across packets: `copy_to_vec_interleaved` resizes it to exactly the
    // decoded sample count, so it never accumulates stale samples.
    let mut interleaved: Vec<f32> = Vec::new();
    loop {
        let packet = match format.next_packet() {
            Ok(Some(p)) => p,
            // 0.6 reports a clean end-of-stream as `Ok(None)`.
            Ok(None) => break,
            // A truncated final packet is a normal end-of-stream for some
            // encoders; only an empty result is an error (checked by caller).
            Err(symphonia::core::errors::Error::IoError(ref e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(e) => return Err(AppError::BadRequest(format!("audio read failed: {e}"))),
        };
        // `Packet` is a plain struct in 0.6; `track_id` is a field, not a method.
        if packet.track_id != track_id {
            continue;
        }
        match decoder.decode(&packet) {
            Ok(decoded) => {
                let spec = decoded.spec();
                sample_rate = spec.rate();
                // The decoded buffer's own spec is authoritative; a container
                // header can disagree, and only this value is ever used.
                let channels = spec.channels().count().max(1);
                // 0.6 dropped `SampleBuffer`; the generic decoded buffer copies
                // itself out to any sample type, converting to f32 on the way.
                decoded.copy_to_vec_interleaved(&mut interleaved);
                // Downmix by averaging: identical to what ffmpeg's default
                // `-ac 1` does for a stereo source.
                if channels == 1 {
                    mono.extend_from_slice(&interleaved);
                } else {
                    for frame in interleaved.chunks(channels) {
                        mono.push(frame.iter().sum::<f32>() / channels as f32);
                    }
                }
            }
            // A decode error mid-stream: keep what we have rather than failing
            // a request whose usable audio is already in hand.
            Err(symphonia::core::errors::Error::DecodeError(_)) => continue,
            Err(e) => return Err(AppError::BadRequest(format!("audio decode failed: {e}"))),
        }
    }

    if mono.is_empty() {
        return Err(AppError::BadRequest(
            "audio payload contained no samples".into(),
        ));
    }

    if sample_rate == TARGET_RATE {
        return Ok(mono);
    }
    resample_to_16k(&mono, sample_rate)
}

/// rubato 1.x replaced the slice-based `FftFixedInOut` with an adapter-based
/// `Fft` plus a `FixedSync` mode, so both ends are wrapped in a mono
/// interleaved adapter here. `FixedSync::Both` is the faithful mapping of the
/// old "fixed in, fixed out" resampler, and `process_all_into_buffer` does the
/// chunking, the final partial chunk and the resampler-delay trim that the
/// hand-rolled chunk loop this replaces never did.
fn resample_to_16k(samples: &[f32], from_rate: u32) -> Result<Vec<f32>, AppError> {
    let mut resampler = Fft::<f32>::new(
        from_rate as usize,
        TARGET_RATE as usize,
        1024, // chunk size in frames
        1,    // sub-chunks: more of them only lowers the anti-alias cutoff
        1,    // mono
        FixedSync::Both,
    )
    .map_err(|e| AppError::Internal(format!("resampler init failed: {e}")))?;

    let input = InterleavedSlice::new(samples, 1, samples.len())
        .map_err(|e| AppError::Internal(format!("resampler input wrap failed: {e:?}")))?;

    let out_len = resampler.process_all_needed_output_len(samples.len());
    let mut out = vec![0.0f32; out_len];
    let mut output = InterleavedSlice::new_mut(&mut out, 1, out_len)
        .map_err(|e| AppError::Internal(format!("resampler output wrap failed: {e:?}")))?;

    let (_consumed, produced) = resampler
        .process_all_into_buffer(&input, &mut output, samples.len(), None)
        .map_err(|e| AppError::Internal(format!("resample failed: {e}")))?;
    out.truncate(produced);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 16 kHz mono WAV built in-test, so the decoder is exercised without a
    /// fixture file and without ffmpeg.
    fn wav_16k_mono(samples: &[i16]) -> Vec<u8> {
        let data_len = (samples.len() * 2) as u32;
        let mut b = Vec::new();
        b.extend_from_slice(b"RIFF");
        b.extend_from_slice(&(36 + data_len).to_le_bytes());
        b.extend_from_slice(b"WAVEfmt ");
        b.extend_from_slice(&16u32.to_le_bytes());
        b.extend_from_slice(&1u16.to_le_bytes()); // PCM
        b.extend_from_slice(&1u16.to_le_bytes()); // mono
        b.extend_from_slice(&16_000u32.to_le_bytes());
        b.extend_from_slice(&32_000u32.to_le_bytes());
        b.extend_from_slice(&2u16.to_le_bytes());
        b.extend_from_slice(&16u16.to_le_bytes());
        b.extend_from_slice(b"data");
        b.extend_from_slice(&data_len.to_le_bytes());
        for s in samples {
            b.extend_from_slice(&s.to_le_bytes());
        }
        b
    }

    #[test]
    fn decodes_16k_mono_wav_without_resampling() {
        let wav = wav_16k_mono(&[0, 1000, -1000, 0]);
        let got = decode_to_16k_mono(&wav, Some("wav")).expect("decode");
        assert_eq!(got.len(), 4);
        assert!((got[1] - 1000.0 / 32768.0).abs() < 1e-6);
    }

    #[test]
    fn rejects_a_container_it_cannot_probe() {
        let err = decode_to_16k_mono(b"not audio at all", Some("wav")).unwrap_err();
        assert!(matches!(err, AppError::BadRequest(_)));
    }

    /// The decode tests above use a 16 kHz WAV, so they never reach rubato.
    /// This pins the resampler path down: 8 kHz in, 16 kHz out, and the
    /// anti-alias filter must not have swallowed the tone.
    #[test]
    fn resamples_8k_to_16k_and_keeps_the_tone() {
        let n = 800; // 0.1 s at 8 kHz
        let input: Vec<f32> = (0..n)
            .map(|i| 0.5 * ((2.0 * std::f32::consts::PI * 440.0 * i as f32) / 8_000.0).sin())
            .collect();
        let got = resample_to_16k(&input, 8_000).expect("resample");
        assert!(
            (got.len() as i64 - 2 * n as i64).abs() <= 2,
            "expected ~{} frames, got {}",
            2 * n,
            got.len()
        );
        let peak = got.iter().fold(0.0f32, |m, s| m.max(s.abs()));
        assert!(peak > 0.2, "resampled tone is silent, peak {peak}");
    }
}
