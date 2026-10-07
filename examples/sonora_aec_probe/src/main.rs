//! Standalone AEC probe. No changes to GLaDOS's production audio pipeline.
//! WAV mode: file RENDER CAPTURE OUTPUT DELAY_MS [bypass]
//! Pipe mode: stream RATE DELAY_MS (LE f32 render frame, then capture frame).
use sonora::config::EchoCanceller;
use sonora::{AudioProcessing, Config, StreamConfig};
use std::error::Error;
use std::io::{self, Read, Write};
use std::time::Instant;

type Result<T> = std::result::Result<T, Box<dyn Error>>;

fn wav(path: &str) -> Result<(u32, Vec<f32>)> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    if spec.channels != 1 {
        return Err("expected mono WAV".into());
    }
    let samples = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<std::result::Result<Vec<_>, _>>()?,
        hound::SampleFormat::Int => {
            let scale = (1_u64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|x| x.map(|v| v as f32 / scale))
                .collect::<std::result::Result<Vec<_>, _>>()?
        }
    };
    if samples.iter().any(|x| !x.is_finite()) {
        return Err("nonfinite input".into());
    }
    Ok((spec.sample_rate, samples))
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let streaming = args.get(1).map(String::as_str) == Some("stream");
    let (rate, render, capture, delay, enabled) = if streaming && args.len() == 4 {
        (
            args[2].parse::<u32>()?,
            Vec::new(),
            Vec::new(),
            args[3].parse::<i32>()?,
            true,
        )
    } else if args.get(1).map(String::as_str) == Some("file")
        && (args.len() == 6 || args.len() == 7)
    {
        let (rate, render) = wav(&args[2])?;
        let (other_rate, capture) = wav(&args[3])?;
        if rate != other_rate || render.len() != capture.len() {
            return Err("WAV rates and lengths must match".into());
        }
        if args.len() == 7 && args[6] != "bypass" {
            return Err("unknown option".into());
        }
        (
            rate,
            render,
            capture,
            args[5].parse::<i32>()?,
            args.len() == 6,
        )
    } else {
        return Err("usage: file RENDER.wav CAPTURE.wav OUTPUT.wav DELAY_MS [bypass] | stream RATE DELAY_MS".into());
    };
    if ![16_000, 32_000, 48_000].contains(&rate) {
        return Err("supported rates: 16000, 32000, 48000".into());
    }
    let stream = StreamConfig::new(rate, 1);
    let frame = stream.num_frames();
    if !streaming && capture.len() % frame != 0 {
        return Err("WAV length must be a multiple of 10 ms".into());
    }
    let mut apm = AudioProcessing::builder()
        .config(Config {
            echo_canceller: enabled.then(EchoCanceller::default),
            ..Default::default()
        })
        .capture_config(stream)
        .render_config(stream)
        .build();
    apm.set_stream_delay_ms(delay)?;
    let mut render_in = vec![0.0; frame];
    let mut capture_in = vec![0.0; frame];
    let mut render_out = vec![0.0; frame];
    let mut capture_out = vec![0.0; frame];
    let mut times = Vec::new();
    let mut bytes = vec![0_u8; frame * 8];
    let mut output_bytes = vec![0_u8; frame * 4];
    let mut stdin = io::stdin().lock();
    let mut stdout = io::stdout().lock();
    let mut writer = if streaming {
        None
    } else {
        Some(hound::WavWriter::create(
            &args[4],
            hound::WavSpec {
                channels: 1,
                sample_rate: rate,
                bits_per_sample: 32,
                sample_format: hound::SampleFormat::Float,
            },
        )?)
    };
    let mut offset = 0;
    loop {
        if streaming {
            let mut read = 0;
            while read < bytes.len() {
                let n = stdin.read(&mut bytes[read..])?;
                if n == 0 {
                    if read != 0 {
                        return Err("truncated pipe frame".into());
                    }
                    break;
                }
                read += n;
            }
            if read == 0 {
                break;
            }
            for (dst, src) in render_in
                .iter_mut()
                .chain(capture_in.iter_mut())
                .zip(bytes.chunks_exact(4))
            {
                *dst = f32::from_le_bytes(src.try_into()?);
                if !dst.is_finite() {
                    return Err("nonfinite pipe input".into());
                }
            }
        } else {
            if offset == capture.len() {
                break;
            }
            render_in.copy_from_slice(&render[offset..offset + frame]);
            capture_in.copy_from_slice(&capture[offset..offset + frame]);
            offset += frame;
        }
        let start = Instant::now();
        apm.process_render_f32(&[&render_in], &mut [&mut render_out])?;
        apm.set_stream_delay_ms(delay)?;
        apm.process_capture_f32(&[&capture_in], &mut [&mut capture_out])?;
        times.push(start.elapsed().as_secs_f64() * 1_000.0);
        if capture_out.iter().any(|x| !x.is_finite()) {
            return Err("nonfinite AEC output".into());
        }
        if let Some(writer) = writer.as_mut() {
            for sample in &capture_out {
                writer.write_sample(*sample)?;
            }
        } else {
            for (src, dst) in capture_out.iter().zip(output_bytes.chunks_exact_mut(4)) {
                dst.copy_from_slice(&src.to_le_bytes());
            }
            stdout.write_all(&output_bytes)?;
            stdout.flush()?;
        }
    }
    if let Some(writer) = writer {
        writer.finalize()?;
    }
    times.sort_by(f64::total_cmp);
    let quantile = |q: f64| {
        times
            .get(((times.len().saturating_sub(1)) as f64 * q) as usize)
            .copied()
    };
    let stats = apm.statistics();
    eprintln!(
        "{}",
        serde_json::json!({
            "rate": rate, "aec": enabled, "frames": times.len(), "frame_ms": 10,
            "mean_processing_ms": times.iter().sum::<f64>() / times.len().max(1) as f64,
            "p95_processing_ms": quantile(0.95), "max_processing_ms": times.last(),
            "delay_hint_ms": delay, "estimated_delay_ms": stats.delay_ms,
            "internal_erle_db": stats.echo_return_loss_enhancement,
            "configuration": "AEC3 defaults (including high-pass); NS/AGC disabled"
        })
    );
    Ok(())
}
